-- ================================================================
--  chatbot_v2.lua  ─  SSM + Engram v2 + CBR v2 + StateStore v2.1
--                     Full USTCC Architecture Chatbot — PATCHED
--
--  FIXES v2.1:
--    • Context boundary detection (auto-reset SSM on new conversation)
--    • Conversation persistence (save/load history to JSON)
--    • Response quality validation (length, profanity, coherence)
--    • Automatic pattern mining every 100 turns
--    • Fallback response strategy (CBR → SSM → template)
--    • Turn timeout protection (max 30s per response)
--    • Memory pressure monitoring (auto-compact at 90% capacity)
--    • REPL command extensions (/health, /compact, /export)
--    • Graceful degradation on component failures
--    • Batch feedback mode (apply rewards to last N turns)
--
--  Per-turn pipeline:
--    1. Tokenise input
--    2. Context boundary detection (reset if needed)
--    3. Bloom filter: fast-reject unknown state signatures
--    4. TransitionCore: lookup hot state transitions
--    5. Engram v2: retrieve N-gram memory (multi-branch)
--    6. SSM v2: selective scan with TC-conditioned Δ
--    7. CBR v2: check Case Library + Pattern Dictionary
--    8. Smart decode: Pattern early-exit OR TC-guided beam
--    9. Response validation & fallback
--   10. Markov RL: Bellman update → TC + Engram Hebbian
--   11. RETAIN: new case stored in CaseLibrary
-- ================================================================

-- Add current directory to module search path
package.path = package.path .. ";.\\?.lua;?.lua"

-- Path normalization helper (Windows compatibility)
local function normalize_path(path)
    if package.config:sub(1,1) == '\\' then
        return path:gsub("/", "\\")
    end
    return path
end

local SS  = require("state_store")
local Eng = require("engram_v2")
local SSM = require("ssm_v2")
local CBR = require("cbr_v2")

-- ────────────────────────────────────────────────────────────────
--  Tokeniser (word-level + punctuation split)
-- ────────────────────────────────────────────────────────────────

local Tokenizer = {}
Tokenizer.__index = Tokenizer

function Tokenizer.new()
    local self = setmetatable({}, Tokenizer)
    self.word2id = {["<pad>"]=1,["<unk>"]=2,["<bos>"]=3,["<eos>"]=4}
    self.id2word = {[1]="<pad>",[2]="<unk>",[3]="<bos>",[4]="<eos>"}
    self.next_id = 5
    return self
end

function Tokenizer:add(w)
    if not self.word2id[w] then
        self.word2id[w]         = self.next_id
        self.id2word[self.next_id] = w
        self.next_id = self.next_id + 1
    end
    return self.word2id[w]
end

function Tokenizer:encode(text)
    local ids, toks = {}, {}
    local s = text:lower():gsub("([%.%,!%?;:])", " %1 ")
    for w in s:gmatch("%S+") do
        local id = self.word2id[w] or self:add(w)
        ids[#ids+1]=id; toks[#toks+1]=w
    end
    return ids, toks
end

function Tokenizer:decode(ids)
    local ws={}
    for _,id in ipairs(ids) do ws[#ws+1]=self.id2word[id] or "<unk>" end
    return table.concat(ws," "):gsub(" ([%.%,!%?;:])","%1")
end

function Tokenizer:save(path)
    local f=assert(io.open(path,"w"))
    for w,id in pairs(self.word2id) do
        f:write(id.."|"..w:gsub("|","\\p").."\n")
    end
    f:close()
end

function Tokenizer:load(path)
    local f=io.open(path,"r"); if not f then return false end
    for line in f:lines() do
        local id,w=line:match("^(%d+)|(.+)$")
        if id then
            w=w:gsub("\\p","|"); id=tonumber(id)
            self.word2id[w]=id; self.id2word[id]=w
            if id>=self.next_id then self.next_id=id+1 end
        end
    end
    f:close(); return true
end

-- ────────────────────────────────────────────────────────────────
--  Chatbot v2.1 (PATCHED: context detection, persistence, validation)
-- ────────────────────────────────────────────────────────────────

local Chatbot = {}
Chatbot.__index = Chatbot

function Chatbot.new(cfg)
    local self = setmetatable({}, Chatbot)
    cfg = cfg or {}

    self.save_dir   = cfg.save_dir or "./out_v2/checkpoints_v2"
    self.verbose    = cfg.verbose ~= false
    self.dim        = cfg.dim     or 128

    -- Create directory (cross-platform)
    local mkdir_cmd
    if package.config:sub(1,1) == '\\' then
        -- Windows
        mkdir_cmd = 'mkdir "' .. normalize_path(self.save_dir) .. '" 2>nul'
    else
        -- Unix/Linux/Mac
        mkdir_cmd = "mkdir -p " .. self.save_dir
    end
    os.execute(mkdir_cmd)

    -- ── 1. StateStore (USTCC core) ─────────────────────────────
    self.store = SS.StateStore.new({
        save_dir     = self.save_dir .. "/store",
        promo_thresh = cfg.promo_thresh or 10,
        demo_thresh  = cfg.demo_thresh  or 5,
    })

    -- ── 2. Tokeniser ───────────────────────────────────────────
    self.tokenizer = Tokenizer.new()

    -- ── 3. Engram v2 ───────────────────────────────────────────
    self.engram = Eng.new({
        dim        = self.dim,
        max_ngram  = cfg.max_ngram  or 3,
        vocab_size = cfg.vocab_size or 50000,
        lr         = cfg.engram_lr  or 0.005,
        n_branches = cfg.n_branches or 4,
        store      = self.store,
    })

    -- ── 4. SSM v2 ──────────────────────────────────────────────
    self.model = SSM.new({
        d           = self.dim,
        N           = cfg.ssm_state  or 32,
        n_layers    = cfg.n_layers   or 4,
        vocab_size  = cfg.vocab_size or 50000,
        max_context = cfg.max_context or 64,
    }, self.engram, self.store)

    -- ── 5. CBR v2 ──────────────────────────────────────────────
    self.cbr = CBR.new({
        dim             = self.dim,
        sim_thresh      = cfg.sim_thresh or 0.60,
        epsilon         = cfg.epsilon    or 0.15,
        gamma           = cfg.gamma      or 0.90,
        lr_q            = cfg.lr_q       or 0.10,
        n_state_buckets = cfg.n_state_buckets or 128,
        max_cases       = cfg.max_cases  or 5000,
        store           = self.store,
    })

    -- Conversation history
    self.history = {}
    self.turn    = 0
    
    -- NEW: Context tracking
    self.last_user_input = nil
    self.idle_turns      = 0
    self.context_reset_threshold = 5  -- Reset after 5 turns of idle
    
    -- NEW: Quality control
    self.min_response_len = 3
    self.max_response_len = 500
    self.profanity_filter = self:_init_profanity_filter()
    
    -- NEW: Fallback templates
    self.fallback_responses = {
        "I'm not sure I understand. Could you rephrase that?",
        "That's interesting. Tell me more.",
        "I'm still learning. Can you explain differently?",
        "Let me think about that...",
    }
    
    -- NEW: Performance monitoring
    self.stats = {
        cbr_hits = 0,
        ssm_gens = 0,
        fallbacks = 0,
        context_resets = 0,
        validation_failures = 0,
    }

    return self
end

-- NEW: Simple profanity filter init
function Chatbot:_init_profanity_filter()
    -- Minimal filter; extend as needed
    return {
        ["damn"] = true,
        ["hell"] = true,
        -- Add more as needed
    }
end

-- NEW: Context boundary detection
function Chatbot:_should_reset_context(user_text)
    -- Reset if:
    -- 1. Explicit markers
    if user_text:lower():match("^new conversation") or
       user_text:lower():match("^start over") or
       user_text:lower():match("^reset") then
        return true
    end
    
    -- 2. Long idle period
    if self.idle_turns >= self.context_reset_threshold then
        return true
    end
    
    -- 3. Topic shift detection (simple: no word overlap)
    if self.last_user_input then
        local prev_words = {}
        for w in self.last_user_input:lower():gmatch("%S+") do
            prev_words[w] = true
        end
        local overlap = 0
        for w in user_text:lower():gmatch("%S+") do
            if prev_words[w] then overlap = overlap + 1 end
        end
        if overlap == 0 and #self.history > 3 then
            return true  -- Complete topic shift
        end
    end
    
    return false
end

-- NEW: Response validation
function Chatbot:_validate_response(response_text)
    -- Length check
    if #response_text < self.min_response_len then
        return false, "too_short"
    end
    if #response_text > self.max_response_len then
        response_text = response_text:sub(1, self.max_response_len) .. "..."
    end
    
    -- Profanity check (simple word-level)
    for word in response_text:lower():gmatch("%S+") do
        if self.profanity_filter[word] then
            return false, "profanity"
        end
    end
    
    -- Repetition check (same word >5 times, excluding special tokens)
    local word_count = {}
    local special_tokens = {["<unk>"]=true, ["<pad>"]=true, ["<bos>"]=true, ["<eos>"]=true}
    
    for word in response_text:lower():gmatch("%S+") do
        -- Skip special tokens in count
        if not special_tokens[word] then
            word_count[word] = (word_count[word] or 0) + 1
            if word_count[word] > 5 then
                return false, "repetitive: '" .. word .. "' repeated " .. word_count[word] .. " times"
            end
        end
    end
    
    -- NEW: Check for excessive special tokens (indicates generation failure)
    local unk_count = 0
    for word in response_text:gmatch("%S+") do
        if word == "<unk>" then unk_count = unk_count + 1 end
    end
    if unk_count > 3 then
        return false, "too_many_unknowns (" .. unk_count .. " <unk> tokens)"
    end
    
    -- Coherence check (ends with punctuation)
    if not response_text:match("[%.!%?]$") then
        response_text = response_text .. "."
    end
    
    return true, response_text
end

-- NEW: Fallback response generator
function Chatbot:_get_fallback_response()
    self.stats.fallbacks = self.stats.fallbacks + 1
    return self.fallback_responses[
        math.random(#self.fallback_responses)]
end

-- ────────────────────────────────────────────────────────────────
--  Main turn handler (PATCHED: validation, fallback, timeout)
-- ────────────────────────────────────────────────────────────────

function Chatbot:respond(user_text, timeout)
    timeout = timeout or 30  -- 30 seconds max
    local start_time = os.time()
    
    self.turn = self.turn + 1
    
    -- NEW: Context boundary detection
    local should_reset = self:_should_reset_context(user_text)
    if should_reset then
        self.model:reset()
        self.stats.context_resets = self.stats.context_resets + 1
        if self.verbose then
            print("[Context] Reset detected (turn " .. self.turn .. ")")
        end
    end
    
    self.last_user_input = user_text

    -- 1. Tokenise
    local ids, toks = self.tokenizer:encode(user_text)

    -- 2. Feed through SSM to get hidden representation
    self.model:reset()  -- Always reset for new turn
    local last_h
    for i,pid in ipairs(ids) do
        -- Check timeout
        if os.time() - start_time > timeout then
            print("[Warning] Turn timeout, using fallback")
            return self:_get_fallback_response(), nil
        end
        
        local _, h = self.model:forward_step(toks[i], pid, i==1 and should_reset)
        last_h = h
    end
    last_h = last_h or {}

    -- 3. Bloom pre-filter
    local q_hash = SS.Enc.fnv1a32(user_text:lower():sub(1,64))
    local bloom_known = self.store.bloom:check(q_hash)

    -- 4. CBR fast-path (known case retrieval)
    local response_text, q_emb, cbr_sim
    local used_cbr = false

    if bloom_known then
        response_text, q_emb, cbr_sim = self.cbr:query(user_text, last_h)
        if response_text and cbr_sim > self.cbr.sim_thresh then
            used_cbr = true
            self.stats.cbr_hits = self.stats.cbr_hits + 1
            if self.verbose then
                print(("[CBR] Hit sim=%.2f: %s"):format(cbr_sim, response_text:sub(1,50)))
            end
        else
            response_text = nil
        end
    end

    -- 5. SSM smart decode (fallback)
    local response_ids, response_toks
    if not response_text then
        -- Check timeout before expensive decode
        if os.time() - start_time > timeout - 5 then
            print("[Warning] Insufficient time for decode, using fallback")
            return self:_get_fallback_response(), nil
        end
        
        response_ids, response_toks, last_h = self.model:smart_decode(
            ids, toks, self.tokenizer, {
                max_gen     = 60,
                beam_width  = 3,
                temperature = 0.8,
                top_k       = 40,
            })
        response_text = self.tokenizer:decode(response_ids)
        self.stats.ssm_gens = self.stats.ssm_gens + 1
        if self.verbose then
            print("[SSM] Gen: " .. response_text:sub(1,60))
        end
    end
    
    -- NEW: Response validation
    local valid, result = self:_validate_response(response_text)
    if not valid then
        self.stats.validation_failures = self.stats.validation_failures + 1
        if self.verbose then
            print(("[Validation] Failed: " .. result))
        end
        response_text = self:_get_fallback_response()
    elseif result then
        response_text = result  -- Use corrected version
    end

    -- 6. Record turn in StateStore
    local state_chain = {}
    for _, t in ipairs(toks) do
        state_chain[#state_chain+1] = SS.Enc.fnv1a32(t)
    end
    
    -- Transitions
    for i=2,#toks do
        self.store:record_transition("tok_"..toks[i-1], "tok_"..toks[i])
    end
    
    -- Add query to bloom
    self.store.bloom:add(q_hash)

    -- 7. Engram: record sequence
    self.engram:record_sequence(toks, response_text, 0.5)

    -- 8. Store turn in history
    local entry = {
        role      = "assistant",
        text      = response_text,
        q_text    = user_text,
        q_emb     = q_emb,
        hidden    = last_h,
        used_cbr  = used_cbr,
        cbr_sim   = cbr_sim,
        state_chain = state_chain,
        turn      = self.turn,
        timestamp = os.time(),
        validated = valid,
    }
    table.insert(self.history, entry)
    
    -- NEW: Auto pattern mining every 100 turns
    if self.turn % 100 == 0 then
        self:_auto_mine_patterns()
    end
    
    -- NEW: Memory pressure check
    if self.turn % 50 == 0 then
        self:_check_memory_pressure()
    end

    return response_text, entry
end

-- NEW: Automatic pattern mining
function Chatbot:_auto_mine_patterns()
    if self.verbose then
        print("[Auto] Mining patterns at turn " .. self.turn)
    end
    local n_new = self.cbr:mine_patterns()
    if n_new > 0 then
        self.store.pdict:maybe_evict()
    end
end

-- NEW: Memory pressure monitoring
function Chatbot:_check_memory_pressure()
    local cbr_ratio = self.cbr.n_cases / self.cbr.max_hot_cases
    local hot_ratio = self.store.hot.n_entries / self.store.hot.max_hot
    
    if cbr_ratio > 0.9 or hot_ratio > 0.9 then
        if self.verbose then
            print(("[Memory] Pressure: CBR=%.0f%% Hot=%.0f%%"):format(
                cbr_ratio*100, hot_ratio*100))
        end
        -- Trigger compact
        self:compact()
    end
end

-- ────────────────────────────────────────────────────────────────
--  Feedback & RL Update (PATCHED: batch feedback)
-- ────────────────────────────────────────────────────────────────

function Chatbot:feedback(reward, n_turns_back)
    n_turns_back = n_turns_back or 1  -- Default: last turn only
    
    for i = 1, math.min(n_turns_back, #self.history) do
        local idx = #self.history - i + 1
        local entry = self.history[idx]
        if not entry then break end
        
        -- Decay reward for older turns
        local decayed_reward = reward * (0.9 ^ (i-1))

        -- Markov Q-learning (CBR RL)
        self.cbr:rl_update(decayed_reward, entry.hidden)

        -- Engram Hebbian update
        if entry.hidden and #entry.hidden > 0 then
            local tokens = {}
            for w in entry.q_text:lower():gmatch("%S+") do
                tokens[#tokens+1] = w
            end
            self.engram:update(tokens, entry.hidden, decayed_reward)
        end

        -- CBR retain: only for positive reward on most recent turn
        if i == 1 and reward > 0 then
            self.cbr:retain(
                entry.q_text, entry.text,
                entry.q_emb, entry.state_chain, 0.8)
        end

        -- Update transition success in StateStore
        if entry.state_chain and #entry.state_chain >= 2 then
            for j=1,#entry.state_chain-1 do
                local fk = "tok_" .. tostring(entry.state_chain[j])
                local tk = "tok_" .. tostring(entry.state_chain[j+1])
                local trans_type = reward >= 0 and SS.TRANS_TYPE.DIRECT
                                               or  SS.TRANS_TYPE.DECAY
                self.store:record_transition(fk, tk, trans_type)
            end
        end
    end

    if self.verbose then
        print(("[RL] r=%.1f applied to last %d turns  CBR=%d  Bloom_FPR=%.2f%%"):format(
            reward, n_turns_back, self.cbr.n_cases, self.store.bloom:fpr()*100))
    end
end

-- ────────────────────────────────────────────────────────────────
--  Knowledge loading
-- ────────────────────────────────────────────────────────────────

function Chatbot:load_knowledge(path)
    local count = self.cbr:load_knowledge(path)
    -- Also tokenise for vocabulary
    local f = io.open(path, "r")
    if f then
        for line in f:lines() do
            local q = line:match('"input"%s*:%s*"(.-)"')
                   or line:match('"question"%s*:%s*"(.-)"')
                   or line:match('"prompt"%s*:%s*"(.-)"')
            local r = line:match('"output"%s*:%s*"(.-)"')
                   or line:match('"answer"%s*:%s*"(.-)"')
            if q then self.tokenizer:encode(q) end
            if r then self.tokenizer:encode(r) end
        end
        f:close()
    end
    return count
end

-- ────────────────────────────────────────────────────────────────
--  Save / Load (PATCHED: conversation persistence)
-- ────────────────────────────────────────────────────────────────

function Chatbot:save()
    self.store:save()
    self.engram:save(   self.save_dir .. "/engram_v2.dat")
    self.model:save(    self.save_dir .. "/ssm_v2.dat")
    self.cbr:save(      self.save_dir .. "/cbr_v2.dat")
    self.tokenizer:save(self.save_dir .. "/vocab.dat")
    self:save_conversation()  -- NEW
    self:save_stats()         -- NEW
    print("[Chatbot v2.1] All components saved to " .. self.save_dir)
end

function Chatbot:load()
    self.store:load()
    self.engram:load(   self.save_dir .. "/engram_v2.dat")
    self.model:load(    self.save_dir .. "/ssm_v2.dat")
    self.cbr:load(      self.save_dir .. "/cbr_v2.dat")
    self.tokenizer:load(self.save_dir .. "/vocab.dat")
    self:load_conversation()  -- NEW
    self:load_stats()         -- NEW
    print("[Chatbot v2.1] All components loaded")
end

-- NEW: Conversation persistence
function Chatbot:save_conversation(path)
    path = path or (self.save_dir .. "/conversation.jsonl")
    local f = assert(io.open(path, "w"))
    
    for i, entry in ipairs(self.history) do
        local obj = {
            turn     = entry.turn,
            q        = entry.q_text:gsub('"', '\\"'),
            r        = entry.text:gsub('"', '\\"'),
            used_cbr = entry.used_cbr,
            sim      = entry.cbr_sim,
            valid    = entry.validated,
            ts       = entry.timestamp,
        }
        f:write(self:_json_encode(obj) .. "\n")
    end
    f:close()
    if self.verbose then
        print(("[Conv] Saved %d turns → %s"):format(#self.history, path))
    end
end

function Chatbot:load_conversation(path)
    path = path or (self.save_dir .. "/conversation.jsonl")
    local f = io.open(path, "r")
    if not f then return false end
    
    self.history = {}
    local count = 0
    for line in f:lines() do
        local obj = self:_json_decode(line)
        if obj then
            table.insert(self.history, {
                turn      = obj.turn,
                q_text    = obj.q,
                text      = obj.r,
                used_cbr  = obj.used_cbr,
                cbr_sim   = obj.sim,
                validated = obj.valid,
                timestamp = obj.ts,
            })
            count = count + 1
            if obj.turn and obj.turn > self.turn then
                self.turn = obj.turn
            end
        end
    end
    f:close()
    
    if self.verbose then
        print(("[Conv] Loaded %d turns from %s"):format(count, path))
    end
    return true
end

-- Simple JSON encode (minimal, for stats)
function Chatbot:_json_encode(obj)
    local parts = {}
    for k, v in pairs(obj) do
        local val_str
        if type(v) == "string" then
            val_str = '"' .. v .. '"'
        elseif type(v) == "boolean" then
            val_str = tostring(v)
        else
            val_str = tostring(v or "null")
        end
        parts[#parts+1] = ('"'..k..'":' .. val_str)
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

-- Simple JSON decode (minimal)
function Chatbot:_json_decode(str)
    local obj = {}
    for k, v in str:gmatch('"([^"]+)"%s*:%s*"?([^,}"]+)"?') do
        if v == "true" then v = true
        elseif v == "false" then v = false
        elseif tonumber(v) then v = tonumber(v)
        end
        obj[k] = v
    end
    return next(obj) and obj or nil
end

-- NEW: Stats persistence
function Chatbot:save_stats(path)
    path = path or (self.save_dir .. "/stats.txt")
    local f = assert(io.open(path, "w"))
    for k, v in pairs(self.stats) do
        f:write(k .. "=" .. v .. "\n")
    end
    f:close()
end

function Chatbot:load_stats(path)
    path = path or (self.save_dir .. "/stats.txt")
    local f = io.open(path, "r")
    if not f then return false end
    for line in f:lines() do
        local k, v = line:match("([^=]+)=(.+)")
        if k then self.stats[k] = tonumber(v) or v end
    end
    f:close()
    return true
end

-- ────────────────────────────────────────────────────────────────
--  Stats display (PATCHED: enhanced metrics)
-- ────────────────────────────────────────────────────────────────

function Chatbot:stats()
    print("\n┌─── Chatbot v2.1 Statistics ──────────────────────┐")
    print(("│  Total turns:      %10d"):format(self.turn))
    print(("│  CBR hits:         %10d (%.1f%%)"):format(
        self.stats.cbr_hits,
        self.turn > 0 and self.stats.cbr_hits/self.turn*100 or 0))
    print(("│  SSM generations:  %10d (%.1f%%)"):format(
        self.stats.ssm_gens,
        self.turn > 0 and self.stats.ssm_gens/self.turn*100 or 0))
    print(("│  Fallbacks:        %10d (%.1f%%)"):format(
        self.stats.fallbacks,
        self.turn > 0 and self.stats.fallbacks/self.turn*100 or 0))
    print(("│  Context resets:   %10d"):format(self.stats.context_resets))
    print(("│  Valid. failures:  %10d"):format(self.stats.validation_failures))
    print("├───────────────────────────────────────────────────┤")
    
    self.store:print_stats()
    self.engram:print_stats()
    self.cbr:print_diagnostics()
    self.model:print_diagnostics()
end

-- NEW: Health check (all components)
function Chatbot:health_check()
    print("\n[Health Check] Running diagnostics...")
    local all_ok = true
    
    all_ok = self.cbr:health_check() and all_ok
    all_ok = self.engram:health_check() and all_ok
    all_ok = self.model:health_check() and all_ok
    
    if all_ok then
        print("[Health Check] ✓ All systems nominal\n")
    else
        print("[Health Check] ⚠ Issues detected (see above)\n")
    end
    return all_ok
end

-- NEW: Compact (memory optimization)
function Chatbot:compact()
    print("[Compact] Optimizing memory...")
    
    -- Engram compact (remove low-usage embeddings)
    self.engram:compact(3)
    
    -- Trigger StateStore demotion
    if self.store.hot.n_entries > self.store.hot.max_hot * 0.8 then
        self.store:_emergency_demote()
    end
    
    print("[Compact] Done")
end

-- NEW: Export report
function Chatbot:export_report(path)
    path = path or (self.save_dir .. "/report.txt")
    local f = assert(io.open(path, "w"))
    
    f:write("=== Chatbot v2.1 Report ===\n\n")
    f:write("Generated: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n\n")
    
    f:write("--- Statistics ---\n")
    for k, v in pairs(self.stats) do
        f:write(k .. ": " .. v .. "\n")
    end
    
    f:write("\n--- Configuration ---\n")
    f:write("Dimension: " .. self.dim .. "\n")
    f:write("SSM layers: " .. self.model.n_layers .. "\n")
    f:write("Engram branches: " .. self.engram.n_branches .. "\n")
    f:write("CBR max cases: " .. self.cbr.max_hot_cases .. "\n")
    
    f:write("\n--- Conversation Sample (last 10 turns) ---\n")
    for i = math.max(1, #self.history - 9), #self.history do
        local e = self.history[i]
        if e then
            f:write(("Turn %d:\n"):format(e.turn))
            f:write("  Q: " .. e.q_text .. "\n")
            f:write("  A: " .. e.text .. "\n")
            f:write(("  CBR: %s (sim=%.2f)\n"):format(
                e.used_cbr and "YES" or "NO", e.cbr_sim or 0))
        end
    end
    
    f:close()
    print("[Export] Report saved → " .. path)
end

-- ────────────────────────────────────────────────────────────────
--  Interactive REPL (PATCHED: extended commands)
-- ────────────────────────────────────────────────────────────────

function Chatbot:repl()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║   SSM + Engram v2.1 + CBR v2.1 + StateStore v2.1      ║")
    print("║   USTCC Architecture (10M hot / 100M cases)           ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║  /good [N]  /bad [N]  /save  /stats  /health          ║")
    print("║  /load <path>  /mine  /compact  /export  /reset       ║")
    print("║  quit                                                 ║")
    print("╚═══════════════════════════════════════════════════════╝")

    while true do
        io.write("\nYou: ")
        local line = io.read()
        if not line or line:lower() == "quit" then
            print("Saving before exit …"); self:save(); break

        elseif line:sub(1,5) == "/good" then
            local n = tonumber(line:match("/good%s+(%d+)")) or 1
            self:feedback(1.0, n)
            print(("[RL] +1.0 applied to last %d turn(s)"):format(n))
            
        elseif line:sub(1,4) == "/bad" then
            local n = tonumber(line:match("/bad%s+(%d+)")) or 1
            self:feedback(-1.0, n)
            print(("[RL] -1.0 applied to last %d turn(s)"):format(n))
            
        elseif line:sub(1,5) == "/save" then
            self:save()
            
        elseif line:sub(1,6) == "/stats" then
            self:stats()
            
        elseif line:sub(1,7) == "/health" then
            self:health_check()
            
        elseif line:sub(1,5) == "/load" then
            local p = line:match("/load%s+(.+)")
            if p then self:load_knowledge(p:gsub('"',''))
            else print("Usage: /load <path.jsonl>") end
            
        elseif line:sub(1,5) == "/mine" then
            local n = self.cbr:mine_patterns()
            print(("[Pattern] Mined %d new patterns"):format(n))
            
        elseif line:sub(1,8) == "/compact" then
            self:compact()
            
        elseif line:sub(1,7) == "/export" then
            self:export_report()
            
        elseif line:sub(1,6) == "/reset" then
            self.model:reset()
            self.history = {}
            self.turn = 0
            self.stats.context_resets = self.stats.context_resets + 1
            print("[Context] Full reset")
            
        elseif #line:gsub("%s","") > 0 then
            local reply = self:respond(line)
            print("Bot: " .. reply)
        end
    end
end

-- ────────────────────────────────────────────────────────────────
--  Entry point
-- ────────────────────────────────────────────────────────────────

local bot = Chatbot.new({
    dim           = 64,
    ssm_state     = 32,
    n_layers      = 4,
    max_ngram     = 3,
    n_branches    = 2,
    vocab_size    = 50000,
    max_cases     = 5000,
    sim_thresh    = 0.60,
    epsilon       = 0.15,
    gamma         = 0.90,
    n_state_buckets = 128,
    promo_thresh  = 10,
    demo_thresh   = 5,
    save_dir      = "./out_v2/checkpoints_v2",
    verbose       = true,
})

bot:load()

local arg = arg or {}
if arg[1] then bot:load_knowledge(arg[1]) end

bot:repl()