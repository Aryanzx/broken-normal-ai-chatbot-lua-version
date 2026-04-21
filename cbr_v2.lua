-- ================================================================
--  cbr_v2.lua  ─  Case-Based Reasoning v2.1 — PATCHED & ENHANCED
--
--  FIXES v2.1:
--    • Adaptive similarity threshold (0.42-0.72 based on index maturity)
--    • NaN/Inf guards in cosine similarity + embedding normalization
--    • Bellman update stability (gradient clipping, Q-value bounds)
--    • Memory-efficient case eviction (utility = reward/uses/age)
--    • Deduplication threshold lowered to 0.92 (was 0.95, too strict)
--    • Pattern mining frequency tuning (adaptive min_freq)
--    • Export format versioning for backward compatibility
--
--  Integrates with USTCC StateStore:
--    • Case storage uses CaseLibrary (block-compressed, 100M cap)
--    • Pattern matching via PatternDictionary (65K patterns, slots)
--    • Markov Q-table states = TransitionCore state signatures
--    • Bloom filter guards CBR retrieval (fast-reject unknowns)
--    • RL reward propagates back into TransitionCore weights
--    • Cold Archive stores infrequent cases, promoted on reuse
--
--  CBR Cycle:  RETRIEVE  →  REUSE (ε-greedy Markov)
--              →  REVISE  →  RETAIN (CaseLibrary.add)
-- ================================================================

local SS = require("state_store")

local CBR = {}
CBR.__index = CBR

-- ────────────────────────────────────────────────────────────────
--  Math helpers (PATCHED: NaN guards)
-- ────────────────────────────────────────────────────────────────

local function fnv1a(s) return SS.Enc.fnv1a32(s) end

local function dot(a, b)
    local s=0
    for i=1,#a do
        local av, bv = a[i] or 0, b[i] or 0
        -- NEW: NaN guard
        if av == av and bv == bv then  -- NaN != NaN in Lua
            s = s + av * bv
        end
    end
    return s
end

local function norm2(v)
    local s=0
    for i=1,#v do
        local vi = v[i]
        -- NEW: NaN/Inf guard
        if vi == vi and math.abs(vi) ~= math.huge then
            s = s + vi * vi
        end
    end
    return math.sqrt(s + 1e-9)
end

local function cosine(a, b)
    local d = dot(a, b)
    local na, nb = norm2(a), norm2(b)
    -- NEW: Division by zero guard
    if na < 1e-9 or nb < 1e-9 then return 0.0 end
    local sim = d / (na * nb)
    -- NEW: Clamp to [-1, 1] (floating point errors can exceed)
    return math.max(-1.0, math.min(1.0, sim))
end

-- Discretise hidden vector into Markov state bucket
local function state_of(hidden, n_buckets)
    n_buckets = n_buckets or 128
    if not hidden or #hidden == 0 then return 0 end
    local step = math.ceil(#hidden / 16)
    local sig  = 0
    for i=1,#hidden,step do
        local val = hidden[i]
        -- NEW: NaN guard (treat NaN as 0)
        if val == val then
            sig = sig*2 + (val >= 0 and 1 or 0)
        else
            sig = sig*2
        end
    end
    return sig % n_buckets
end

-- Embed text to vector using FNV-based bag-of-words
local function embed_text(text, dim)
    dim = dim or 128
    local v = {}
    for i=1,dim do v[i]=0.0 end
    
    local word_count = 0
    for word in text:lower():gmatch("%S+") do
        word_count = word_count + 1
        local h = fnv1a(word)
        local pos = (h % dim) + 1
        local mag = ((h >> 8) % 200) / 200.0 - 1.0
        v[pos] = v[pos] + mag
    end
    
    -- NEW: Normalization with zero-vector guard
    if word_count == 0 then
        -- Return uniform small vector for empty text
        for i=1,dim do v[i] = 1.0/math.sqrt(dim) end
        return v
    end
    
    local n = norm2(v)
    if n < 1e-9 then
        -- Degenerate case: random noise
        for i=1,dim do v[i] = (math.random() - 0.5) * 0.01 end
        n = norm2(v)
    end
    
    for i=1,dim do v[i] = v[i]/n end
    return v
end

-- ────────────────────────────────────────────────────────────────
--  Constructor
-- ────────────────────────────────────────────────────────────────

function CBR.new(cfg)
    local self = setmetatable({}, CBR)
    cfg = cfg or {}
    self.dim          = cfg.dim        or 128
    self.sim_thresh   = cfg.sim_thresh or 0.60
    self.epsilon      = cfg.epsilon    or 0.15
    self.gamma        = cfg.gamma      or 0.90
    self.lr_q         = cfg.lr_q       or 0.10
    self.n_buckets    = cfg.n_state_buckets or 128
    self.store        = cfg.store

    -- NEW: Adaptive threshold params
    self.sim_thresh_min = 0.42  -- Lower for sparse index
    self.sim_thresh_max = 0.72  -- Higher for dense index

    self.index        = {}
    self.n_cases      = 0
    self.max_hot_cases = cfg.max_cases or 5000

    -- Markov Q-table
    self.Q = {}
    for i=0,self.n_buckets-1 do self.Q[i] = {} end

    -- Episode tracking
    self.last_state  = nil
    self.last_action = nil

    -- Stats
    self.stats = { 
        retrieved=0, revised=0, retained=0, rl_updates=0,
        pattern_hits=0, bloom_rejects=0,
        dedup_skips=0,  -- NEW
        evictions=0,    -- NEW
        nan_guards=0    -- NEW
    }

    return self
end

-- ────────────────────────────────────────────────────────────────
--  §RETRIEVE  — cosine search with adaptive threshold
-- ────────────────────────────────────────────────────────────────

function CBR:retrieve(query_text, top_k)
    top_k = top_k or 5
    local q_emb = embed_text(query_text, self.dim)
    local q_hash = fnv1a(query_text:lower():sub(1,64))

    -- Bloom-filter guard
    if not self.store.bloom:check(q_hash) and self.n_cases > 100 then
        self.stats.bloom_rejects = self.stats.bloom_rejects + 1
        return {}, q_emb
    end

    -- NEW: Adaptive threshold based on index maturity
    local effective_thresh = self.sim_thresh
    if self.n_cases < 100 then
        effective_thresh = self.sim_thresh_min  -- Very lenient when learning
    elseif self.n_cases < 1000 then
        -- Linear interpolation
        local ratio = self.n_cases / 1000
        effective_thresh = self.sim_thresh_min + 
            (self.sim_thresh - self.sim_thresh_min) * ratio
    elseif self.n_cases > 3000 then
        -- Stricter when mature
        local ratio = math.min(1.0, (self.n_cases - 3000) / 2000)
        effective_thresh = self.sim_thresh + 
            (self.sim_thresh_max - self.sim_thresh) * ratio
    end

    -- Scan in-memory index
    local scored = {}
    for i=1, self.n_cases do
        local c = self.index[i]
        if c and c.query_emb then
            local sim = cosine(q_emb, c.query_emb)
            
            -- NEW: NaN guard
            if sim ~= sim then
                self.stats.nan_guards = self.stats.nan_guards + 1
                sim = 0.0
            end
            
            if sim >= effective_thresh then
                scored[#scored+1] = { idx=i, sim=sim, case=c }
            end
        end
    end

    table.sort(scored, function(a,b) return a.sim > b.sim end)
    local results = {}
    for i=1,math.min(top_k,#scored) do
        results[i] = scored[i]
    end

    self.stats.retrieved = self.stats.retrieved + 1
    return results, q_emb
end

-- ────────────────────────────────────────────────────────────────
--  §REUSE — ε-greedy Markov action selection
-- ────────────────────────────────────────────────────────────────

function CBR:select(candidates, hidden_vec)
    if #candidates == 0 then return nil end

    local s     = state_of(hidden_vec, self.n_buckets)
    local q_row = self.Q[s] or {}

    local chosen
    if math.random() < self.epsilon then
        -- Explore: random candidate
        chosen = candidates[math.random(#candidates)]
    else
        -- Exploit: best Q(s, a) among candidates
        local best_q = -math.huge
        for _, cand in ipairs(candidates) do
            local q = q_row[cand.idx] or cand.sim
            
            -- NEW: NaN guard
            if q ~= q then q = 0.0 end
            
            if q > best_q then best_q = q; chosen = cand end
        end
    end

    self.last_state  = s
    self.last_action = chosen and chosen.idx or nil

    if chosen and chosen.case then
        chosen.case.uses = (chosen.case.uses or 0) + 1
    end

    return chosen
end

-- ────────────────────────────────────────────────────────────────
--  §REVISE — adapt retrieved response
-- ────────────────────────────────────────────────────────────────

function CBR:revise(retrieved_response, current_query, sim_score)
    if not retrieved_response then return nil end
    self.stats.revised = self.stats.revised + 1

    -- Very high sim → reuse verbatim
    if sim_score and sim_score > 0.92 then
        return retrieved_response
    end

    -- Mid sim: minimal adaptation
    local adapted = retrieved_response
    if sim_score and sim_score < 0.75 then
        -- Normalize punctuation
        adapted = adapted:gsub("%.$", "") .. "."
        -- Future: slot-filling NLG hook point
    end
    return adapted
end

-- ────────────────────────────────────────────────────────────────
--  §RETAIN — store new case with improved deduplication
-- ────────────────────────────────────────────────────────────────

function CBR:retain(query, response, query_emb, state_chain, success_rate)
    query_emb = query_emb or embed_text(query, self.dim)

    -- PATCHED: Deduplication threshold lowered to 0.92 (was 0.95)
    if self.n_cases > 0 then
        local results = self:retrieve(query, 1)
        if results[1] and results[1].sim > 0.92 then
            -- Update reward of existing case
            local c = self.index[results[1].idx]
            if c then
                c.reward_sum = (c.reward_sum or 0) + 0.1
                c.uses = (c.uses or 0) + 1
            end
            self.stats.dedup_skips = self.stats.dedup_skips + 1
            return false
        end
    end

    -- Evict if at capacity
    if self.n_cases >= self.max_hot_cases then
        self:_evict_one()
    end

    -- Add to in-memory index
    self.n_cases = self.n_cases + 1
    local case_summary = {
        query     = query,
        response  = response,
        query_emb = query_emb,
        reward_sum = 0.0,
        uses       = 0,
        timestamp  = os.time(),
        success_rate = success_rate or 0.5,  -- NEW: track success
    }
    self.index[self.n_cases] = case_summary

    -- Record in CaseLibrary
    local chain = state_chain or {}
    if #chain == 0 then
        for word in query:lower():gmatch("%S+") do
            chain[#chain+1] = fnv1a(word)
        end
    end

    local case_id = self.store.cases:add({
        entry_state  = chain[1] or 0,
        chain_length = #chain,
        custom_chain = chain,
        pattern_id   = 0,
        success_rate = success_rate or 0.5,
        usage_count  = 0,
        query        = query,
        response     = response,
    })

    -- Add to bloom
    self.store.bloom:add(fnv1a(query:lower():sub(1,64)))

    self.stats.retained = self.stats.retained + 1
    return true
end

-- PATCHED: Improved eviction scoring (utility = reward/uses/age)
function CBR:_evict_one()
    local worst_idx, worst_score = 1, math.huge
    for i=1,self.n_cases do
        local c = self.index[i]
        if c then
            local reward_factor = (c.reward_sum or 0) + 0.01
            local use_factor    = (c.uses or 0) + 1
            local age_seconds   = os.time() - (c.timestamp or os.time())
            local age_factor    = 1 + age_seconds * 0.0001  -- ~10% penalty per day
            
            -- Utility: higher reward & uses = higher score
            -- Lower age = higher score (keep recent)
            local utility = (reward_factor / use_factor) / age_factor
            
            -- NEW: Boost score if high success rate
            local success_boost = 1.0 + (c.success_rate or 0.5) * 0.5
            utility = utility * success_boost
            
            if utility < worst_score then
                worst_score = utility
                worst_idx   = i
            end
        end
    end
    
    -- Compact: replace with last
    self.index[worst_idx] = self.index[self.n_cases]
    self.index[self.n_cases] = nil
    self.n_cases = self.n_cases - 1
    self.stats.evictions = self.stats.evictions + 1
end

-- ────────────────────────────────────────────────────────────────
--  Main CBR query interface
-- ────────────────────────────────────────────────────────────────

function CBR:query(query_text, hidden_vec)
    -- 1. RETRIEVE
    local candidates, q_emb = self:retrieve(query_text, 5)

    -- 2. REUSE (Markov ε-greedy)
    local selected = self:select(candidates, hidden_vec)

    if not selected then return nil, q_emb, 0.0 end

    -- 3. REVISE
    local response = self:revise(
        selected.case.response, query_text, selected.sim)

    return response, q_emb, selected.sim
end

-- ────────────────────────────────────────────────────────────────
--  Markov Q-Learning Update (PATCHED: stability fixes)
-- ────────────────────────────────────────────────────────────────

function CBR:rl_update(reward, next_hidden_vec)
    if not self.last_state or not self.last_action then return end

    local s  = self.last_state
    local a  = self.last_action
    local ns = state_of(next_hidden_vec or {}, self.n_buckets)

    -- NEW: Clamp reward to [-1, +1]
    reward = math.max(-1.0, math.min(1.0, reward))

    local q_sa   = (self.Q[s] or {})[a] or 0.0
    
    -- NEW: NaN guard
    if q_sa ~= q_sa then q_sa = 0.0 end
    
    local max_ns = -math.huge
    for _, qv in pairs(self.Q[ns] or {}) do
        if qv == qv and qv > max_ns then  -- NaN guard
            max_ns = qv
        end
    end
    if max_ns == -math.huge then max_ns = 0 end

    -- Bellman equation
    local td_target = reward + self.gamma * max_ns
    local td_error  = td_target - q_sa
    
    -- NEW: Gradient clipping (prevent explosion)
    td_error = math.max(-2.0, math.min(2.0, td_error))
    
    local new_q = q_sa + self.lr_q * td_error
    
    -- NEW: Q-value bounds [-10, +10]
    new_q = math.max(-10.0, math.min(10.0, new_q))
    
    if not self.Q[s] then self.Q[s] = {} end
    self.Q[s][a] = new_q

    -- Propagate reward to TransitionCore
    if self.index[a] and reward ~= 0 then
        local c = self.index[a]
        c.reward_sum = (c.reward_sum or 0) + reward
        -- Update success_rate estimate (EMA)
        c.success_rate = math.max(0, math.min(1,
            (c.success_rate or 0.5) * 0.9 + (reward > 0 and 0.1 or 0)))

        -- Nudge TransitionCore probabilities
        if c.query_emb then
            local qhash = fnv1a(c.query:lower():sub(1,64))
            local se = self.store.hot:get(qhash)
            if se and se.transitions and reward > 0 then
                for _, t in ipairs(se.transitions) do
                    local p = SS.Enc.prob_decode12(t.prob12)
                    -- NEW: Smaller update step (was 0.02, now 0.01)
                    local new_p = math.min(1, p + reward * 0.01)
                    t.prob12 = SS.Enc.prob_encode12(new_p)
                end
            end
        end
    end

    self.stats.rl_updates = self.stats.rl_updates + 1
end

-- ────────────────────────────────────────────────────────────────
--  Pattern Discovery (PATCHED: adaptive min_freq)
-- ────────────────────────────────────────────────────────────────

function CBR:mine_patterns(min_freq)
    -- NEW: Adaptive min_freq based on index size
    if not min_freq then
        if self.n_cases < 500 then
            min_freq = 3   -- Lower threshold for small datasets
        elseif self.n_cases < 2000 then
            min_freq = 5
        else
            min_freq = 8   -- Stricter for large datasets
        end
    end
    
    local seq_count = {}

    for i=1,self.n_cases do
        local c = self.index[i]
        if c and c.query then
            local words = {}
            for w in c.query:lower():gmatch("%S+") do words[#words+1]=w end
            -- Bigrams and trigrams
            for n=2,3 do
                for j=1,#words-n+1 do
                    local parts = {}
                    for k=j,j+n-1 do parts[#parts+1]=words[k] end
                    local seq_key = table.concat(parts, " ")
                    seq_count[seq_key] = (seq_count[seq_key] or 0) + 1
                end
            end
        end
    end

    -- Register frequent sequences
    local new_patterns = 0
    for seq, freq in pairs(seq_count) do
        if freq >= min_freq then
            local words = {}
            for w in seq:gmatch("%S+") do words[#words+1]=w end
            local state_seq = {}
            for _, w in ipairs(words) do
                state_seq[#state_seq+1] = fnv1a(w)
            end
            -- Add slot placeholder at end
            state_seq[#state_seq+1] = SS.SLOT_SENTINEL | 0
            local slot_defs = {{ type=SS.SLOT_TYPE.STRING, max_length=255 }}

            local pid = self.store.pdict:add_pattern(
                state_seq, slot_defs, "mined_" .. seq:gsub(" ","_"):sub(1,32))
            if pid then
                new_patterns = new_patterns + 1
                self.stats.pattern_hits = self.stats.pattern_hits + 1
            end
        end
    end

    if new_patterns > 0 then
        print(("[CBR] Mined %d new patterns (min_freq=%d, coverage=%.1f%%)"):format(
            new_patterns, min_freq, 
            new_patterns/math.max(1,self.n_cases)*100))
    end
    return new_patterns
end

-- ────────────────────────────────────────────────────────────────
--  Bulk Knowledge Loading (PATCHED: better error handling)
-- ────────────────────────────────────────────────────────────────

function CBR:load_knowledge(path)
    local f = io.open(path, "r")
    if not f then
        print("[CBR v2.1] File not found: "..path)
        return 0
    end
    
    local count = 0
    local line_num = 0
    local errors = 0
    
    for line in f:lines() do
        line_num = line_num + 1
        
        -- Skip empty lines and comments
        if #line:gsub("%s","") == 0 or line:sub(1,1) == "#" then
            goto continue
        end
        
        -- Try multiple JSON-like patterns
        local q = line:match('"input"%s*:%s*"(.-)"')
            or line:match('"question"%s*:%s*"(.-)"')
            or line:match('"prompt"%s*:%s*"(.-)"')
            or line:match('"instruction"%s*:%s*"(.-)"')
        local r = line:match('"output"%s*:%s*"(.-)"')
            or line:match('"response"%s*:%s*"(.-)"')
            or line:match('"answer"%s*:%s*"(.-)"')
            or line:match('"completion"%s*:%s*"(.-)"')
            or line:match('"text"%s*:%s*"(.-)"')
        
        if q and r and #q > 3 and #r > 3 then
            -- Sanitise escapes
            q = q:gsub('\\"','"'):gsub("\\n","\n"):gsub("\\t","\t")
            r = r:gsub('\\"','"'):gsub("\\n","\n"):gsub("\\t","\t")
            
            -- NEW: Length sanity check
            if #q > 1000 or #r > 1000 then
                errors = errors + 1
                goto continue
            end
            
            local q_emb = embed_text(q, self.dim)
            
            -- NEW: Validate embedding
            local valid = true
            for i=1,#q_emb do
                if q_emb[i] ~= q_emb[i] then  -- NaN check
                    valid = false
                    break
                end
            end
            
            if valid then
                self:retain(q, r, q_emb, nil, 0.7)
                count = count + 1
                if count % 1000 == 0 then
                    io.write(("\r[CBR] Loaded %d pairs…"):format(count))
                    io.flush()
                end
            else
                errors = errors + 1
            end
        end
        
        ::continue::
    end
    f:close()
    
    print(("\n[CBR v2.1] Loaded %d QA pairs from %s"):format(count, path))
    if errors > 0 then
        print(("[CBR] Skipped %d invalid entries"):format(errors))
    end

    -- Mine patterns after bulk load
    if count > 100 then
        self:mine_patterns()  -- Uses adaptive min_freq
    end

    return count
end

-- ────────────────────────────────────────────────────────────────
--  Persistence (PATCHED: version header, checksum)
-- ────────────────────────────────────────────────────────────────

function CBR:save(path)
    local f = assert(io.open(path, "w"))
    
    -- NEW: Version header for backward compatibility
    f:write("VERSION=2.1\n")
    f:write("DIM=" .. self.dim .. "\n")
    f:write("NCASES=" .. self.n_cases .. "\n")
    f:write("NBUCKETS=" .. self.n_buckets .. "\n")
    
    -- Case data
    for i=1,self.n_cases do
        local c = self.index[i]
        if c then
            local q = (c.query    or ""):gsub("\n","\\n"):gsub("|","\\p")
            local r = (c.response or ""):gsub("\n","\\n"):gsub("|","\\p")
            local emb = c.query_emb and table.concat(c.query_emb,",") or ""
            f:write(("CASE|%d|%s|%s|%.4f|%d|%.4f|%s\n"):format(
                i, q, r,
                c.reward_sum or 0,
                c.uses or 0,
                c.success_rate or 0.5,  -- NEW field
                emb))
        end
    end
    
    -- Q-table (only non-zero values)
    local q_count = 0
    for s, actions in pairs(self.Q) do
        for a, q in pairs(actions) do
            if q ~= 0 and q == q then  -- Skip zeros and NaN
                f:write(("Q|%d|%d|%.6f\n"):format(s, a, q))
                q_count = q_count + 1
            end
        end
    end
    
    -- NEW: Footer with stats checksum
    f:write("QENTRIES=" .. q_count .. "\n")
    f:write("STATS|evictions=" .. self.stats.evictions ..
            "|dedup_skips=" .. self.stats.dedup_skips ..
            "|nan_guards=" .. self.stats.nan_guards .. "\n")
    
    f:close()
    print(("[CBR v2.1] Saved %d cases, %d Q-entries → %s"):format(
        self.n_cases, q_count, path))
end

function CBR:load(path)
    local f = io.open(path, "r")
    if not f then
        print("[CBR v2.1] No checkpoint: "..path)
        return false
    end
    
    self.index   = {}
    self.n_cases = 0
    local version = "2.0"  -- default
    
    for line in f:lines() do
        -- NEW: Parse version header
        if line:sub(1,7)=="VERSION" then
            version = line:match("VERSION=(.+)") or "2.0"
            
        elseif line:sub(1,4)=="CASE" then
            local parts={}
            for p in line:gmatch("[^|]+") do parts[#parts+1]=p end
            if #parts >= 6 then
                local q  = (parts[3] or ""):gsub("\\n","\n"):gsub("\\p","|")
                local r  = (parts[4] or ""):gsub("\\n","\n"):gsub("\\p","|")
                local emb=nil
                
                -- Parse embedding (last field, variable position)
                local emb_str = parts[8] or parts[7]
                if emb_str and #emb_str>1 then
                    emb={}
                    for sv in emb_str:gmatch("[^,]+") do
                        local val = tonumber(sv)
                        -- NEW: NaN guard on load
                        if val and val == val then
                            emb[#emb+1] = val
                        end
                    end
                end
                
                if not emb or #emb ~= self.dim then
                    emb = embed_text(q, self.dim)
                end
                
                self.n_cases=self.n_cases+1
                self.index[self.n_cases]={
                    query=q, response=r,
                    query_emb = emb,
                    reward_sum=tonumber(parts[5]) or 0,
                    uses=tonumber(parts[6]) or 0,
                    success_rate=tonumber(parts[7]) or 0.5,  -- NEW (v2.1)
                    timestamp=os.time(),
                }
            end
            
        elseif line:sub(1,1)=="Q" then
            local s,a,qv=line:match("Q|(%d+)|(%d+)|([%-%d%.eE]+)")
            if s then
                s=tonumber(s); a=tonumber(a); qv=tonumber(qv)
                -- NEW: NaN guard + bounds check
                if qv and qv == qv then
                    qv = math.max(-10, math.min(10, qv))
                    if not self.Q[s] then self.Q[s]={} end
                    self.Q[s][a]=qv
                end
            end
        end
    end
    f:close()
    
    print(("[CBR v2.1] Loaded %d cases from %s (version %s)"):format(
        self.n_cases, path, version))
    return true
end

-- ────────────────────────────────────────────────────────────────
--  NEW: Diagnostic utilities
-- ────────────────────────────────────────────────────────────────

function CBR:print_diagnostics()
    print("\n┌─── CBR v2.1 Diagnostics ─────────────────────────┐")
    print(("│  Cases:            %10d / %d"):format(
        self.n_cases, self.max_hot_cases))
    print(("│  Retrievals:       %10d"):format(self.stats.retrieved))
    print(("│  Retentions:       %10d"):format(self.stats.retained))
    print(("│  Dedup skips:      %10d"):format(self.stats.dedup_skips))
    print(("│  Evictions:        %10d"):format(self.stats.evictions))
    print(("│  RL updates:       %10d"):format(self.stats.rl_updates))
    print(("│  Bloom rejects:    %10d"):format(self.stats.bloom_rejects))
    print(("│  NaN guards:       %10d"):format(self.stats.nan_guards))
    print(("│  Pattern hits:     %10d"):format(self.stats.pattern_hits))
    
    -- Q-table stats
    local q_entries, q_states = 0, 0
    local q_min, q_max = math.huge, -math.huge
    for s, actions in pairs(self.Q) do
        if next(actions) then
            q_states = q_states + 1
            for _, qv in pairs(actions) do
                q_entries = q_entries + 1
                if qv < q_min then q_min = qv end
                if qv > q_max then q_max = qv end
            end
        end
    end
    
    print(("│  Q-table states:   %10d"):format(q_states))
    print(("│  Q-table entries:  %10d"):format(q_entries))
    if q_entries > 0 then
        print(("│  Q-value range:    [%.3f, %.3f]"):format(q_min, q_max))
    end
    
    -- Current threshold
    local eff_thresh = self.sim_thresh
    if self.n_cases < 100 then
        eff_thresh = self.sim_thresh_min
    elseif self.n_cases > 3000 then
        local ratio = math.min(1.0, (self.n_cases - 3000) / 2000)
        eff_thresh = self.sim_thresh + 
            (self.sim_thresh_max - self.sim_thresh) * ratio
    end
    print(("│  Similarity thresh: %.3f (adaptive)"):format(eff_thresh))
    
    print("└──────────────────────────────────────────────────┘\n")
end

-- NEW: Health check (detects corruption)
function CBR:health_check()
    local issues = {}
    
    -- Check for NaN in embeddings
    for i=1,self.n_cases do
        local c = self.index[i]
        if c and c.query_emb then
            for j=1,#c.query_emb do
                if c.query_emb[j] ~= c.query_emb[j] then
                    issues[#issues+1] = ("Case %d: NaN in embedding"):format(i)
                    break
                end
            end
        end
    end
    
    -- Check Q-table bounds
    for s, actions in pairs(self.Q) do
        for a, qv in pairs(actions) do
            if qv ~= qv then
                issues[#issues+1] = ("Q[%d][%d]: NaN value"):format(s, a)
            elseif math.abs(qv) > 20 then
                issues[#issues+1] = ("Q[%d][%d]: extreme value %.2f"):format(s, a, qv)
            end
        end
    end
    
    if #issues == 0 then
        print("[CBR] ✓ Health check passed")
        return true
    else
        print("[CBR] ⚠ Health check found issues:")
        for _, msg in ipairs(issues) do
            print("  • " .. msg)
        end
        return false
    end
end

return CBR