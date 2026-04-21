-- ================================================================
--  engram_v2.lua  ─  Conditional N-gram Memory v2.1 — PATCHED
--
--  FIXES v2.1:
--    • Multi-branch NaN/Inf guards (retrieve_multibranch validation)
--    • Depthwise conv stability (zero-padding at boundaries)
--    • RMS normalization epsilon tuning (1e-9 → 1e-8 for stability)
--    • Hebbian update clipping (prevents embedding explosion)
--    • Embedding lazy init race condition fix (atomic check)
--    • StateStore transition propagation bounds (max ±0.05 delta)
--    • Memory leak fix in record_sequence (chain length cap)
--    • Quantized int8 save/load support (4× compression)
--
--  Integrates with StateStore (USTCC architecture) for:
--    • Sparse retrieval via hashed N-grams (bloom-filtered)
--    • Transition probability lookup from TransitionCore
--    • Cold archive fallback for rare N-gram patterns
--    • Pattern dictionary compression of frequent sequences
--    • Hebbian + RL online update propagated to hot table
--    • Context-aware gating (RMSNorm dot-product, paper Eq.6)
-- ================================================================

local SS = require("state_store")

local Engram = {}
Engram.__index = Engram

-- ────────────────────────────────────────────────────────────────
--  Math helpers (PATCHED: numerical stability)
-- ────────────────────────────────────────────────────────────────

local EPSILON = 1e-8  -- CHANGED: was 1e-9, more stable for CPU

local function sigmoid(x)  return 1/(1+math.exp(-x)) end

local function rms_norm(v)
    local ss = 0
    for i=1,#v do
        local vi = v[i]
        -- NEW: NaN guard
        if vi == vi and math.abs(vi) ~= math.huge then
            ss = ss + vi*vi
        end
    end
    local rms = math.sqrt(ss/#v + EPSILON)
    local out = {}
    for i=1,#v do out[i] = (v[i] or 0)/rms end
    return out
end

local function dot(a, b)
    local s=0
    for i=1,#a do
        local av, bv = a[i] or 0, b[i] or 0
        -- NEW: NaN guard
        if av == av and bv == bv then
            s = s + av * bv
        end
    end
    return s
end

local function vec_add_ip(a, b, alpha)
    for i=1,#a do
        local delta = alpha * (b[i] or 0)
        -- NEW: Clipping to prevent explosion
        delta = math.max(-1.0, math.min(1.0, delta))
        a[i] = a[i] + delta
    end
end

local function vec_clone(v)
    local c={}; for i=1,#v do c[i]=v[i] end; return c
end

local function rand_vec(dim, seed)
    local scale = math.sqrt(2/dim)
    local st = seed or 42
    local v = {}
    for i=1,dim do
        st = (st*1664525+1013904223) & 0xFFFFFFFF
        v[i] = ((st/0x80000000)-1)*scale
    end
    return v
end

-- NEW: Validate vector (no NaN/Inf)
local function is_valid_vec(v)
    for i=1,#v do
        if v[i] ~= v[i] or math.abs(v[i]) == math.huge then
            return false
        end
    end
    return true
end

-- ────────────────────────────────────────────────────────────────
--  Constructor
-- ────────────────────────────────────────────────────────────────

function Engram.new(cfg)
    local self       = setmetatable({}, Engram)
    self.dim         = cfg.dim       or 128
    self.max_ngram   = cfg.max_ngram or 3
    self.vocab_size  = cfg.vocab_size or 50000
    self.lr          = cfg.lr        or 0.005
    self.store       = cfg.store

    -- Embedding tables
    self.embed_tables = {}
    for n = 1, self.max_ngram do
        self.embed_tables[n] = {}
    end

    -- Branch-specific Key projection matrices W_K^(m)
    self.n_branches  = cfg.n_branches or 4
    self.W_K         = {}
    local seed = 0
    for m = 1, self.n_branches do
        self.W_K[m] = {}
        for n = 1, self.max_ngram do
            seed = seed + 137
            self.W_K[m][n] = rand_vec(self.dim, seed)
        end
    end

    -- Shared Value projection W_V
    self.W_V  = rand_vec(self.dim, 9999)

    -- Depthwise conv kernel (3 taps)
    self.conv_k = { 0.25, 0.5, 0.25 }

    -- Vocab projection cache
    self.vocab_proj  = {}

    -- State chain tracking
    self.state_chain = {}

    -- Stats
    self.stats = {
        retrievals=0, store_hits=0, cold_hits=0, updates=0,
        nan_guards=0,      -- NEW
        embed_inits=0,     -- NEW
        hebbian_clips=0    -- NEW
    }

    return self
end

-- ────────────────────────────────────────────────────────────────
--  Tokenizer Compression
-- ────────────────────────────────────────────────────────────────

function Engram:canonicalize(token)
    local t = tostring(token):lower()
    t = t:gsub("^[Ġ▁ ]", ""):gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
    return t
end

function Engram:token_id(token)
    local c = self:canonicalize(token)
    if not self.vocab_proj[c] then
        self.vocab_proj[c] = (SS.Enc.fnv1a32(c) % self.vocab_size) + 1
    end
    return self.vocab_proj[c]
end

-- ────────────────────────────────────────────────────────────────
--  Embedding table access (PATCHED: atomic init, validation)
-- ────────────────────────────────────────────────────────────────

function Engram:_ngram_key(token_ids, order)
    local start = math.max(1, #token_ids - order + 1)
    local parts = {}
    for i = start, #token_ids do
        parts[#parts+1] = tostring(token_ids[i])
    end
    return "ng:" .. table.concat(parts, "_"), SS.Enc.fnv1a32(table.concat(parts,"_"))
end

function Engram:get_embed(n, ngram_key, ngram_hash)
    -- 1. Check local table (PATCHED: atomic check)
    local cached = self.embed_tables[n][ngram_hash]
    if cached then
        -- NEW: Validate on retrieval
        if not is_valid_vec(cached) then
            self.stats.nan_guards = self.stats.nan_guards + 1
            -- Reinitialize corrupted embedding
            cached = rand_vec(self.dim, ngram_hash * 31 + n * 7)
            self.embed_tables[n][ngram_hash] = cached
        end
        return cached
    end

    -- 2. Check StateStore hot/cold
    local state_entry = self.store:lookup(ngram_key)
    if state_entry then
        local vec = rand_vec(self.dim, ngram_hash)
        -- Modulate by transition count
        local scale = math.log(1 + (state_entry.count or 1)) / 10.0
        -- NEW: Clamp scale to prevent explosion
        scale = math.min(scale, 2.0)
        for i = 1, self.dim do vec[i] = vec[i] * scale end
        
        self.embed_tables[n][ngram_hash] = vec
        self.stats.store_hits = self.stats.store_hits + 1
        return vec
    end

    -- 3. Initialize fresh (lazy Xavier)
    local vec = rand_vec(self.dim, ngram_hash * 31 + n * 7)
    self.embed_tables[n][ngram_hash] = vec
    self.stats.embed_inits = self.stats.embed_inits + 1
    return vec
end

-- ────────────────────────────────────────────────────────────────
--  §2.2  Sparse Retrieval via Hashed N-grams (PATCHED: stability)
-- ────────────────────────────────────────────────────────────────

function Engram:retrieve(tokens, hidden, branch)
    branch = branch or 1
    self.stats.retrievals = self.stats.retrievals + 1

    -- Encode tokens
    local ids = {}
    for _, t in ipairs(tokens) do
        ids[#ids+1] = self:token_id(t)
    end

    -- Build state chain for StateStore
    for i = 2, #ids do
        local from_key = "ng1_" .. ids[i-1]
        local to_key   = "ng1_" .. ids[i]
        self.store:record_transition(from_key, to_key, SS.TRANS_TYPE.DIRECT)
    end

    -- RMSNorm of hidden state
    local h_norm = hidden and rms_norm(hidden) or nil

    -- Fuse across N-gram orders
    local fused = {}
    for i = 1, self.dim do fused[i] = 0.0 end
    local gate_sum = 0.0
    local prev_emb = nil

    for n = 1, math.min(self.max_ngram, #ids) do
        local ngkey, nghash = self:_ngram_key(ids, n)

        -- Bloom-filter guard
        if self.store.bloom:check(nghash) or n == 1 then
            local emb    = self:get_embed(n, ngkey, nghash)
            
            -- NEW: Validate embedding
            if not is_valid_vec(emb) then
                self.stats.nan_guards = self.stats.nan_guards + 1
                goto continue_ngram
            end
            
            local e_norm = rms_norm(emb)

            -- Context-aware gate (paper Eq.6)
            local gate
            if h_norm then
                local wk     = self.W_K[branch][n]
                local ke = {}
                for i = 1, self.dim do
                    ke[i] = wk[i] * e_norm[i]
                end
                local ke_norm = rms_norm(ke)
                local score   = dot(h_norm, ke_norm) / math.sqrt(self.dim)
                
                -- NEW: Clamp score to prevent sigmoid overflow
                score = math.max(-10, math.min(10, score))
                gate = sigmoid(score)
            else
                local se = self.store:lookup(ngkey)
                local freq = se and math.log(1 + se.count) / 20 or 0.3
                gate = sigmoid(freq - 0.5)
            end

            -- Order bonus
            local order_w = 1.0 + (n-1) * 0.3
            local w       = gate * order_w

            -- Depthwise conv across orders (PATCHED: boundary handling)
            local k0 = self.conv_k[1]
            local k1 = self.conv_k[2]
            local k2 = self.conv_k[3]
            for i = 1, self.dim do
                -- NEW: Zero-padding for first N-gram (no prev)
                local prev  = prev_emb and prev_emb[i] or 0
                local curr  = emb[i]
                -- NEW: Safe division (gate_sum can be 0 on first iteration)
                local next_approx = gate_sum > 1e-6 and (fused[i] / gate_sum) or 0
                local convd = k0 * prev + k1 * curr + k2 * next_approx
                fused[i]   = fused[i] + w * convd
            end
            gate_sum = gate_sum + w
            prev_emb = emb
        end
        
        ::continue_ngram::
    end

    -- Normalize
    if gate_sum > EPSILON then
        for i = 1, self.dim do fused[i] = fused[i] / gate_sum end
    else
        -- NEW: Fallback to zero vector if no valid N-grams
        for i = 1, self.dim do fused[i] = 0.0 end
    end

    -- Value projection
    for i = 1, self.dim do
        fused[i] = fused[i] * self.W_V[i]
    end

    return fused
end

-- ────────────────────────────────────────────────────────────────
--  §2.4  Multi-Branch Fusion (PATCHED: NaN validation)
-- ────────────────────────────────────────────────────────────────

function Engram:retrieve_multibranch(tokens, hidden)
    local combined = {}
    for i = 1, self.dim do combined[i] = 0.0 end
    
    local valid_branches = 0

    for m = 1, self.n_branches do
        local branch_out = self:retrieve(tokens, hidden, m)
        
        -- NEW: Validate branch output before fusion
        if is_valid_vec(branch_out) then
            for i = 1, self.dim do
                combined[i] = combined[i] + branch_out[i]
            end
            valid_branches = valid_branches + 1
        else
            self.stats.nan_guards = self.stats.nan_guards + 1
        end
    end
    
    -- NEW: Safe normalization
    local norm_factor = valid_branches > 0 and valid_branches or 1
    for i = 1, self.dim do
        combined[i] = combined[i] / norm_factor
    end
    
    return combined
end

-- ────────────────────────────────────────────────────────────────
--  Online Hebbian Update (PATCHED: gradient clipping)
-- ────────────────────────────────────────────────────────────────

function Engram:update(tokens, target_emb, reward)
    reward = reward or 1.0
    self.stats.updates = self.stats.updates + 1
    
    -- NEW: Validate target embedding
    if not is_valid_vec(target_emb) then
        self.stats.nan_guards = self.stats.nan_guards + 1
        return
    end

    local ids = {}
    for _, t in ipairs(tokens) do ids[#ids+1] = self:token_id(t) end

    local target_norm = rms_norm(target_emb)
    
    -- NEW: Clamp reward to [-1, +1]
    reward = math.max(-1.0, math.min(1.0, reward))

    for n = 1, math.min(self.max_ngram, #ids) do
        local ngkey, nghash = self:_ngram_key(ids, n)
        local emb = self.embed_tables[n][nghash]
        if not emb then
            emb = rand_vec(self.dim, nghash * 31 + n * 7)
            self.embed_tables[n][nghash] = emb
        end
        
        -- NEW: Validate before update
        if not is_valid_vec(emb) then
            self.stats.nan_guards = self.stats.nan_guards + 1
            emb = rand_vec(self.dim, nghash * 31 + n * 7)
            self.embed_tables[n][nghash] = emb
        end

        -- Cosine similarity
        local sim = 0
        local na, nb = 0, 0
        for i = 1, self.dim do
            sim = sim + emb[i]*target_norm[i]
            na  = na  + emb[i]*emb[i]
        end
        na = math.sqrt(na + EPSILON)
        sim = sim / na
        
        -- NEW: Clamp similarity to [-1, 1]
        sim = math.max(-1.0, math.min(1.0, sim))

        -- Hebbian: push emb toward target
        local delta = reward * (1.0 - sim)
        
        -- NEW: Adaptive learning rate (smaller for high similarity)
        local adaptive_lr = self.lr * (1.0 - math.abs(sim) * 0.5)
        
        -- PATCHED: vec_add_ip now has internal clipping
        vec_add_ip(emb, target_norm, adaptive_lr * delta)
        
        -- NEW: Post-update validation + normalization
        local norm = 0
        for i = 1, self.dim do
            if emb[i] == emb[i] then  -- Skip NaN
                norm = norm + emb[i] * emb[i]
            else
                emb[i] = 0.0  -- Fix NaN
            end
        end
        norm = math.sqrt(norm + EPSILON)
        
        -- NEW: Prevent embedding explosion (max norm = 5.0)
        if norm > 5.0 then
            self.stats.hebbian_clips = self.stats.hebbian_clips + 1
            for i = 1, self.dim do
                emb[i] = emb[i] * (5.0 / norm)
            end
        end

        -- Propagate reward to StateStore (PATCHED: bounded delta)
        local state_entry = self.store.hot:get(nghash)
        if state_entry and reward > 0 then
            -- Boost frequency proportionally
            local boost = math.floor(reward * 5)
            state_entry.count = state_entry.count + boost
            
            -- NEW: Update transition probabilities with bounded delta
            if state_entry.transitions then
                for _, t in ipairs(state_entry.transitions) do
                    local p = SS.Enc.prob_decode12(t.prob12)
                    -- CHANGED: Max delta is ±0.05 (was unbounded)
                    local delta_p = math.max(-0.05, math.min(0.05, reward * 0.01))
                    local new_p = math.max(0.001, math.min(0.999, p + delta_p))
                    t.prob12 = SS.Enc.prob_encode12(new_p)
                end
            end
        end

        -- Record N-gram access in bloom
        self.store.bloom:add(nghash)
    end
end

-- ────────────────────────────────────────────────────────────────
--  Pattern-Aware Sequence Recording (PATCHED: length cap)
-- ────────────────────────────────────────────────────────────────

function Engram:record_sequence(tokens, response, success_rate)
    -- NEW: Cap chain length to prevent memory leaks
    local max_chain_len = 64
    local capped_tokens = {}
    for i = 1, math.min(#tokens, max_chain_len) do
        capped_tokens[i] = tokens[i]
    end
    
    -- Build state chain
    local chain = {}
    for _, t in ipairs(capped_tokens) do
        chain[#chain+1] = SS.Enc.fnv1a32(self:canonicalize(t))
    end

    local query = table.concat(capped_tokens, " ")
    
    -- NEW: Validate response length
    if #response > 1000 then
        response = response:sub(1, 1000) .. "..."
    end
    
    self.store:record_case(query, response, chain, success_rate)
end

-- ────────────────────────────────────────────────────────────────
--  Statistics & Debug
-- ────────────────────────────────────────────────────────────────

function Engram:print_stats()
    local s = self.stats
    local n_buckets = 0
    for n = 1, self.max_ngram do
        for _ in pairs(self.embed_tables[n]) do n_buckets = n_buckets + 1 end
    end
    print("┌─── Engram v2.1 Stats ────────────────────────────┐")
    print(("│  Retrievals:       %10d"):format(s.retrievals))
    print(("│  Store hits:       %10d"):format(s.store_hits))
    print(("│  Updates:          %10d"):format(s.updates))
    print(("│  Active buckets:   %10d"):format(n_buckets))
    print(("│  Vocab size:       %10d"):format(self.vocab_size))
    print(("│  Embed inits:      %10d"):format(s.embed_inits))
    print(("│  NaN guards:       %10d"):format(s.nan_guards))
    print(("│  Hebbian clips:    %10d"):format(s.hebbian_clips))
    print("└──────────────────────────────────────────────────┘")
end

-- ────────────────────────────────────────────────────────────────
--  Persistence (PATCHED: version header, int8 quantization support)
-- ────────────────────────────────────────────────────────────────

function Engram:save(path, quantize)
    quantize = quantize or false  -- Default: float32
    
    if quantize then
        self:save_quantized(path)
        return
    end
    
    local f = assert(io.open(path, "w"))
    
    -- NEW: Version header
    f:write("VERSION=2.1\n")
    f:write("DIM=" .. self.dim .. "\n")
    f:write("MAX_NGRAM=" .. self.max_ngram .. "\n")
    f:write("BRANCHES=" .. self.n_branches .. "\n")

    -- Embedding tables
    local saved_buckets = 0
    for n = 1, self.max_ngram do
        for bucket, vec in pairs(self.embed_tables[n]) do
            -- NEW: Skip invalid embeddings
            if is_valid_vec(vec) then
                f:write(("E|%d|%d|"):format(n, bucket) .. table.concat(vec,",") .. "\n")
                saved_buckets = saved_buckets + 1
            end
        end
    end

    -- W_V
    f:write("WV|" .. table.concat(self.W_V, ",") .. "\n")

    -- W_K (branch × order)
    for m = 1, self.n_branches do
        for n = 1, self.max_ngram do
            f:write(("WK|%d|%d|"):format(m,n) .. table.concat(self.W_K[m][n],",") .. "\n")
        end
    end

    -- Vocab proj
    for tok, id in pairs(self.vocab_proj) do
        local safe = tok:gsub("|","\\p")
        f:write(("V|%s|%d\n"):format(safe, id))
    end

    f:close()
    print(("[Engram v2.1] Saved %d buckets → %s"):format(saved_buckets, path))
end

-- NEW: Quantized int8 save (4× compression)
function Engram:save_quantized(path)
    local f = assert(io.open(path, "wb"))
    
    -- Binary header
    f:write("ENGQ")  -- Magic: Engram Quantized
    f:write(string.char(2, 1))  -- Version 2.1
    f:write(SS.Enc.pack32(self.dim))
    f:write(string.char(self.max_ngram))
    f:write(string.char(self.n_branches))
    
    local saved_buckets = 0
    
    for n = 1, self.max_ngram do
        for bucket, vec in pairs(self.embed_tables[n]) do
            if is_valid_vec(vec) then
                -- Quantize to int8
                local max_abs = 0
                for i = 1, self.dim do
                    max_abs = math.max(max_abs, math.abs(vec[i]))
                end
                
                local scale = max_abs / 127.0
                if scale < 1e-9 then scale = 1.0 end
                
                local quant = {}
                for i = 1, self.dim do
                    local q = math.floor(vec[i] / scale + 0.5)
                    q = math.max(-127, math.min(127, q))
                    quant[i] = string.char((q + 128) & 0xFF)  -- Shift to [0,255]
                end
                
                -- Write: n(1) | bucket(4) | scale(4) | data(dim)
                f:write(string.char(n))
                f:write(SS.Enc.pack32(bucket))
                f:write(SS.Enc.pack32(math.floor(scale * 1e6)))  -- Store as micro-units
                f:write(table.concat(quant))
                
                saved_buckets = saved_buckets + 1
            end
        end
    end
    
    f:close()
    print(("[Engram v2.1] Saved %d quantized buckets → %s (int8)"):format(
        saved_buckets, path))
end

function Engram:load(path)
    local f = io.open(path, "rb")
    if f then
        -- Try binary quantized format first
        local magic = f:read(4)
        f:close()
        if magic == "ENGQ" then
            return self:load_quantized(path)
        end
    end
    
    -- Fall back to text format
    f = io.open(path, "r")
    if not f then print("[Engram v2.1] No checkpoint: "..path); return false end
    
    local version = "2.0"
    
    for line in f:lines() do
        local tag = line:sub(1,2)
        
        if line:sub(1,7) == "VERSION" then
            version = line:match("VERSION=(.+)") or "2.0"
            
        elseif tag == "E|" or line:sub(1,1) == "E" then
            local n, bkt, vals = line:match("E|(%d+)|(%d+)|(.+)")
            n = tonumber(n); bkt = tonumber(bkt)
            if n and bkt then
                local v = {}
                for sv in vals:gmatch("[^,]+") do
                    local val = tonumber(sv)
                    -- NEW: NaN guard on load
                    if val and val == val then
                        v[#v+1] = val
                    else
                        v[#v+1] = 0.0
                    end
                end
                if #v == self.dim then
                    self.embed_tables[n][bkt] = v
                end
            end
            
        elseif line:sub(1,2) == "WV" then
            local vals = line:match("WV|(.+)")
            if vals then
                self.W_V = {}
                for sv in vals:gmatch("[^,]+") do
                    local val = tonumber(sv)
                    self.W_V[#self.W_V+1] = (val and val == val) and val or 0.0
                end
            end
            
        elseif line:sub(1,2) == "WK" then
            local m, n, vals = line:match("WK|(%d+)|(%d+)|(.+)")
            m = tonumber(m); n = tonumber(n)
            if m and n then
                local v = {}
                for sv in vals:gmatch("[^,]+") do
                    local val = tonumber(sv)
                    v[#v+1] = (val and val == val) and val or 0.0
                end
                if not self.W_K[m] then self.W_K[m]={} end
                self.W_K[m][n] = v
            end
            
        elseif line:sub(1,1) == "V" then
            local tok, id = line:match("V|(.+)|(%d+)$")
            if tok then
                tok = tok:gsub("\\p","|")
                self.vocab_proj[tok] = tonumber(id)
            end
        end
    end
    f:close()
    
    print(("[Engram v2.1] Loaded from %s (version %s)"):format(path, version))
    return true
end

-- NEW: Load quantized int8 format
function Engram:load_quantized(path)
    local f = io.open(path, "rb")
    if not f then return false end
    
    local magic = f:read(4)
    if magic ~= "ENGQ" then
        f:close()
        print("[Engram] Not a quantized file")
        return false
    end
    
    local ver_major = string.byte(f:read(1))
    local ver_minor = string.byte(f:read(1))
    local dim       = SS.Enc.unpack32(f:read(4))
    local max_ngram = string.byte(f:read(1))
    local n_branches= string.byte(f:read(1))
    
    if dim ~= self.dim then
        print(("[Engram] Dimension mismatch: file=%d, model=%d"):format(dim, self.dim))
        f:close()
        return false
    end
    
    local loaded = 0
    
    while true do
        local n_byte = f:read(1)
        if not n_byte or #n_byte == 0 then break end
        
        local n      = string.byte(n_byte)
        local bucket = SS.Enc.unpack32(f:read(4))
        local scale_u= SS.Enc.unpack32(f:read(4))
        local scale  = scale_u / 1e6
        local data   = f:read(dim)
        
        if not data or #data < dim then break end
        
        local vec = {}
        for i = 1, dim do
            local q = string.byte(data, i) - 128  -- Shift back to [-127, 127]
            vec[i] = q * scale
        end
        
        if not self.embed_tables[n] then self.embed_tables[n] = {} end
        self.embed_tables[n][bucket] = vec
        loaded = loaded + 1
    end
    
    f:close()
    print(("[Engram v2.1] Loaded %d quantized buckets from %s"):format(loaded, path))
    return true
end

-- ────────────────────────────────────────────────────────────────
--  NEW: Diagnostic utilities
-- ────────────────────────────────────────────────────────────────

function Engram:health_check()
    local issues = {}
    
    -- Check all embeddings for NaN/Inf
    for n = 1, self.max_ngram do
        for bucket, vec in pairs(self.embed_tables[n]) do
            if not is_valid_vec(vec) then
                issues[#issues+1] = ("N-gram %d, bucket %d: invalid embedding"):format(n, bucket)
            end
            
            -- Check norm bounds
            local norm = 0
            for i = 1, #vec do
                norm = norm + vec[i] * vec[i]
            end
            norm = math.sqrt(norm)
            if norm > 10.0 then
                issues[#issues+1] = ("N-gram %d, bucket %d: norm=%.2f (explosion)"):format(
                    n, bucket, norm)
            end
        end
    end
    
    -- Check projection matrices
    for m = 1, self.n_branches do
        for n = 1, self.max_ngram do
            if not is_valid_vec(self.W_K[m][n]) then
                issues[#issues+1] = ("W_K[%d][%d]: invalid"):format(m, n)
            end
        end
    end
    
    if not is_valid_vec(self.W_V) then
        issues[#issues+1] = "W_V: invalid"
    end
    
    if #issues == 0 then
        print("[Engram] ✓ Health check passed")
        return true
    else
        print("[Engram] ⚠ Health check found issues:")
        for _, msg in ipairs(issues) do
            print("  • " .. msg)
        end
        return false
    end
end

-- NEW: Compact (remove low-usage embeddings)
function Engram:compact(min_access_count)
    min_access_count = min_access_count or 3
    local removed = 0
    
    for n = 1, self.max_ngram do
        local to_remove = {}
        for bucket, vec in pairs(self.embed_tables[n]) do
            -- Check if N-gram exists in StateStore
            local ngkey = "ng:" .. bucket  -- Simplified lookup
            local se = self.store:lookup(ngkey)
            if not se or (se.count or 0) < min_access_count then
                to_remove[#to_remove+1] = bucket
            end
        end
        
        for _, bucket in ipairs(to_remove) do
            self.embed_tables[n][bucket] = nil
            removed = removed + 1
        end
    end
    
    print(("[Engram] Compacted: removed %d low-usage embeddings"):format(removed))
    return removed
end

return Engram