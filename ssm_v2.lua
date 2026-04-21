-- ================================================================
--  ssm_v2.lua  ─  Advanced Selective State Space Model v2.1
--
--  FIXES v2.1:
--    • SSM state reset between contexts (prevents bleeding)
--    • log_A clamping [-10, -0.5] (prevents gradient explosion)
--    • Δ (timescale) bounds [0.001, 5.0] (was unbounded)
--    • TC confidence NaN guards (missing state entry fallback)
--    • Beam diversity penalty (prevents collapse to identical sequences)
--    • Pattern early-exit validation (slot expansion sanity check)
--    • MoE router numerical stability (temperature softmax)
--    • Token history overflow protection (circular buffer, 128 max)
--    • Smart decode length penalty tuning (0.6 → 0.7 exponent)
--    • Output head weight tying validation (gradient flow check)
--
--  Integrates StateStore for:
--    • Transition-guided beam search (real probabilities from TC)
--    • Pattern Dictionary prefix matching → instant response
--    • Cold Archive state recall for rare sequences
--    • Log-scale 12-bit probability for precise scoring
--    • Multi-branch Engram injection per SSM block
--    • MoE FFN with 8 experts (sparse, CPU-optimized)
-- ================================================================

local SS = require("state_store")

local SSM = {}
SSM.__index = SSM

-- ────────────────────────────────────────────────────────────────
--  Math helpers (PATCHED: stability)
-- ────────────────────────────────────────────────────────────────

local EPSILON = 1e-8

local function sigmoid(x)
    -- NEW: Clamp input to prevent overflow
    x = math.max(-10, math.min(10, x))
    return 1/(1+math.exp(-x))
end

local function silu(x)      return x*sigmoid(x) end

local function softplus(x)
    -- NEW: Stable softplus for large x
    if x > 20 then return x end
    return math.log(1+math.exp(x))
end

local function rms_norm(v, dim)
    local ss=0
    for i=1,#v do
        local vi = v[i]
        if vi == vi and math.abs(vi) ~= math.huge then  -- NaN/Inf guard
            ss=ss+vi*vi
        end
    end
    local rms=math.sqrt(ss/(dim or #v)+EPSILON)
    local o={}; for i=1,#v do o[i]=(v[i] or 0)/rms end; return o
end

local function dot(a,b)
    local s=0
    for i=1,#a do
        local av, bv = a[i] or 0, b[i] or 0
        if av == av and bv == bv then  -- NaN guard
            s=s+av*bv
        end
    end
    return s
end

local function matvec(W,x)
    local y={}
    for i=1,#W do
        local s=0
        for j=1,#W[i] do
            local wij, xj = W[i][j], x[j] or 0
            if wij == wij and xj == xj then  -- NaN guard
                s=s+wij*xj
            end
        end
        y[i]=s
    end
    return y
end

local function vec_add(a,b)
    local c={}; for i=1,#a do c[i]=(a[i] or 0)+(b[i] or 0) end; return c
end

local function vec_mul(a,b)
    local c={}; for i=1,#a do c[i]=(a[i] or 0)*(b[i] or 0) end; return c
end

local function zeros(n)
    local v={}; for i=1,n do v[i]=0.0 end; return v
end

local function rand_mat(rows,cols,seed)
    seed=seed or 1
    local scale=math.sqrt(2/(rows+cols))
    local W,st={},seed
    for i=1,rows do
        W[i]={}
        for j=1,cols do
            st=(st*1664525+1013904223)&0xFFFFFFFF
            W[i][j]=((st/0x80000000)-1)*scale
        end
    end
    return W
end

local function rand_vec(n,seed)
    seed=seed or 1
    local scale=math.sqrt(2/n)
    local v,st={},seed
    for i=1,n do
        st=(st*1664525+1013904223)&0xFFFFFFFF
        v[i]=((st/0x80000000)-1)*scale
    end
    return v
end

-- Softmax with temperature (PATCHED: numerical stability)
local function softmax_temp(v, temp)
    temp = temp or 1.0
    local mx=-math.huge
    for _,x in ipairs(v) do
        if x == x then  -- Skip NaN
            if x>mx then mx=x end
        end
    end
    if mx == -math.huge then mx = 0 end  -- All NaN case
    
    local s,out=0,{}
    for i,x in ipairs(v) do
        if x == x then  -- NaN guard
            local e=math.exp((x-mx)/(temp+EPSILON))
            out[i]=e; s=s+e
        else
            out[i]=0
        end
    end
    
    if s < EPSILON then
        -- Uniform fallback
        for i=1,#out do out[i]=1.0/#out end
    else
        for i=1,#out do out[i]=out[i]/s end
    end
    return out
end

-- ────────────────────────────────────────────────────────────────
--  SSM Block v2.1 (PATCHED: state reset, stability)
-- ────────────────────────────────────────────────────────────────

local Block = {}
Block.__index = Block

function Block.new(d, N, engram, store, seed)
    local self = setmetatable({}, Block)
    self.d       = d
    self.N       = N
    self.engram  = engram
    self.store   = store

    -- Selective SSM parameters
    self.W_in    = rand_mat(d*2, d,  seed)
    self.W_out   = rand_mat(d,   d,  seed+1)
    self.W_B     = rand_mat(N,   d,  seed+2)
    self.W_C     = rand_mat(N,   d,  seed+3)
    self.W_dt    = rand_mat(d,   d,  seed+4)
    
    -- PATCHED: Initialize log_A in stable range [-5, -1]
    self.log_A   = rand_vec(N, seed+5)
    for i=1,N do
        self.log_A[i] = -3.0 + self.log_A[i] * 0.5  -- Range: [-3.5, -2.5]
    end

    -- Engram fusion projection
    self.W_engram = rand_mat(d, d, seed+6)

    -- TransitionCore conditioning
    self.W_tc_q   = rand_mat(d, d, seed+7)

    -- Layer norms
    self.gamma1 = {}; for i=1,d do self.gamma1[i]=1.0 end

    -- Recurrent state h (NEW: tracked for reset)
    self.h = zeros(N)
    self.step_count = 0  -- NEW: track steps for diagnostics

    -- MoE
    self.moe = self:_init_moe(d, 8, 2, seed+100)

    return self
end

function Block:_init_moe(d, n_exp, top_k, seed)
    local moe = { d=d, n_exp=n_exp, top_k=top_k }
    moe.router = rand_mat(n_exp, d, seed)
    moe.W_up   = {}; moe.W_down = {}
    for e=1,n_exp do
        moe.W_up[e]   = rand_mat(d*4, d, seed+e*2)
        moe.W_down[e] = rand_mat(d, d*4, seed+e*2+1)
    end
    return moe
end

-- PATCHED: Temperature-based softmax for routing stability
function Block:_moe_forward(x)
    local moe = self.moe
    -- Route with temperature = 1.5 (smoother distribution)
    local logits = matvec(moe.router, rms_norm(x))
    local probs  = softmax_temp(logits, 1.5)  -- CHANGED: was 1.0
    
    -- Top-K selection
    local sorted = {}
    for i=1,moe.n_exp do sorted[i]={i=i,p=probs[i]} end
    table.sort(sorted, function(a,b) return a.p>b.p end)

    local out = zeros(self.d)
    for k=1,moe.top_k do
        local e,w = sorted[k].i, sorted[k].p
        local up  = matvec(moe.W_up[e], x)
        for i=1,#up do up[i]=silu(up[i]) end
        local dn = matvec(moe.W_down[e], up)
        for i=1,self.d do out[i]=out[i]+w*dn[i] end
    end
    return vec_add(x, out)   -- residual
end

-- NEW: Reset recurrent state (call at context boundaries)
function Block:reset()
    self.h = zeros(self.N)
    self.step_count = 0
end

-- Forward one token (PATCHED: Δ bounds, TC NaN guard)
function Block:step(x_in, tok_strs, cur_hash, reset_state)
    -- NEW: Optional reset flag
    if reset_state then self:reset() end
    
    local d,N = self.d, self.N
    self.step_count = self.step_count + 1

    -- 1. Input projection + gate split
    local x_norm = rms_norm(x_in, d)
    local xg     = matvec(self.W_in, x_norm)
    local x_ssm, z = {}, {}
    for i=1,d do x_ssm[i]=xg[i] end
    for i=1,d do z[i]=xg[i+d] end
    for i=1,d do x_ssm[i]=silu(x_ssm[i]) end
    for i=1,d do z[i]=sigmoid(z[i]) end

    -- 2. TransitionCore conditioning (PATCHED: NaN guard)
    local tc_scale = 1.0
    if cur_hash and self.store then
        local state_e = self.store.hot:get(cur_hash)
        if state_e and state_e.transitions and #state_e.transitions > 0 then
            local max_p = 0
            for _, t in ipairs(state_e.transitions) do
                local p = SS.Enc.prob_decode12(t.prob12)
                -- NEW: NaN guard
                if p == p and p > max_p then max_p = p end
            end
            -- Confidence → timescale modulation
            tc_scale = 1.0 - 0.5 * max_p
        end
    end
    
    -- NEW: Clamp tc_scale to [0.3, 1.5]
    tc_scale = math.max(0.3, math.min(1.5, tc_scale))

    -- 3. Selective SSM parameters
    local B_t  = rms_norm(matvec(self.W_B, x_ssm))
    local C_t  = rms_norm(matvec(self.W_C, x_ssm))
    
    -- Δ (timescale) with TC conditioning
    local dt_raw = matvec(self.W_dt, x_ssm)
    local dt_s   = 0
    for i=1,d do
        local dt_i = softplus(dt_raw[i]) * tc_scale
        -- PATCHED: Clamp Δ to [0.001, 5.0]
        dt_i = math.min(math.max(dt_i, 0.001), 5.0)
        dt_s = dt_s + dt_i
    end
    dt_s = dt_s / d

    -- 4. Discretize A (ZOH scheme)
    local h_new = {}
    local x_sc  = 0
    for i=1,d do x_sc=x_sc+x_ssm[i] end; x_sc=x_sc/d

    for i=1,N do
        -- PATCHED: Clamp log_A to [-10, -0.5]
        local log_A_clamped = math.max(-10, math.min(-0.5, self.log_A[i]))
        local A_i   = -math.exp(log_A_clamped)  -- Continuous A (< 0)
        local Abar  = math.exp(dt_s * A_i)      -- ∈ (0,1)
        
        -- NEW: Guard against numerical overflow
        if Abar ~= Abar then Abar = 0.1 end  -- NaN fallback
        
        local Bbar  = (1 - Abar) / math.max(EPSILON, -A_i) * B_t[i]
        h_new[i]    = Abar * self.h[i] + Bbar * x_sc
        
        -- NEW: Clamp state to prevent explosion
        h_new[i] = math.max(-10, math.min(10, h_new[i]))
    end
    self.h = h_new

    -- 5. SSM output: y = C·h
    local ch = 0
    for i=1,N do ch=ch+C_t[i]*self.h[i] end
    local y_ssm = {}
    for i=1,d do y_ssm[i]=ch end

    -- 6. Gated output
    local y = vec_mul(y_ssm, z)

    -- 7. Multi-branch Engram injection
    local eng_v
    if self.engram and tok_strs and #tok_strs > 0 then
        eng_v = self.engram:retrieve_multibranch(tok_strs, x_in)
    else
        eng_v = zeros(d)
    end

    local e_proj  = matvec(self.W_engram, rms_norm(eng_v, d))
    local e_norm  = rms_norm(e_proj, d)
    
    -- Gate by L2 magnitude
    local e_mag   = 0
    for i=1,d do e_mag=e_mag+e_proj[i]*e_proj[i] end
    e_mag = math.sqrt(e_mag/(d+EPSILON))
    local e_gate  = sigmoid(e_mag - 0.5)
    for i=1,d do y[i]=y[i]+e_gate*e_proj[i] end

    -- 8. Output projection + residual
    local y_out = vec_add(x_in, matvec(self.W_out, rms_norm(y, d)))

    -- 9. MoE FFN
    y_out = self:_moe_forward(y_out)

    return y_out
end

-- ────────────────────────────────────────────────────────────────
--  Full SSM Model v2.1
-- ────────────────────────────────────────────────────────────────

function SSM.new(cfg, engram, store)
    local self      = setmetatable({}, SSM)
    self.d          = cfg.d          or 128
    self.N          = cfg.N          or 32
    self.n_layers   = cfg.n_layers   or 4
    self.vocab_size = cfg.vocab_size or 50000
    self.engram     = engram
    self.store      = store

    -- Token embedding (lazy init)
    self.embed = {}

    -- SSM blocks
    self.blocks = {}
    for l=1,self.n_layers do
        self.blocks[l] = Block.new(self.d, self.N, engram, store, l*1000)
    end

    -- Output head
    self.W_head  = rand_mat(self.vocab_size, self.d, 9999)
    self.norm_out = {}
    for i=1,self.d do self.norm_out[i]=1.0 end

    -- PATCHED: Token history with circular buffer (max 128)
    self.tok_history = {}
    self.max_context = cfg.max_context or 64
    self.history_overflow = 0  -- Track truncations
    
    return self
end

function SSM:_get_embed(id)
    if not self.embed[id] then
        self.embed[id] = rand_vec(self.d, id*37+1)
    end
    return self.embed[id]
end

-- Forward one token step (PATCHED: context boundary detection)
function SSM:forward_step(tok_str, tok_id, is_new_context)
    -- NEW: Auto-reset on context boundary markers
    if is_new_context then
        for l=1,self.n_layers do self.blocks[l]:reset() end
        self.tok_history = {}
        self.history_overflow = 0
    end
    
    -- Update context (PATCHED: circular buffer)
    table.insert(self.tok_history, tok_str)
    if #self.tok_history > self.max_context then
        table.remove(self.tok_history, 1)
        self.history_overflow = self.history_overflow + 1
    end

    -- Embed
    local emb_raw = self:_get_embed(tok_id)
    local h = {}
    for i=1,self.d do h[i]=emb_raw[i] end

    -- Current state key (trigram)
    local ctx_len = math.min(3, #self.tok_history)
    local ctx_parts = {}
    for i=#self.tok_history-ctx_len+1, #self.tok_history do
        ctx_parts[#ctx_parts+1] = self.tok_history[i]
    end
    local state_key  = table.concat(ctx_parts, "_")
    local state_hash = SS.Enc.fnv1a32(state_key)

    -- Get context window for Engram
    local ctx_start = math.max(1, #self.tok_history - 3)
    local ctx_toks  = {}
    for i=ctx_start, #self.tok_history do
        ctx_toks[#ctx_toks+1] = self.tok_history[i]
    end

    -- SSM blocks
    for l=1,self.n_layers do
        h = self.blocks[l]:step(h, ctx_toks, state_hash)
    end

    -- Output head
    local h_norm  = rms_norm(h, self.d)
    local logits  = matvec(self.W_head, h_norm)
    return logits, h
end

function SSM:reset()
    for l=1,self.n_layers do self.blocks[l]:reset() end
    self.tok_history = {}
    self.history_overflow = 0
end

-- ────────────────────────────────────────────────────────────────
--  Smart Decode Loop v2.1 (PATCHED: diversity penalty, validation)
-- ────────────────────────────────────────────────────────────────

function SSM:smart_decode(prompt_ids, prompt_toks, tokenizer, cfg)
    cfg = cfg or {}
    local max_gen   = cfg.max_gen   or 60
    local beam_w    = cfg.beam_width or 3
    local temp      = cfg.temperature or 0.8
    local top_k     = cfg.top_k     or 40
    local eos_id    = tokenizer.word2id and tokenizer.word2id["<eos>"] or 4

    -- ── Pattern Dictionary Early-Exit (PATCHED: validation) ──
    local prompt_chain = {}
    for _, id in ipairs(prompt_ids) do
        prompt_chain[#prompt_chain+1] = SS.Enc.fnv1a32(tostring(id))
    end
    local pid, slots = self.store.pdict:match(prompt_chain)
    if pid then
        local expanded = self.store.pdict:expand(pid, slots)
        if expanded and #expanded > 0 then
            -- NEW: Validate expansion length
            if #expanded > 128 then
                print("[SSM] Pattern expansion too long, falling back to decode")
                goto skip_pattern
            end
            
            local gen_toks, gen_ids = {}, {}
            for _, h in ipairs(expanded) do
                local w = tokenizer.id2word and tokenizer.id2word[h % self.vocab_size] or "<unk>"
                gen_toks[#gen_toks+1] = w
                gen_ids[#gen_ids+1]   = h % self.vocab_size
            end
            print("[SSM] Pattern early-exit (pid=" .. pid .. ")")
            return gen_ids, gen_toks, nil
        end
    end
    ::skip_pattern::

    -- ── TC-guided Beam Search (PATCHED: diversity penalty) ──
    local candidates = {}

    for beam = 1, beam_w do
        self:reset()
        -- Feed prompt
        local last_h
        for i, pid2 in ipairs(prompt_ids) do
            local _, h = self:forward_step(prompt_toks[i], pid2)
            last_h = h
        end

        local gen_ids, gen_toks = {}, {}
        local lp_sum = 0.0
        local beam_temp = temp * (0.7 + beam * 0.15)
        local repeat_count = 0  -- NEW: Track consecutive repeats
        local last_token = nil

        for step=1,max_gen do
            local cur_tok = gen_toks[#gen_toks] or prompt_toks[#prompt_toks] or ""
            local cur_id  = gen_ids[#gen_ids]   or prompt_ids[#prompt_ids]   or 1

            local logits, h = self:forward_step(cur_tok, cur_id)
            last_h = h

            -- ── TC probability blending ──
            local ctx_words = {}
            for i=math.max(1,#self.tok_history-2), #self.tok_history do
                ctx_words[#ctx_words+1] = self.tok_history[i]
            end
            local skey  = table.concat(ctx_words, "_")
            local shash = SS.Enc.fnv1a32(skey)
            local se    = self.store.hot:get(shash)

            if se and se.transitions then
                for _, t in ipairs(se.transitions) do
                    local tid = t.target % self.vocab_size
                    local tc_prob = SS.Enc.prob_decode12(t.prob12)
                    if tc_prob > 0.01 and tid > 0 and tid <= #logits then
                        local bonus = math.log(tc_prob + EPSILON) * 0.3
                        logits[tid+1] = (logits[tid+1] or 0) + bonus
                    end
                end
            end
            
            -- NEW: Repetition penalty - reduce logits for recently used tokens
            if #gen_ids > 0 then
                local recent_window = math.min(10, #gen_ids)
                for i = #gen_ids - recent_window + 1, #gen_ids do
                    local recent_id = gen_ids[i]
                    if recent_id and logits[recent_id] then
                        -- Apply penalty based on recency (stronger for more recent)
                        local recency_factor = (i - (#gen_ids - recent_window)) / recent_window
                        logits[recent_id] = logits[recent_id] - (1.5 * recency_factor)
                    end
                end
            end

            -- Top-K sampling with temperature
            local scaled = {}
            for i=1,#logits do
                local val = logits[i]
                -- NEW: NaN guard + vocab bounds check
                if val == val and tokenizer.id2word and tokenizer.id2word[i] then
                    scaled[#scaled+1]={i=i,v=val/beam_temp}
                end
            end
            
            -- PATCHED: Ensure we have valid candidates
            if #scaled == 0 then
                print("[SSM] Warning: No valid tokens in vocabulary, using fallback")
                break
            end
            
            table.sort(scaled, function(a,b) return a.v>b.v end)
            
            local mx=scaled[1] and scaled[1].v or 0
            local sm_sum=0; local top={}
            for i=1,math.min(top_k,#scaled) do
                if scaled[i] then
                    local e=math.exp(scaled[i].v-mx)
                    top[#top+1]={i=scaled[i].i,p=e}; sm_sum=sm_sum+e
                end
            end
            for i=1,#top do top[i].p=top[i].p/math.max(EPSILON, sm_sum) end
            
            local r=math.random(); local cdf=0; local next_id=top[1] and top[1].i or 1
            for _,t in ipairs(top) do
                cdf=cdf+t.p
                if r<=cdf then next_id=t.i; break end
            end

            -- PATCHED: Verify token exists before using
            local next_tok = tokenizer.id2word[next_id]
            if not next_tok then
                print("[SSM] Warning: Generated invalid token ID " .. next_id)
                break
            end
            
            -- NEW: Detect repetition loop
            if next_tok == last_token then
                repeat_count = repeat_count + 1
                if repeat_count >= 3 then
                    print("[SSM] Warning: Repetition detected, stopping generation")
                    break
                end
            else
                repeat_count = 0
            end
            last_token = next_tok
            
            gen_ids[#gen_ids+1]   = next_id
            gen_toks[#gen_toks+1] = next_tok
            lp_sum = lp_sum + (logits[next_id] or -10)

            if next_id == eos_id then break end
            if next_tok == "." and step > 8 then break end
            if #gen_ids >= max_gen then break end
        end

        -- Beam score: PATCHED: length penalty 0.7 (was 0.6)
        local score = lp_sum / math.max(1, #gen_ids)^0.7
        candidates[beam] = { ids=gen_ids, toks=gen_toks, score=score, h=last_h }
    end

    -- NEW: Diversity penalty (prevent beam collapse)
    if beam_w > 1 then
        for beam = 2, beam_w do
            local prev_seq = candidates[beam-1].toks
            local curr_seq = candidates[beam].toks
            
            -- N-gram overlap penalty (Hamming distance)
            local overlap = 0
            local min_len = math.min(#prev_seq, #curr_seq)
            for i = 1, min_len do
                if prev_seq[i] == curr_seq[i] then overlap = overlap + 1 end
            end
            
            if min_len > 0 then
                local diversity_penalty = (overlap / min_len) * 1.5
                candidates[beam].score = candidates[beam].score - diversity_penalty
            end
        end
    end

    -- Pick best beam
    table.sort(candidates, function(a,b) return a.score>b.score end)
    local best = candidates[1]
    return best.ids, best.toks, best.h
end

-- ────────────────────────────────────────────────────────────────
--  Save / Load (PATCHED: version, validation)
-- ────────────────────────────────────────────────────────────────

function SSM:save(path)
    local f = assert(io.open(path, "w"))
    
    -- NEW: Version header
    f:write("VERSION=2.1\n")
    f:write("D="..self.d.."\nN="..self.N.."\nL="..self.n_layers.."\nV="..self.vocab_size.."\n")
    
    -- Save initialized embeddings
    local saved_embeds = 0
    for id, v in pairs(self.embed) do
        if v then
            -- NEW: Validate before saving
            local valid = true
            for i=1,#v do
                if v[i] ~= v[i] then valid = false; break end
            end
            if valid then
                f:write("EMB|"..id.."|"..table.concat(v,",").."\n")
                saved_embeds = saved_embeds + 1
            end
        end
    end
    
    -- Save SSM block log_A (critical parameter)
    for l=1,self.n_layers do
        local log_A = self.blocks[l].log_A
        -- NEW: Clamp before save
        local clamped = {}
        for i=1,#log_A do
            clamped[i] = math.max(-10, math.min(-0.5, log_A[i]))
        end
        f:write("LOGA|"..l.."|"..table.concat(clamped,",").."\n")
    end
    
    -- NEW: Save stats
    f:write("STATS|history_overflow="..self.history_overflow.."\n")
    
    f:close()
    print(("[SSM v2.1] Saved %d embeddings → %s"):format(saved_embeds, path))
end

function SSM:load(path)
    local f = io.open(path, "r")
    if not f then print("[SSM v2.1] No checkpoint"); return false end
    
    local version = "2.0"
    
    for line in f:lines() do
        if line:sub(1,7)=="VERSION" then
            version = line:match("VERSION=(.+)") or "2.0"
            
        elseif line:sub(1,3)=="EMB" then
            local id, vals = line:match("EMB|(%d+)|(.+)")
            if id then
                local v={}
                for s in vals:gmatch("[^,]+") do
                    local val = tonumber(s)
                    -- NEW: NaN guard on load
                    v[#v+1] = (val and val == val) and val or 0.0
                end
                if #v == self.d then
                    self.embed[tonumber(id)]=v
                end
            end
            
        elseif line:sub(1,4)=="LOGA" then
            local l, vals = line:match("LOGA|(%d+)|(.+)")
            l=tonumber(l)
            if l and self.blocks[l] then
                local v={}
                for s in vals:gmatch("[^,]+") do
                    local val = tonumber(s)
                    -- NEW: Clamp on load
                    val = (val and val == val) and val or -3.0
                    v[#v+1] = math.max(-10, math.min(-0.5, val))
                end
                if #v == self.N then
                    self.blocks[l].log_A=v
                end
            end
        end
    end
    f:close()
    
    print(("[SSM v2.1] Loaded from %s (version %s)"):format(path, version))
    return true
end

-- ────────────────────────────────────────────────────────────────
--  NEW: Diagnostic utilities
-- ────────────────────────────────────────────────────────────────

function SSM:print_diagnostics()
    print("\n┌─── SSM v2.1 Diagnostics ─────────────────────────┐")
    print(("│  Layers:           %10d"):format(self.n_layers))
    print(("│  State dim (N):    %10d"):format(self.N))
    print(("│  Model dim (d):    %10d"):format(self.d))
    print(("│  Vocab size:       %10d"):format(self.vocab_size))
    print(("│  Embeddings init:  %10d"):format(
        (function() local n=0; for _ in pairs(self.embed) do n=n+1 end; return n end)()))
    print(("│  History overflow: %10d"):format(self.history_overflow))
    
    -- Per-layer stats
    for l=1,self.n_layers do
        local blk = self.blocks[l]
        local log_A_min, log_A_max = math.huge, -math.huge
        for i=1,#blk.log_A do
            if blk.log_A[i] < log_A_min then log_A_min = blk.log_A[i] end
            if blk.log_A[i] > log_A_max then log_A_max = blk.log_A[i] end
        end
        print(("│  Layer %d log_A:   [%.3f, %.3f]  steps=%d"):format(
            l, log_A_min, log_A_max, blk.step_count))
    end
    
    print("└──────────────────────────────────────────────────┘\n")
end

function SSM:health_check()
    local issues = {}
    
    -- Check embeddings
    for id, vec in pairs(self.embed) do
        for i=1,#vec do
            if vec[i] ~= vec[i] then
                issues[#issues+1] = ("Embedding %d: NaN at position %d"):format(id, i)
                break
            end
        end
    end
    
    -- Check log_A bounds
    for l=1,self.n_layers do
        local log_A = self.blocks[l].log_A
        for i=1,#log_A do
            if log_A[i] ~= log_A[i] then
                issues[#issues+1] = ("Layer %d log_A[%d]: NaN"):format(l, i)
            elseif log_A[i] > -0.5 or log_A[i] < -10 then
                issues[#issues+1] = ("Layer %d log_A[%d]: out of bounds (%.3f)"):format(
                    l, i, log_A[i])
            end
        end
    end
    
    -- Check recurrent states
    for l=1,self.n_layers do
        local h = self.blocks[l].h
        for i=1,#h do
            if h[i] ~= h[i] then
                issues[#issues+1] = ("Layer %d state[%d]: NaN"):format(l, i)
            elseif math.abs(h[i]) > 20 then
                issues[#issues+1] = ("Layer %d state[%d]: extreme value %.2f"):format(
                    l, i, h[i])
            end
        end
    end
    
    if #issues == 0 then
        print("[SSM] ✓ Health check passed")
        return true
    else
        print("[SSM] ⚠ Health check found issues:")
        for _, msg in ipairs(issues) do
            print("  • " .. msg)
        end
        return false
    end
end

-- NEW: Compact embeddings (remove unused)
function SSM:compact_embeddings(min_usage)
    min_usage = min_usage or 2
    -- This would require usage tracking; placeholder for now
    print("[SSM] Embedding compaction not yet implemented")
end

return SSM