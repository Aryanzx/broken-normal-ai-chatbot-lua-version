# Bug Fix: Repetitive Token Generation

## Problem
The chatbot was generating repetitive `<unk>` tokens, causing validation failures and fallback to generic error messages:

```
You: hi
[SSM] Gen: <unk> <unk> godfather <unk> <unk> <unk> <unk> <unk> <unk> <u
[Validation] Failed: repetitive
Bot: I'm still learning. Can you explain differently?
```

## Root Causes

### 1. Invalid Token ID Generation
The SSM decoder was generating token IDs that didn't exist in the tokenizer's vocabulary. When decoded, these became `<unk>` tokens.

**Location**: [`ssm_v2.lua:528-555`](ssm_v2.lua:528)

**Issue**: The sampling loop iterated over all logits without checking if the corresponding token ID existed in the vocabulary.

### 2. No Repetition Detection During Generation
The decoder had no mechanism to detect when it was stuck in a repetition loop.

### 3. Overly Strict Validation
The validation counted special tokens like `<unk>` in the repetition check, which would trigger even with a few unknown tokens.

**Location**: [`chatbot_v2.lua:269-276`](chatbot_v2.lua:269)

## Fixes Applied

### Fix 1: Vocabulary Bounds Checking (ssm_v2.lua)
**Lines 528-560**

Added validation to ensure only valid token IDs are considered during sampling:

```lua
-- Before: Included all logits
for i=1,#logits do
    if val == val then
        scaled[i]={i=i,v=val/beam_temp}
    end
end

-- After: Only include tokens that exist in vocabulary
for i=1,#logits do
    if val == val and tokenizer.id2word and tokenizer.id2word[i] then
        scaled[#scaled+1]={i=i,v=val/beam_temp}
    end
end
```

Added safety check for empty candidate list:
```lua
if #scaled == 0 then
    print("[SSM] Warning: No valid tokens in vocabulary, using fallback")
    break
end
```

### Fix 2: Repetition Penalty (ssm_v2.lua)
**Lines 530-544**

Added a repetition penalty that reduces logits for recently generated tokens:

```lua
-- Repetition penalty - reduce logits for recently used tokens
if #gen_ids > 0 then
    local recent_window = math.min(10, #gen_ids)
    for i = #gen_ids - recent_window + 1, #gen_ids do
        local recent_id = gen_ids[i]
        if recent_id and logits[recent_id] then
            local recency_factor = (i - (#gen_ids - recent_window)) / recent_window
            logits[recent_id] = logits[recent_id] - (1.5 * recency_factor)
        end
    end
end
```

### Fix 3: Early Repetition Detection (ssm_v2.lua)
**Lines 497-502, 562-577**

Added tracking of consecutive identical tokens with early stopping:

```lua
local repeat_count = 0
local last_token = nil

-- In generation loop:
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
```

### Fix 4: Improved Validation (chatbot_v2.lua)
**Lines 269-289**

Enhanced validation to:
1. Exclude special tokens from repetition count
2. Detect excessive `<unk>` tokens specifically
3. Provide better diagnostic messages

```lua
-- Skip special tokens in repetition count
local special_tokens = {["<unk>"]=true, ["<pad>"]=true, ["<bos>"]=true, ["<eos>"]=true}

for word in response_text:lower():gmatch("%S+") do
    if not special_tokens[word] then
        word_count[word] = (word_count[word] or 0) + 1
        if word_count[word] > 5 then
            return false, "repetitive: '" .. word .. "' repeated " .. word_count[word] .. " times"
        end
    end
end

-- Check for excessive special tokens
local unk_count = 0
for word in response_text:gmatch("%S+") do
    if word == "<unk>" then unk_count = unk_count + 1 end
end
if unk_count > 3 then
    return false, "too_many_unknowns (" .. unk_count .. " <unk> tokens)"
end
```

## Expected Behavior After Fix

1. **No Invalid Token IDs**: Only tokens that exist in the vocabulary will be sampled
2. **Diverse Generation**: Repetition penalty encourages the model to use different tokens
3. **Early Stopping**: Generation stops if stuck in a 3+ token repetition loop
4. **Better Diagnostics**: Validation messages now indicate the specific issue
5. **Graceful Fallback**: If generation fails, the system falls back to template responses

## Testing Recommendations

1. Test with simple inputs like "hi", "hello", "how are you"
2. Monitor console output for warning messages
3. Check that responses are coherent and non-repetitive
4. Verify that `<unk>` tokens are rare or absent
5. Test edge cases with very short or very long inputs

## Related Files Modified

- [`ssm_v2.lua`](ssm_v2.lua) - SSM decoder improvements
- [`chatbot_v2.lua`](chatbot_v2.lua) - Validation improvements
