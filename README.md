# SSM + Engram v2 + USTCC StateStore Chatbot
**Architecture v2.0  ·  ~2.5 GB Budget  ·  10M Hot States / 100M Cases**

```
USTCC Components (state_store.lua)
┌──────────────────────────────────────────────────────────────────┐
│  MEMORY  (541 MB)                    DISK  (1.99 GB)             │
│  ┌──────────────────┐  25 MB         ┌──────────────────┐        │
│  │  Bloom Filter    │  20M states    │  Case Library    │ 1.5 GB │
│  │  k=7, FPR<1%     │               │  100M cases      │ zstd   │
│  └──────────────────┘               │  block-compressed│        │
│  ┌──────────────────┐  505 MB        └──────────────────┘        │
│  │  TransitionCore  │  10M hot       ┌──────────────────┐        │
│  │  MPHF open-addr  │  12-bit prob   │  Cold Archive    │ 489 MB │
│  │  5-byte TransEnt │  log-scale     │  65K LZ4 buckets │        │
│  └──────────────────┘               └──────────────────┘        │
│  ┌──────────────────┐  5 MB                                      │
│  │  Pattern Dict    │  10K patterns                              │
│  │  slot templates  │                                            │
│  └──────────────────┘                                            │
└──────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|---|---|
| `state_store.lua` | USTCC engine: Bloom + TransitionCore + ColdArchive + CaseLibrary + PatternDict |
| `engram_v2.lua` | Engram memory: multi-branch gating (§2.4), depthwise conv (§2.3), TC-aware retrieval |
| `ssm_v2.lua` | SSM: TC-conditioned Δ scaling, pattern early-exit, TC-guided beam |
| `cbr_v2.lua` | CBR: CaseLibrary-backed, Pattern mining, Markov RL with Bellman updates |
| `chatbot_v2.lua` | Main orchestrator REPL |
| `train_colab_v2.py` | Python TPU trainer with full USTCC-aware model + Markov RL injection |

---

## USTCC Architecture Details

### Bloom Filter (25 MB, covers 20M states)
- **Blocked variant**: 512-bit blocks → all 7 probes in one cache line (~50ns)
- **Double hashing**: `h_i(x) = h1(x) + i·h2(x)` (Kirsch-Mitzenmacher)
- **10 bits/element** → FPR ≈ 0.82% at 20M elements
- Rebuilt every 24h to handle evictions (2 sec for 20M insertions)

### TransitionCore (505 MB memory, HOT)
- Open-addressing hash map approximating MPHF (no external library needed)
- **5-byte TransitionEntry**: `[24b target_idx][12b log-prob][4b type]`
- **12-bit log-scale probability**: `encode(p) = round(-log2(p)×256)` → 4096 levels, 0.014% error at 50%
- **6 transition types**: direct / backoff / interpolated / inferred / forced / decay
- Collision resolution via auxiliary 64-bit hash table (~186 KB for 11.6K collisions)
- Promotion/demotion thresholds: hot if count≥10, demote if count<5/90d (hysteresis=5)

### Cold Archive (489 MB disk)
- **65,536 buckets** by `hash32 >> 16`; each bucket LZ4-compressed (~153 states, ~4KB)
- **512 KB bucket index** loaded in memory → O(1) bucket lookup → disk read → LZ4 decompress (~2μs) → binary search
- Latency: 0.5–2ms vs hot at 100ns (5,000–20,000× slower → bloom filter saves 49.6% of cold reads)

### Case Library (1.5 GB disk, 100M cases)
- **Time-sortable case_id**: `timestamp_48 | sequence_16` (up to 65K cases/ms)
- **Block-compressed**: 64KB blocks, ~2,112 cases/block, ~47,348 blocks
- **Pattern variant** (90% of cases): stores `pattern_id + slot_values` (avg 27 bytes vs 60 bytes custom)
- **Custom variant** (10% of cases): full `state_hash_32[]` chain

### Pattern Dictionary (5 MB memory, 10K patterns)
- Slot sentinel: `0xFFFF0000 | slot_index` in state sequence
- Template hash for O(1) matching; LRU eviction by frequency × recency
- Background mining: scans recent cases for frequent subsequences → auto-register patterns
- Retroactively converts custom chains to pattern chains (50–80% space saving)

---

## Engram v2 Key Features

### Multi-Branch Architecture (§2.4)
```
α^(m) = σ( RMSNorm(h)ᵀ · RMSNorm(W_K^(m) · e) / √d )   [Eq.6]
u^(m) = α^(m) · W_V · e                                   [shared W_V]
u     = (1/M) Σ_m u^(m)                                   [M=4 branches]
```

### Depthwise Conv Refinement (§2.3)
3-tap causal conv across N-gram orders: `conv(prev_order, curr_order)`

### StateStore Integration
- Engram retrieval **bloom-guards** unknown N-grams (fast reject)
- TransitionCore **transition count** scales embedding magnitude
- RL reward signal propagates to TC transition probabilities via Hebbian update

---

## SSM v2 Smart Decode Loop
1. **Pattern early-exit**: if prompt matches PatternDict entry → instant O(1) expansion
2. **TC-conditioned Δ**: high-confidence TC transitions → shorter state horizon (faster scan)
3. **TC-guided beam**: logits boosted by `log(tc_prob) × 0.3` for likely next states
4. **3-beam diversity**: temperature varied per beam (0.7 + beam×0.15)
5. **Length penalty**: `score = Σlogp / length^0.6`

---

## Quick Start

### Lua CPU Chatbot
```bash
sudo apt install luajit
luajit chatbot_v2.lua                      # bare start
luajit chatbot_v2.lua my_knowledge.jsonl   # with knowledge file
```

REPL commands:
```
/good          → Markov RL +1 reward → TC prob boost + Engram Hebbian
/bad           → Markov RL -1 reward → TC decay transition
/save          → save all 6 components
/stats         → Bloom FPR, hot/cold counts, pattern count
/load path     → bulk load JSONL knowledge
/mine          → run pattern mining on current cases
quit           → save & exit
```

### Colab TPU Training
```python
!pip install datasets transformers torch_xla pandas torch -q

# HuggingFace dataset
!python train_colab_v2.py --dataset blended_skill_talk --epochs 3

# Local JSONL
!python train_colab_v2.py --dataset ./data.jsonl \
    --input_col "input" --output_col "output"

# CSV (auto-cleaned with pandas)
!python train_colab_v2.py --dataset ./convos.csv \
    --input_col "user" --output_col "bot"

# Transfer to Lua
import shutil
shutil.copytree("out_v2/checkpoints_v2", "./checkpoints_v2")
```

---

## Memory Budget vs Spec

| Component | Spec | This Implementation |
|---|---|---|
| Bloom Filter | 25 MB (20M×10bits) | ✓ 25 MB |
| Transition Core | 505 MB | ✓ ~480 MB (10M×48B avg) |
| Pattern Dictionary | 5 MB | ✓ 5 MB |
| Cold Archive | 489 MB disk | ✓ bucket-indexed |
| Case Library | 1.5 GB disk | ✓ block-compressed |
| **Total Memory** | **~541 MB** | **✓ ~510 MB** |
| **Total Disk** | **~1.99 GB** | **✓ on-demand** |

---

## Encoding Reference (state_store.lua §A)

| Encoding | Formula | Used For |
|---|---|---|
| FNV-1a 32 | `h=(h^b)×16777619 mod 2³²` | All state/N-gram keys |
| FNV-1a 64 | Same with 64-bit | Collision resolution |
| Varint | bit7=continue, bits0-6=payload | `total_count` in StateEntry |
| Log-prob 12 | `v=round(-log₂(p)×256)` | TransitionEntry probability |
| Case ID | `ts_48 \| seq_16` | Time-sortable, 65K cases/ms |
| CRC-32 | Castagnoli | Per-4KB block integrity |
