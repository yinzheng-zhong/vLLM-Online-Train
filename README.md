# vllm-online-train

Train a speculative draft head on live production traffic, in the serving process, on
the serving GPU — or on a second GPU, if you have one to spare (see
[Train on a second GPU](#train-on-a-second-gpu)). It installs as a vLLM plugin: no
patched vLLM source, no separate
training job holding a second copy of the model, and no offline distillation dataset.

The reason I have done this is because for small organisations or individules that serves model with vLLM themselves might not be able to
afford to train a draft head or cannot find a pretrained head. Furthuremore, GPUs that is used to serve won't be fully ultilised most of the time.

**DFlash is the only head architecture implemented today**, meaning vLLM's
`method: "dflash"` speculator and checkpoints shaped like `z-lab/Qwen3-4B-DFlash-b16`.

The package splits in two by what each half talks to: `engine/` reads the serving
engine, `training/` knows nothing about it.

| | |
|---|---|
| **`engine/`** | **everything that touches vLLM** |
| `engine/hook/` | the foothold: the method patch, and the settings it loads |
| `engine/capture/` | engine step → rollout pool — head-agnostic |
| `engine/state/` | the borrowed target state, on the engine's device or a second one |
| **`training/`** | **everything that trains the head** |
| `training/collate/` | rollout pool → anchored replay batches — head-agnostic |
| `training/head/` | the DFlash module, its layers, its mask geometry and the target layers that feed it — DFlash-typed |
| `training/checkpoint/` | the on-disk format, export and hot publish — DFlash-typed |
| `training/loss/`, `training/optim/` | the objective, and the optimizer loop it runs under |
| `training/` (top) | the idle gate, the trainer thread, the metrics sink and the manager |
| `config/`, `contracts/` | the leaf layer both halves depend on |
| `step.py` | one engine step's activations, as the hook hands them over |
| `assembler.py` | the one wiring site for a training session |
| `cli.py` | `online-train-init-head` |

`capture/`, `collate/` and the idle gate move hidden states and time, not architecture.
`head/`, `checkpoint/` and the config resolver name DFlash, and are what a second
architecture would have to generalise.

`config/` and `contracts/` sit outside both halves because the hook needs the settings
at install time. That is what keeps `register()` off the training stack entirely — a
property `tests/test_wiring.py` pins.

## Why capture is cheap

When the engine runs `method: "dflash"` it already builds
`torch.cat([h[:T] for h in aux_hidden_states], dim=-1)` to feed the drafter — the
exact tensor the training objective needs as `target_features`. **Capture is a copy,
not a computation.**

And because `worker_base.py` calls `load_general_plugins()` *inside the worker
process*, a `vllm.general_plugins` entry point runs where the model, the target LM
head and the GPU already are. That is what makes the exact full-vocabulary teacher
affordable: the buffer holds the final hidden state (`[T, H]`, 5 KiB/token at
H=2560) and the teacher logits are regenerated at train time through the target's
own `compute_logits`. Buffering `[T, 151936]` logits instead would cost 297 KiB
per token.

Two things are *not* cheap, and it is worth being precise about them:

- **Time is scavenged, not spare.** A training step is a full GPU step taken in the
  wall-clock gaps between engine steps, admitted by the idle gate. Nothing here
  reclaims idle SMs *within* a serving step.
- **Less available memory for KV cache.**

## Install

```bash
uv pip install -e /path/to/vllm-online-train
```

The entry point is live the moment it is installed, which is why `VLLM_PLUGINS`
matters (see below).

## 1. Materialise a head

vLLM has no random-init path: `speculative_config.model` must point at a real
directory. So "ship from a random head" starts by writing one.

```bash
online-train-init-head \
    --target Qwen/Qwen3-4B \
    --out ./heads/qwen3-4b-3l-v0 \
    --layers 3 --features 5 --block-size 16
```

No target *weights* are read, only its config — the serving path binds the target's
own `embed_tokens` and `lm_head` onto the draft when the checkpoint omits them, so
there is nothing to copy and this runs on CPU.

`--layers` and `--features` are independent. `--features` is the one that costs at
runtime: it sets `fc`'s width and the per-token capture cost
(`(features + 1) × hidden × 2` bytes).

## 2. Serve it

```bash
vllm serve Qwen/Qwen3-4B \
    --speculative-config '{"method":"dflash","model":"./heads/qwen3-4b-3l-v0","num_speculative_tokens":15}' \
    --gpu-memory-utilization 0.62 \
    --max-model-len 2048 \
    --enforce-eager
```

`num_speculative_tokens` must be `block_size - 1`: slot 0 of a block re-feeds the
anchor and predicts nothing, so a K-slot block yields K-1 tokens.

Acceptance will be at floor — the head is random. That is the point of this step:
it proves the checkpoint format before any training code runs.

## 3. Turn training on

```bash
VLLM_PLUGINS=online_train \
ONLINE_TRAIN_CONFIG=./train.json \
vllm serve Qwen/Qwen3-4B \
    --speculative-config '{"method":"dflash","model":"./heads/qwen3-4b-3l-v0","num_speculative_tokens":15}' \
    --gpu-memory-utilization 0.62 --max-model-len 2048 --enforce-eager
```

`VLLM_PLUGINS` is the on/off switch: unset loads every discovered plugin, set loads
only the named ones. Setting it explicitly is how to A/B against a clean baseline
without uninstalling.

`ONLINE_TRAIN_CONFIG` takes a path to JSON or inline JSON. Pointing it at a config
is itself the opt-in; you do not also need `enabled: true` inside.

## 4. Watch the right number first

Metrics live in the **worker** process, so `/metrics` cannot see them. They go to
the JSONL sink from `metrics_path`, one file per worker pid:

```bash
tail -f ./online-train.<pid>.jsonl | jq -c '{step, loss:."loss/total", top1:."train/top1_agree", gate:."gate/open_fraction", rollouts:."buffer/rollouts"}'
```

**Watch `gate/open_fraction` before you watch the loss.** If the gate rarely opens
under your traffic then nothing downstream of it matters, and that is a property of
the deployment rather than the head.

Then `buffer/dropped_*`: a high `dropped_prefix_cache_hit` means prefix caching is
eating your training data (see Traps).

## 5. Adopt what it learned

Restart-to-adopt first, always:

```
serve v0 → traffic fills buffer → gate opens → trainer steps
         → checkpoint_dir gets v1 → restart pointing at v1
```

The exported directory carries a verbatim copy of the running `config.json`, so it
is a drop-in replacement. Restart-to-adopt cannot be subtly wrong: the process
either loads the directory or dies.

Only then consider `publish_mode: "hot"`, and validate it by asserting acceptance
equality against `serve(v1)`. The reason for that order is that a bad hot publish is
**silent** — speculation is always verified against the target, so stale draft
weights can never produce a wrong token, they only sag the acceptance rate, which is
indistinguishable from a badly-trained head.

## Train on a second GPU

`train_device` moves the head, its gradients, its optimizer state and its batches onto
another device. The serving GPU keeps only the capture, so the trainer's footprint
stops coming out of `gpu_memory_utilization`:

```json
{ "train_device": "cuda:1" }
```

```bash
VLLM_PLUGINS=online_train ONLINE_TRAIN_CONFIG=./train.json \
vllm serve Qwen/Qwen3-4B \
    --speculative-config '{"method":"dflash","model":"./heads/qwen3-4b-3l-v0","num_speculative_tokens":15}' \
    --gpu-memory-utilization 0.85 --max-model-len 2048 --enforce-eager
```

The rollout buffer was already host memory, so it is the device boundary for free: a
training step reads pinned host tensors and sends them to GPU 1, and nothing comes
back. What does not come for free is the **teacher**. Scoring it through the engine's
own `compute_logits` would put a `[N, 151936]` projection back on the serving GPU and
then drag the logits across the bus twice per step, once for the forward and once for
the recompute. So `state/mirrored.py` holds its own copy of the target's vocabulary
projection on the training device (0.72 GiB at bf16) and projects there.

That copy is a bare matmul against the LM head, which is what vLLM's
`LogitsProcessor` reduces to for Qwen3 and **not** what it reduces to in general — a
logit scale, a soft cap or a quantised head would each make it the wrong teacher, and a
wrong teacher trains silently. So at startup both projections score the same four
random states and the plugin refuses to run if they disagree by more than 2% of the
largest logit. The deviation is in the log line either way:

```
Online training on cuda:1, off the engine's cuda:0: the teacher projection is
mirrored there at torch.bfloat16 and matches the engine's own to 0 of its largest
logit
```

Three things to know before reaching for it:

- **The index is the worker's, not `nvidia-smi`'s.** `CUDA_VISIBLE_DEVICES` narrows and
  renumbers what the plugin can see, and vLLM sets it per process for data-parallel
  deployments. An index that is not visible fails at startup rather than at the first
  allocation. `null` and a bare `"cuda"` both mean the serving device; only an explicit
  index moves anything.
- **The idle gate becomes a throughput knob.** On the serving GPU it is what bounds the
  latency a training step can add. On a second GPU there is no step to collide with, so
  `idle_ms: 0` is a defensible setting — the trainer and the engine share only the
  rollout pool, which the engine writes and the trainer reads.
- **A hot publish still crosses back.** `publish_mode: "hot"` copies 672 MiB of trained
  tensors into the drafter's parameters on the serving GPU, staged one tensor at a
  time. Leave it ~1 GiB of headroom, or stay on restart-to-adopt.

## Memory budget (Qwen3-4B, 3 layers, 5 features, 24 GiB card)

| item | GiB | with `train_device` |
|---|---|---|
| target bf16 (4.02B) | 7.49 | serving |
| draft head bf16 (336M) | 0.63 | serving |
| trainable head fp32 (336M) + shared bf16 (778M) | 2.70 | training |
| gradients fp32 | 1.25 | training |
| AdamW moments | 2.50 | training |
| mirrored teacher projection bf16 (389M) | 0.72 | training only |
| **subtotal** | **14.57** | 8.12 serving / 7.17 training |
| KV cache + training activations | remainder | |

The trainer allocates *after* vLLM's memory profiling, so `gpu_memory_utilization`
has to leave it room. The trainer-side rows above come to 6.45 GiB, and the
activations push it to roughly 8 GiB, so budget `(1 - util) × total ≥ ~8 GiB`. On a
24 GiB card that means **util ≤ 0.66**; 0.62 is a comfortable setting.

On a second GPU that constraint goes away and **util ≤ 0.9** is reasonable, since what
is left on the serving card is the capture's device-side gather and a hot publish. The
training card wants ~9 GiB: 7.17 above plus activations, and a peak while the borrowed
embedding table and LM head are held in fp32 on the way to their storage dtype.

Levers, in order of effect:

- `kl_chunk_size` — the dominant activation. A `[chunk, 151936]` fp32 log-softmax is
  ~296 MiB per chunk per term. 512 → ~1.2 GiB, 128 → ~300 MiB.
- `sequences_per_step` × `anchors_per_sequence` — the batch. 2 × 8 with
  `grad_accum_steps: 4` gives the same effective batch as 8 × 8 at a quarter of the
  peak.
- `--features 3` — cuts `fc` from 12800 to 7680 wide and capture from 30 to 20
  KiB/token. Requires regenerating the head.
- `shared_dtype: "float32"` doubles the borrowed embed/LM-head cost (1.45 → 2.90
  GiB) and exists only for objective-parity work against an fp32 reference.

`buffer_capacity_tokens` is **host** RAM: 30 KiB/token here, so 200k tokens is
5.7 GiB of pageable host memory. 60k (1.7 GiB) is a saner default on a workstation.

## Traps

Each of these still trains while silently capping acceptance.

1. **Valid positions.** In a verify step with `r` accepted, only the first `r+1`
   positions are conditioned on real tokens; the rest sit on a rejected token and
   are off-policy. `r+1` is exactly the non-`-1` count of the sampled row.
2. **Slot 0.** A K-slot block yields K-1 tokens. Getting this wrong trains a
   parallel next-token predictor that drafts one extra token and flatters the
   acceptance rate.
3. **Prefix-cache holes.** A cache hit computes no hidden states for the cached
   prefix, so the draft's context KV cannot be built. The capture path drops those
   requests (`skip_prefix_cache_hits`) and counts them. Under heavy prefix reuse this
   can drop most of your traffic — check the metric before concluding the gate is the
   problem.
4. **Layer-id off-by-one.** vLLM adds 1 to DFlash's `target_layer_ids`. Always read
   the resolved aux ids.
5. **`inference_mode` escape.** Tensors created under `inference_mode` cannot join
   an autograd graph and `.clone()` inherits the flag. Capture escapes by `copy_`-ing
   into buffers allocated outside it.
6. **Publish aliasing.** `_build_context_kv_buffers` reassigns `torch.cat`
   snapshots at new addresses; `checkpoint/weight_publisher.py` restores the originals.
7. **Traffic memorisation.** The head trains on production traffic and the weights
   carry it. This is opt-in and must never be default-on.

## Version pinning

This wraps the private `GPUModelRunner.propose_draft_token_ids`. When the patch
installs — inside the worker, at startup — the signature is asserted against
`SignatureGuard.EXPECTED_PARAMS` and the patch **refuses to install** on any
mismatch, so a vLLM bump fails loudly rather than silently capturing the wrong
tensor. Re-verify on every bump.

## Tests

```bash
python -m pytest tests -q                    # 337 tests
python -m pytest tests -q -m "not slow"      # skip the real-module load tests
```

Run from the repo root; `vllm` has to be importable. The `slow` tests additionally
need `Qwen/Qwen3-4B`'s config in the local HF cache, and skip without it.

`tests/` mirrors the package tree. The load-bearing file is
`tests/head/test_serving_parity.py`: it builds a real `DFlashQwen3ForCausalLM`
under a real `VllmConfig` and asserts that every trained tensor in an `init_head`
checkpoint lands, that only the target-bound tensors do not, and that a 5-layer
head is name-, shape- and dtype-identical to the published
`z-lab/Qwen3-4B-DFlash-b16`.
