# G-Zero: Self-Play for Open-Ended Generation from Zero Data

> **Note (this fork):** the pipeline here runs **fully locally** (vLLM sampling +
> peft LoRA training) instead of the upstream Tinker cloud API. See
> [**What this fork changes vs. the official repo**](#what-this-fork-changes-vs-the-official-repo)
> below for a file-by-file map of every change, and [LOCAL.md](LOCAL.md) for
> environment notes.


> Self-play preference optimization for open-ended generation, with **no
> ground-truth answers, no LLM judge in training, and no majority vote**.
> The training signal is the **hint-induced log-prob shift δ** — purely
> intrinsic, computed from the Solver itself.

The Challenger writes a request `q` and a short hint `h`. The Solver
produces an un-assisted response `a_hard ~ π(·|q)`. We measure how much
the hint shifts the Solver's distribution away from `a_hard`:

```
δ(q, h, a_hard) = mean_t log π(a_hard_t | q, a_hard_<t)
                − mean_t log π(a_hard_t | q, h, a_hard_<t)
```

The Solver is then DPO-trained on `chosen = a_assisted` vs.
`rejected = a_hard`, conditioned on **only `q`** — internalizing the
hint-induced style improvement so that hints are no longer needed at
test time.

---

## What this fork changes vs. the official repo

Upstream: [Chengsong-Huang/G-Zero](https://github.com/Chengsong-Huang/G-Zero).
**TL;DR — same algorithm, different execution backend.** The upstream release
runs every GPU-touching step (sampling, log-probs, GRPO, DPO) on the
[Tinker](https://thinkingmachines.ai/tinker) cloud API. This fork runs the
identical pipeline **fully locally**: [vLLM](https://github.com/vllm-project/vllm)
for sampling and teacher-forced log-probs, [peft](https://github.com/huggingface/peft)
LoRA (rank 32, as in paper Table 4) for training. The algorithm itself is
untouched: same δ definition, same bot50 filter and quality filters, same
Dr.GRPO mean-only advantages, same DPO loss (length-normalized log-ratios,
q-only prompt).


### New files (the local backend)

| File | What it does |
|---|---|
| [`g_zero/local_backend.py`](g_zero/local_backend.py) | `LocalSampler` — wraps `vllm.LLM` for batched sampling and teacher-forced log-probs (via `prompt_logprobs`). Replaces `tinker.SamplingClient`. TP size / KV fraction / ctx length come from `Config`. |
| [`g_zero/local_render.py`](g_zero/local_render.py) | Prompt renderer + tokenizer built on the model's **own** `apply_chat_template`. Replaces the `tinker_cookbook` renderers; works unmodified for Qwen3 and Llama-3.2-Instruct. |
| [`g_zero/local_train.py`](g_zero/local_train.py) | The LoRA **DPO** training loop (peft). π_ref = the same base model with the adapter disabled (`disable_adapter()`) — no second weight copy. The adapter is merged into the base before saving, so every checkpoint is a plain HF model that vLLM loads directly. |
| [`g_zero/lora_compat.py`](g_zero/lora_compat.py) | LoRA wrap/merge helpers + a workaround for a peft-0.19 × torchao-0.9 incompatibility (details in [LOCAL.md](LOCAL.md)). |
| [`g_zero/phase1_sample.py`](g_zero/phase1_sample.py) | Per-GRPO-step vLLM **sampling subprocess** for Phase 1 (Challenger rollouts + δ-scoring). It exits after each step, releasing all vLLM GPU memory before the resident adapter takes its gradient step. |
| [`LOCAL.md`](LOCAL.md) | Detailed fork change log: memory layout, environment gotchas, cost notes. |

### Rewritten files (Tinker → local)

| File | Upstream (Tinker) | This fork |
|---|---|---|
| [`g_zero/phase1.py`](g_zero/phase1.py) | Challenger GRPO through Tinker's training API | **LoRA Dr.GRPO** run locally: the adapter trains resident on `cuda:0`; rollouts + δ come from the `phase1_sample.py` vLLM subprocess. Same reward (δ − length hinge − BLEU cluster share), same mean-only advantages, −1 floor for invalid rollouts. |
| [`g_zero/phase2.py`](g_zero/phase2.py) | Challenger + Solver as two live Tinker clients | Same pool-building logic; the two vLLM samplers are loaded **sequentially** (generation fully precedes scoring) so only one engine is resident at a time. |
| [`g_zero/phase3.py`](g_zero/phase3.py) | Full DPO implementation against Tinker forward/backward | Thin wrapper delegating to [`local_train.py`](g_zero/local_train.py); identical DPO math and call signature. |
| [`g_zero/hint_delta.py`](g_zero/hint_delta.py) | Async per-pair Tinker calls | All prompts batched through one `LocalSampler` call (vLLM batches internally). Same δ formula, same `delta_only` / `full` modes. |
| [`g_zero/multi_round.py`](g_zero/multi_round.py) | Outer round loop, resumable | Same loop + resume semantics. |
| [`g_zero/config.py`](g_zero/config.py) | Tinker model/renderer fields | Adds LoRA + vLLM engine fields; several defaults changed (table below). |
| Environment | `tinker`, `tinker-cookbook` | `torch`, `vllm`, `transformers`, `peft`, `accelerate` (see `../../requirements/requirements-train-vllm011.txt` or `scripts/recreate_verl_env.sh`). |

**Identical to upstream:** `parse.py` (byte-identical) and `bleu_penalty.py`
(code-identical; docstring reworded). Upstream's
single-round `main.py` driver, ablation scripts, and eval code are not shipped
— this release carries exactly the multi-round paper pipeline.

### Changed defaults in `config.py`

| Field | Upstream | This fork | Why |
|---|---|---|---|
| `challenger_batch_size` | 8 | **128** | Paper Table 4 (128 groups × 16 rollouts = 2048 rollouts/step). The upstream default was a cheap-demo setting. |
| `solver_max_tokens` | 8192 | **16384** | Generator max_tokens per paper Table 4. |
| `lora_alpha` / `lora_dropout` / `lora_target_modules` | *(new)* | 64 / 0.0 / all attn+MLP proj | Standard LoRA setup around the upstream `lora_rank=32`. |
| `tensor_parallel_size`, `gpu_memory_utilization`, `vllm_max_model_len`, `dtype` | *(new)* | 4 / 0.7 / 32768 / bf16 | vLLM engine knobs (one 4-GPU node). |
| `phase1_max_tokens`, `challenger_gen_sub_batch` | *(new)* | 8192 / 8 | Phase-1 `a_hard` cap (Table 4); the sub-batch knob is unused by the vLLM sampling path. |

### The runs this code produced

| Run tag (`runs/<tag>`) | Model | Launch |
|---|---|---|
| `lora_4b_paper` | Qwen/Qwen3-4B-Base | [`scripts/lora_4b.sh`](scripts/lora_4b.sh) |
| `lora_8b_paper` | Qwen/Qwen3-8B-Base | [`scripts/lora_8b.sh`](scripts/lora_8b.sh) |
| `lora_llama3b_paper` | meta-llama/Llama-3.2-3B-Instruct | [`scripts/lora_llama3b.sh`](scripts/lora_llama3b.sh) |

The exact conda environment behind these runs can be rebuilt with
[`scripts/recreate_verl_env.sh`](scripts/recreate_verl_env.sh) — note the hard
pin `transformers==4.56.1`, matching
[`requirements/requirements-train-vllm011.txt`](../../requirements/requirements-train-vllm011.txt)
(vLLM 0.11 breaks on transformers 5.x).

---

## Why δ instead of a reward model or judge

Most "self-play" pipelines in the open-ended regime depend on either a
ground-truth verifier (RLVR), a frozen LLM judge (RLAIF, Self-Rewarding,
DICE), or majority-vote pseudo-labels (R-Zero, SPIN). All three break in
the open-ended setting: no verifier exists, judges are slow / biased /
position-sensitive, and majority vote presupposes a single right answer.

δ is **intrinsic to the Solver itself**:

- High δ ⇒ the hint reveals an answer the Solver wouldn't have produced
  un-aided ("answer leakage"). DPO over high-δ pairs teaches the Solver
  to copy from hints — hurts no-hint generalization.
- Low δ (≈0) ⇒ the hint provides structural framing without revealing
  the answer. DPO over low-δ pairs distills a **style shift** — what
  actually transfers.

Empirically, δ is positive in ~99% of valid (q, h) pairs. We retain the
**lower half** of the empirical δ distribution (the `bot50` filter) —
`(q, h)` pairs whose hint nudges structure without leaking the answer.

---

## Method

A single training round consists of three interacting phases. The
Challenger generates `(q, h)` pairs; the Solver produces an un-assisted
response `a_hard ~ π(·|q)`; we compute the hint-induced log-prob shift
δ; and the Solver is DPO-trained on a δ-filtered subset of the pool.

### Phase 1 — Challenger (GRPO, optional)

The Challenger is GRPO-trained against:

```
r(q, h) = δ(q, h, a_hard)
        − λ_len · max(0, |h| − L_tgt) / 100         # length hinge (λ=0.03, L=200)
        − BLEU_cluster_share(q, batch)              # diversity penalty
```

We use Dr.GRPO mean-only advantages. Without the length hinge, the
Challenger reward-hacks via verbose hints; without the BLEU cluster
penalty it collapses to a single high-δ template. The
`--run_phase1 false` ablation (drops Phase 1 entirely; uses the base
model directly as the Challenger) matches Phase-1-on R1 within noise on
Qwen3-8B-Base.

### Phase 2 — Build the DPO pool

For each Challenger-generated `(q, h)`:

1. Sample `a_hard ~ π(·|q)` and `a_assisted ~ π(·|q, h)`.
2. Compute `δ` via `compute_logprobs` of `a_hard` under both contexts.
3. Cache the full scored pool (`raw_pool.jsonl`) so re-filtering is free.
4. Apply δ-percentile filter `[pct_low, pct_high]` (paper-main = bot50)
   plus structural quality filters (length, repetition via zlib ratio,
   prompt echo, role-marker prefix).

### Phase 3 — DPO the Solver

Standard DPO-sigmoid (Rafailov 2023) with two specifics:

1. **Length-normalized log-ratios**: `dot(logp, mask) / mask_sum` so a
   longer `chosen` doesn't dominate the gradient.
2. **DPO prompt is `q` only, no hint**. We distill the hint-assisted
   behavior into the Solver's q-only conditional, so the trained Solver
   no longer needs hints at test time.

`π_ref` is the Solver's weights at the start of Phase 3 (in this fork:
the same base model with the LoRA adapter disabled — no second copy).

---

## Setup

This fork runs on **local GPUs** (one 4-GPU node) with
Python 3.12, torch 2.8, vllm 0.11, peft 0.19:

```bash
uv venv --python 3.12 <venv> && uv pip install --python <venv>/bin/python \
  -r ../../requirements/requirements-train-vllm011.txt
# or reproduce the exact pinned conda env used for the paper runs:
bash scripts/recreate_verl_env.sh
```

Notes:

- **`transformers` must stay on the 4.x line** (`==4.56.1`, the pin in
  `requirements/requirements-train-vllm011.txt`): vLLM 0.11 declares only a
  lower bound, and transformers 5.x removes an API vLLM's tokenizer cache
  depends on.
- No `TINKER_API_KEY` — nothing talks to Tinker anymore.
- Once models/datasets are cached, export
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` so an
  expired HF token can't 401 a multi-day run midway.
---

## Reproducing the paper runs

One script per released run — 10 rounds each, all hyperparameters from the
paper-faithful `config.py` defaults:

```bash
bash scripts/lora_4b.sh          # Qwen3-4B-Base      (tag=lora_4b_paper)
bash scripts/lora_8b.sh          # Qwen3-8B-Base      (tag=lora_8b_paper)
bash scripts/lora_llama3b.sh     # Llama-3.2-3B-Instruct (tag=lora_llama3b_paper)
```

Every run is resumable (`runs/<tag>/resume_state.json`) — rerun the same
script to continue after a crash. Checkpoints are evaluated post-hoc by the
14-benchmark suite at the repo root (`../../benchmarks/`).

### Outputs

Multi-round runs (`python -m g_zero.multi_round`) write one directory per round.
Every `final_model/` is a **merged plain-HF checkpoint** (LoRA adapter already
folded into the base weights) that vLLM — or anything else — loads directly.

```
runs/<tag>/
├── resume_state.json             # multi-round resume pointer (crash-safe)
├── rounds_summary.json           # per-round artifact paths for all finished rounds
└── round_<k>/
    ├── challenger/
    │   ├── final_model/          # merged HF checkpoint of the trained Challenger
    │   ├── metrics.jsonl         # reward / δ / valid_frac / hint length per step
    │   ├── _adapter/             # last un-merged LoRA adapter (the sampling snapshot
    │   │                         #   phase1.py hands to the vLLM subprocess each step)
    │   └── _sample_work/         # phase1.py's per-step vLLM sampling scratch
    │                             #   (step<k>/ work dirs + step<k>.json rollouts)
    ├── raw_pool.jsonl            # Phase 2 full scored pool (q, h, a_hard, a_assisted, δ)
    ├── dpo_data.jsonl            # Phase 2 filtered (prompt, chosen, rejected, δ)
    └── solver/
        ├── final_model/          # merged HF checkpoint of the trained Solver
        └── metrics.jsonl         # DPO loss / accuracy / margin per step
```

The two leading-underscore directories are Phase-1 scratch, not checkpoints:
`_adapter/` is the un-merged adapter left behind by the last sampling snapshot
(the merged weights are in `final_model/`; the Solver's LoRA path merges before
saving and never writes one), and `_sample_work/` holds the vLLM sampling
subprocess' per-step working dirs and rollout JSON. Both are safe to delete
once a round has finished.

---

## Codebase layout

Legend: ★ new in this fork · ● rewritten for the local backend · unmarked =
code-identical to upstream (comments/docstrings may differ).

```
.
├── README.md                  ← this file
├── LOCAL.md                   ★ fork change log + environment notes
├── scripts/
│   ├── lora_4b.sh                   ★ lora_4b_paper (Qwen3-4B, rounds 1–10)
│   ├── lora_8b.sh                   ★ lora_8b_paper (Qwen3-8B, rounds 1–10)
│   ├── lora_llama3b.sh              ★ lora_llama3b_paper (Llama-3.2-3B, rounds 1–10)
│   └── recreate_verl_env.sh         ★ rebuild the pinned conda env
└── g_zero/
    ├── config.py              ● every hyperparameter (+ LoRA & vLLM fields)
    ├── prompts.py             ← Challenger / Solver prompts
    ├── parse.py               ← <question>/<hint> XML extractor
    ├── bleu_penalty.py        ← BLEU-cluster duplication penalty (Phase 1)
    ├── local_backend.py       ★ LocalSampler over vllm.LLM (sampling + logprobs)
    ├── local_render.py        ★ chat-template renderer / tokenizer
    ├── local_train.py         ★ LoRA DPO training loop (peft)
    ├── lora_compat.py         ★ LoRA wrap/merge + peft×torchao workaround
    ├── hint_delta.py          ● δ via local vLLM teacher-forced logprobs
    ├── phase1.py              ● Challenger LoRA Dr.GRPO (local)
    ├── phase1_sample.py       ★ Phase-1 vLLM sampling subprocess
    ├── phase2.py              ● (q, h) generation + δ-scoring + filtering
    ├── phase3.py              ● Solver DPO (thin wrapper over local_train)
    └── multi_round.py         ● outer loop over rounds
```

Each phase is a stand-alone module: nothing in `phase2.py` imports from
`phase3.py`. This makes it easy to swap in a different Solver update rule
(e.g. on-policy distillation in place of DPO) — see the docstrings.

---

## Caching & resumability

- **Phase 2 pool**: `raw_pool.jsonl` is cached. To change `pct_low /
  pct_high / chosen_*` filters, delete `dpo_data.jsonl` first — its cache
  is checked before the filters run — then re-running re-runs only the
  filter step.
- **DPO data**: `dpo_data.jsonl` is similarly cached.
- **Solver / Challenger checkpoints**: each phase saves a merged plain-HF
  `final_model/` directory, loadable by vLLM (or anything else) directly.
- **Multi-round**: per-round resume state in `runs/<tag>/resume_state.json`
  — re-running the same command picks up at the next round.

---

## Configuration knobs that matter

The defaults in [`g_zero/config.py`](g_zero/config.py) reproduce the
paper-main numbers. Knobs that are *load-bearing*:

| Field                            | Default | Why it matters |
|----------------------------------|---------|----------------|
| `pct_low`, `pct_high`            | 0, 50   | bot50 cutoff. Upper tail = answer leakage. |
| `hint_length_target_chars`       | 200     | Length-hinge target — without it, Challenger reward-hacks. |
| `hint_length_penalty_lambda`     | 0.03    | Hinge weight. Larger → shorter hints → less signal. |
| `dpo_beta`                       | 2.0     | DPO temperature; lower-than-typical because the chosen / rejected gap is small in the open-ended regime. |
| `dpo_lr`                         | 1e-5    | DPO has a narrow, model-dependent sweet spot. |
| `solver_max_tokens`              | 16384   | Generator cap (paper Table 4), with `chosen_max_chars=10000`. |

---

## Citation

```bibtex
@misc{huang2026gzeroselfplayopenendedgeneration,
      title={G-Zero: Self-Play for Open-Ended Generation from Zero Data},
      author={Chengsong Huang and Haolin Liu and Tong Zheng and Runpeng Dai and Langlin Huang and Jinyuan Li and Zongxia Li and Zhepei Wei and Yu Meng and Jiaxin Huang},
      year={2026},
      eprint={2605.09959},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.09959},
}
```

---
