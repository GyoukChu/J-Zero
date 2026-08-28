# G-Zero — Local port (4 GPUs, paper-faithful LoRA)

The upstream repo runs entirely on the **Tinker cloud API**. This fork replaces
that backend with **local vLLM** (sampling + teacher-forced log-probs) and a
**local LoRA** training loop (Phase-1 GRPO + Phase-3 DPO), so the whole pipeline
runs on one 4-GPU node (vLLM tp=4) while reproducing the paper's setup (arXiv:2605.09959, Table 4).

## What changed vs. upstream

| Concern | Upstream (Tinker) | Local port |
|---|---|---|
| Sampling / log-probs | `tinker.SamplingClient` | `g_zero/local_backend.py` — `LocalSampler` over `vllm.LLM` (TP=4). `prompt_logprobs` gives the teacher-forced δ score. |
| Renderer / tokenizer | `tinker_cookbook` | `g_zero/local_render.py` — Qwen3 chat template via the model's own `apply_chat_template`. |
| Solver update (Phase 3) | Tinker LoRA DPO | `g_zero/local_train.py` — **LoRA DPO** (peft, rank 32); π_ref = base with the adapter disabled (`disable_adapter()`); length-normalized, q-only loss; adapter merged before saving. |
| Challenger GRPO (Phase 1) | Tinker GRPO | `g_zero/phase1.py` — **LoRA Dr.GRPO** (mean-only adv); rollouts + δ-scoring via a per-step vLLM subprocess (`g_zero/phase1_sample.py`), adapter trained resident on `cuda:0`; adapter merged before saving. |
| Model / tokens | Qwen3-8B-Base, 8192/16384 | Proposer 8192 / Generator 16384 with `chosen_max_chars=10k`, as in the paper. |

The defaults in `g_zero/config.py` reproduce paper Table 4: LoRA rank 32, β=2.0,
DPO lr 1e-5 / 50 steps / batch 8, Proposer lr 4e-5 / 6 steps / group 16 / batch
128 (= 2048 rollouts/step), num_questions 2000, [0,50] (bot50) δ-filter, all
quality filters. Only the *infra* differs from the paper (local vLLM + peft LoRA
instead of Tinker).

## Run it

```bash
# The paper runs in this repo (tags = runs/<tag>/):
bash scripts/lora_4b.sh                        # lora_4b_paper (Qwen3-4B, rounds 1-10)
bash scripts/lora_8b.sh                        # lora_8b_paper (Qwen3-8B, rounds 1-10)
bash scripts/lora_llama3b.sh                   # lora_llama3b_paper (Llama-3.2-3B, rounds 1-10)

# Generic entry points:
python -m g_zero.multi_round --num_rounds 2    # uses config.py defaults directly
```

Every multi-round run is resumable: state lives in `runs/<tag>/resume_state.json`,
and rerunning the same command continues from the next unfinished round.

Per round: Challenger (LoRA GRPO) generates `(q,h)` → previous Solver samples
`a_hard`/`a_assisted` and scores δ → bot50 + quality filter → LoRA DPO.
Artifacts under `runs/<tag>/round_<k>/`; each saved `final_model` is a
**merged** plain HF checkpoint (no adapter files) that vLLM loads directly.

## Environment notes (important)

- Hosts without a C/CUDA compiler (`gcc`/`nvcc`) are supported: vLLM JIT paths
  are disabled by default in `local_backend.py`: `VLLM_ATTENTION_BACKEND=FLASH_ATTN`,
  `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0`, `enforce_eager=True`.
- **peft + torchao 0.9 incompatibility:** peft 0.19's `is_torchao_available()`
  *raises* on torchao 0.9 (it wants >0.16) while dispatching every LoRA Linear.
  `g_zero/lora_compat.py` monkeypatches it to return `False` (torchao is unused —
  bf16, no quantization) so peft uses the normal Linear-LoRA path. This patch
  runs automatically before `get_peft_model`.
- Conda env: `verl` (torch 2.8 + vllm 0.11).
  `scripts/recreate_verl_env.sh` rebuilds it from scratch — note the hard pin
  `transformers==4.56.1`, the same pin as
  `requirements/requirements-train-vllm011.txt` (vllm 0.11 breaks on transformers 5.x).
- Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` for runs
  (once the model is pre-downloaded) to avoid HF-token-expiry 401s mid-run.

## Cost note

Phase 1 at paper scale is expensive locally: 2048 rollouts/step × 6 steps, each
generating a_hard up to 8192 tokens. Sampling goes through a per-step vLLM
subprocess (`g_zero/phase1_sample.py`) for continuous batching — eager HF
`.generate()` here would take days per round. Lower
`--phase1_max_tokens` or `--challenger_batch_size` to trade fidelity/throughput
for speed.
