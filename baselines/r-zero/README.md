# R-Zero (verl)

Reproduction of *R-Zero: Self-Evolving Reasoning LLM from Zero Data* on top of verl, using
4 B200 GPUs (indices 0,1,2,3) and Qwen3-4B/8B-Base.

## Structure

```
recipe_R-Zero/
  run_rzero.sh                   # Iter 1..3 orchestrator
  challenger_train.sh            # vLLM serve + verl GRPO (challenger)
  solver_train.sh                # question gen + filter + verl GRPO (solver)
  evaluate/run_solver_eval.sh    # 7-dataset eval + local recheck (see RUNS.md)
  reward/
    challenger_reward.py         # paper-spec: max(0, 1 - 2|p_hat - 0.5| - r_rep)
    solver_reward.py             # binary pseudo-label match (math_verify, not mathruler)
  vllm_service/
    start.sh                     # reward servers (RZERO_REWARD_GPUS, ports RZERO_BASE_PORT+i)
    start_vllm_server.py         # Flask + vLLM (solver pseudo-label batch)
  question_gen/
    question_generate.py         # vLLM question generator
    run_parallel.sh              # 4 GPU parallel (0,1,2,3)
  question_filter/
    evaluate.py                  # solver answers m=10 + heuristics + delta band
    run_parallel.sh              # 4 GPU parallel
    upload_to_parquet.py         # concat shards -> verl parquet (no HF push)
  data/
    dummy_train.parquet          # static prompts for challenger GRPO trainer (rebuilt each run by challenger_train.sh)
    build_dummy.py
```

## Conventions

- Local checkpoints saved to `${VERL_REPO}/checkpoints/R-Zero-Repro/<exp_name>/`.
- HF-format actor weights produced via `python -m verl.model_merger merge --backend fsdp`.
- Generated artifacts (questions, eval, logs): `$OUT_ROOT/iter{N}_<tag>/`
  (`OUT_ROOT` must be exported before running).
- WandB project `R-Zero-Repro`; exp names `iter{N}_challenger`, `iter{N}_solver`.

## Run

```
bash recipe_R-Zero/run_rzero.sh
```

Smoke test variant (1 challenger step, 100 questions, 1 solver step):
```
SMOKE=1 bash recipe_R-Zero/run_rzero.sh
```

## Hyperparameters (paper-aligned, Appendix B.1.1 / B.1.2)

- Challenger: GBS=16 (GBS=128 in paper, but GBS=16 in official repo), top_p=0.99, **n=4** rollout, **5 steps** (ckpt@5),
  reward = `max(0, 1 - 2|p_hat - 0.5| - r_rep)`; BLEU clust threshold τ=0.5, λ=1.
  Format-fail (no `<question>` + `\boxed{}`) → `-1 - r_rep` (R-Zero official, paper silent).
- Solver: GBS=128, **top_p=0.99**, **n=5** rollout, **15 steps** (ckpt@15),
  binary pseudo-label match reward.
- Reward server (solver inference) and Question filter:
  temperature=1.0, **top_p=0.99**, n=m=10 (paper). R-Zero code uses top_p=1.0 + top_k=40
  which deviates from paper; we follow paper.
- Filter: δ=0.25 band [0.25, 0.75], drop questions with `'证明' in q`,
  `'box' in q.lower()`, or `'text' in a.lower()` (R-Zero official heuristics).
- m (solver answers per question) = 10. N (questions per iter) = 8000 (4 GPU × 2000).
- lr 1e-6, kl_loss_coef 1e-2, clip [0.20, 0.28] (paper does not specify clip).
- Iterations = 3 (both the launchers and the bare script default to `RZERO_NITER=3`,
  the paper-baseline schedule; export `RZERO_NITER=10` for the R-Zero paper's own schedule).
- In-training validation: **DISABLED** (verl routes val through `solver_reward.py`
  = rule-based grading only, which under-scores). Authoritative eval is `evaluate/run_solver_eval.sh`,
  combo 4 = verl math_verify + LLM recheck against the LOCAL self-hosted judge
  (Qwen3.6-27B via vLLM; no external API).
- Eval: greedy decoding (temp=0, n=1) for math500 / minervamath / olympiadbench / gsm8k.
  Sampling (temp=0.6, top_p=1.0, n=32) for amc23 / aime24 / aime25 (mean@32 per paper).
  max_tokens=4096 throughout.
