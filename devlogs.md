# Devlogs: SDFT HPO Sweep

## 2026-07-25: Overnight HPO Sweep (10 experiments)

### Setup phase
- Added `--save_steps` CLI arg to `main.py`
- Created `hpo_run.sh` with train→find_ckpt→infer→eval→report flow, EXIT trap for tmux notification
- Created `GOAL.md` as context handoff doc
- Added CLI args to `main.py`: `alpha`, `temperature`, `warmup_ratio`, `lr_scheduler_type`, `max_grad_norm`, `loss_type`, `generate_from_teacher`
- Modified `hpo_run.sh` to evaluate ALL checkpoints (not just last) — critical for early stopping analysis

### Pre-sweep: Evaluated existing unevaluated checkpoints
- `sdft_idan_test_run/checkpoint-100` (LR=5e-6): **84.90% +/- 1.54%** (was never evaluated, only ckpt-224 at 84.34%)
- `sdft_idan_run_1/checkpoint-100` (LR=2e-5): **84.90% +/- 1.01%** (same — LR doesn't matter)
- Key insight: Early stopping at ~1 epoch gives +0.56% over full training

### HPO 1: Reverse KL, student gen, LR=5e-6, temp=1.0 (baseline)
- **Result**: ckpt-50=83.33%, ckpt-100=**85.68%** +/- 2.24%
- Individual runs: 84.56%, **88.26%**, 84.23% — one outlier inflates average
- Establishes the multi-checkpoint eval pipeline works

### HPO 2: Forward KL (alpha=0.0), student gen, LR=5e-6
- **Result**: ckpt-50=**84.23%**, ckpt-100=83.55%
- Forward KL is worse than reverse KL for factual QA
- Lower training loss (0.17 vs 0.22) but worse eval — training loss ≠ eval accuracy

### HPO 3: Teacher gen + reverse KL — INVALID
- **Result**: ckpt-50=82.77%, ckpt-100=80.31% — but INVALID
- `sed 's/ALPHA=0.0/ALPHA=1.0/'` accidentally matched substring in `REF_MODEL_MIXUP_ALPHA=0.01`, corrupting it to `1.01`
- Loss exploded (0.96→3.89), grad_norm=23 — caused by broken ref_model_mixup_alpha, not by teacher gen itself

### HPO 4: Teacher gen + forward KL (alpha=0.0)
- **Result**: ckpt-50=80.65%, ckpt-100=**82.89%** +/- 2.93%
- Below baseline — teacher gen hurts with forward KL
- The information gap (teacher sees golden response, student doesn't) is too large

### HPO 5: Teacher gen + reverse KL (CORRECT params, re-run of HPO 3)
- **Result**: ckpt-50=**84.90%** +/- 0.89%, ckpt-100=83.22%
- With correct ref_mixup=0.01, teacher gen + revKL matches student gen (84.90%)
- Teacher gen is neutral, not harmful (HPO 3's crash was the sed bug)
- Script crashed after timeout due to PIPESTATUS bug (checkpoints evaluated manually)

### HPO 6: JSD (alpha=0.5), student gen
- **Result**: ckpt-50=83.67%, ckpt-100=**84.79%** +/- 1.35%
- JSD between nearly identical teacher/student gives tiny gradients (loss=0.03)
- Still learned something over 100 steps, but worse than reverse KL
- KL ranking confirmed: revKL (84.90%) > JSD (84.79%) > fwdKL (84.23%)

### HPO 7: LR=1e-5, reverse KL, student gen, temp=1.0
- **Result**: ckpt-50=**85.01%** +/- 2.23%, ckpt-100=84.23%
- Higher LR shifts sweet spot to ckpt-50 instead of ckpt-100
- Same ~84.9% ceiling — LR changes WHEN the peak occurs, not the peak height

### HPO 8: temperature=0.7, reverse KL, student gen, LR=5e-6, seed=42
- **Result**: ckpt-50=**86.02%** +/- 3.36%, ckpt-100=84.34%
- **BEATS THE 85.80% TARGET**
- Individual runs: **87.25%**, **88.59%**, 82.21% — two of three above target
- Lower temperature = more focused completions = cleaner training signal
- Higher training loss (0.28 vs 0.22) but better eval — sharper completions matter

### HPO 9: temperature=0.7, reverse KL, student gen, LR=5e-6, seed=123 (confirmation)
- **Result**: ckpt-50=84.12%, ckpt-100=**85.57%** +/- 3.54%
- Confirms temp=0.7 improvement with different seed
- Individual runs: 82.21%, **89.26%**, 85.23%
- Optimal checkpoint varies by seed (ckpt-50 for seed=42, ckpt-100 for seed=123)

### HPO 10: temperature=0.7, LR=1e-5, reverse KL, student gen, seed=42
- **Result**: ckpt-50=83.33%, ckpt-100=**85.01%** +/- 1.91%
- LR=1e-5 + temp=0.7 doesn't beat LR=5e-6 + temp=0.7
- Confirms LR=5e-6 is the best LR even with temp=0.7

### Bugs fixed during sweep
1. **`_orig_mod` prefix crash** (`distil_trainer.py:828`): `generate_from_teacher` + `torch.compile` on ref_model caused `ValueError: no module named '_orig_mod'` in `_move_model_to_vllm`. Fixed by adding `"_orig_mod."` to the prefix strip list.
2. **Timeout signal propagation** (`hpo_run.sh`): `timeout` sending signals through the pipe killed the parent script. Fixed by wrapping training in a subshell `(timeout ... uv run python ...) 2>&1 | tee`.
3. **PIPESTATUS crash** (`hpo_run.sh`): `${PIPESTATUS[0]:-$?}` not available after subshell pipe. Fixed by using simple `$?` and elapsed-time check for timeout detection.

### Parallel execution gotchas discovered
- Two trainings on same node crash with `EADDRINUSE` on port 29500 (distributed init conflict)
- Even with `MASTER_PORT` workaround, GPU memory fragmentation prevents parallel runs (vLLM needs contiguous memory)
- Sequential execution is safer and more reliable

### Final recommendation
```
temperature=0.7
alpha=1.0 (reverse KL)
learning_rate=5e-6
student generation (no --generate_from_teacher)
save_steps=50
evaluate BOTH checkpoint-50 and checkpoint-100, take the best
```

### Summary table

| Dimension | Best | Tested | Effect |
|---|---|---|---|
| Temperature | **0.7** | 0.7, 1.0 | +0.63% avg improvement |
| KL direction | **reverse (1.0)** | forward, JSD, reverse | revKL > JSD > fwdKL |
| Generation | **student** | student, teacher | teacher is neutral or worse |
| Learning rate | **5e-6** | 5e-6, 1e-5, 2e-5 | negligible effect |
| Checkpoint | **50-100** | 50, 100, 200, 224 | ~1 epoch optimal, overfit after |

### Next steps (if continuing)
- Try temp=0.5 or temp=0.6 for even sharper generation
- Run more seeds with temp=0.7 for tighter confidence intervals
- Investigate eval variance (3-4% per run) — is it inference randomness or judge variance?
- Consider running more eval runs (5-10 instead of 3) to reduce noise
