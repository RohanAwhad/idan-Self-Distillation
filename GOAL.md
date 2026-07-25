# GOAL: Hyperparameter Optimization for SDFT (Qwen3-8B)

## Objective

Improve Qwen3-8B accuracy on the MaaS knowledge eval benchmark via Self-Distillation Fine-Tuning (SDFT) hyperparameter optimization.

- **Baseline**: 84.67% (qwen3-8b-base, no fine-tuning)
- **Current best**: 85.80% (sdft_sdg_hub_v3/step_113)
- **Goal**: Beat 85.80%

## Critical Constraint: Overfitting

The single most important finding from prior runs: **longer training degrades accuracy significantly**. Both `sdft_asynth_v3` (84.00% -> 77.07%) and `sdft_sdg_hub_v3` (85.80% -> 81.77%) show massive drops at later steps. Early checkpoints are consistently better. The sweet spot appears to be around 100-150 steps.

---

## Prior Results

| Checkpoint | Avg Accuracy | LR | Notes |
|---|---|---|---|
| qwen3-8b-base (baseline) | **84.67%** | - | No fine-tuning |
| sdft_sdg_hub_v3/step_113 | **85.80%** | unknown | Current best, early step |
| sdft_idan_test_run/checkpoint-224 | 84.34% | 5e-6 | 2 epochs, full training |
| sdft_idan_run_1/checkpoint-224 | 84.34% | 2e-5 | 2 epochs, full training |
| sdft_asynth_v3/step_104 | 84.00% | unknown | Early step OK |
| sdft_sdg_hub_v3_run_6/step_90 | 83.89% | unknown | |
| sdft_sdg_hub_v3_run_6/step_80 | 81.99% | unknown | |
| sdft_sdg_hub_v3/step_1120 | 81.77% | unknown | Overfit (late step) |
| sdft_asynth_v3/step_1030 | 77.07% | unknown | Severely overfit |

### Loss Curves from Prior Runs

**sdft_idan_run_1 (LR=2e-5, 2 epochs, 224 steps)**:
- Loss: 0.327 -> 0.204 (min ~step 150) -> 0.224 (end)
- KL: 0.299 -> 0.187 (end)
- Eval: 84.34% at checkpoint-224

**sdft_idan_test_run (LR=5e-6, 2 epochs, 224 steps)**:
- Loss: 0.243 -> 0.171 (min ~step 150) -> 0.201 (end)
- KL: 0.242 -> 0.173 (end)
- Eval: 84.34% at checkpoint-224
- Training time: ~4.2 hours (15,220s), ~68s/step

Lower LR (5e-6) achieves lower loss but same eval accuracy as higher LR (2e-5).

---

## Training Setup

### Model & Data
- **Model**: `Qwen/Qwen3-8B` (8.19B params)
- **Dataset**: `tooluse` (3,592 samples, loaded from `data/tooluse_data/train_data/`)
- **Steps per epoch**: ~112 (3592 / 32 effective batch)
- **Training method**: Self-Distillation Fine-Tuning (student and teacher are the same architecture, teacher generates completions, student learns from teacher's output distribution via KL divergence)

### GPU Allocation
- **Training**: `CUDA_VISIBLE_DEVICES=4,5,6,7` (8x H100 80GB available)
  - GPU 4 (CUDA 0): vLLM colocated generation
  - GPU 5 (CUDA 1): Teacher model
  - GPU 6,7 (CUDA 2,3): Student model (device_map="auto")
- **Inference** (after training): GPU 4 (first training GPU, freed after training completes)

### Hardware
- 8x NVIDIA H100 80GB HBM3
- All on same node

---

## Evaluation Pipeline

The eval benchmark tests the model on 298 MaaS knowledge questions, scored by Claude Sonnet as judge. Each checkpoint is evaluated 3 times (for variance estimation).

### Commands

Run from `/home/rohan/1_Projects/maas-knowledge-eval`:

```bash
# Step 1: Inference (requires GPU, loads model)
just infer 0 sdft_idan_hpo_1/checkpoint-50

# Step 2: Evaluation (calls Anthropic API, no GPU needed)
just eval-claude sdft_idan_hpo_1/checkpoint-50
```

### Model Path Convention
The eval expects checkpoints at:
```
/mnt/nvme7n1/rawhad/amortize_maas_rag/<RUN_NAME>/<CHECKPOINT>/
```

### Output Format
- Results land in: `eval_results_anthropic_sonnet/v3_sdft/<RUN_NAME>/<CHECKPOINT>/`
- Summary JSON per run: `rag_v2_eval_summary_*-{1,2,3}.json`
- Key field: `accuracy` (float, e.g. 0.8580 = 85.80%)
- Also has per-type breakdown (cross_reference, edge_case, multi_step, specificity, troubleshooting) and per-hop breakdown (1-4 hops)
- Failure modes: hallucination, off_topic, correct

---

## The Script: `hpo_run.sh`

Location: `/home/rohan/1_Projects/idan_sdft/hpo_run.sh`

### Flow
```
1. Set hyperparameters (variables at top of script)
2. Train with `timeout 7200` (2 hours hard cap)
3. Find ALL checkpoint-* directories in output_dir (sorted by step)
4. For EACH checkpoint:
   a. Run inference on GPU 4 (just infer)
   b. Run eval-claude (Claude Sonnet judge, 3 runs)
   c. Parse accuracy, track best
5. Report: best checkpoint + per-checkpoint breakdown
6. Send result summary to tmux idans_sdft:0.0
7. Append to /tmp/sdft_hpo_results.log
```

### How to Run
```bash
# From a tmux window (e.g., idans_sdft:4 "hpo" window):
cd /home/rohan/1_Projects/idan_sdft
bash hpo_run.sh
```

### How to Modify for Next Run
Edit the hyperparameter block at the top of `hpo_run.sh`. Bump `RUN_NAME` to `sdft_idan_hpo_2`, etc.

---

## Hard Constraints (DO NOT CHANGE)

1. **`enable_thinking` = OFF** (do not pass `--enable_thinking`)
2. **`save_steps` = 50** (checkpoint every 50 steps, do not change)
3. **Training timeout = 7200s** (2 hours)
4. **Each experiment must be git committed before running** (commit the hpo_run.sh changes so the experiment is tracked)
5. **Run naming**: `sdft_idan_hpo_N` (increment N for each experiment)
6. **Dataset**: `tooluse`

---

## Tunable Hyperparameters

These are the knobs you can turn between runs. Modify them in the `HYPERPARAMETERS` block at the top of `hpo_run.sh`.

### Exposed in hpo_run.sh (all tunable)

| Parameter | Default | Range/Options | Notes |
|---|---|---|---|
| `LEARNING_RATE` | 5e-6 | 1e-6 to 2e-5 | Most impactful knob |
| `NUM_EPOCHS` | 3 | 1 to 5 | But 2h timeout caps actual training |
| `NUM_PROMPTS_PER_BATCH` | 16 | 8, 16, 32 | gradient_accumulation_steps |
| `PER_DEVICE_BATCH_SIZE` | 2 | 1, 2, 4 | Effective batch = this * NUM_PROMPTS_PER_BATCH |
| `REF_MODEL_MIXUP_ALPHA` | 0.01 | 0.001 to 0.1 | How fast ref model tracks student (TR-DPO) |
| `SEED` | 42 | any int | For reproducibility |
| `ALPHA` | 1.0 | 0.0=forward KL, 0.5=JSD, 1.0=reverse KL | KL direction |
| `TEMPERATURE` | 1.0 | 0.7 to 1.5 | Generation sampling temperature |
| `WARMUP_RATIO` | 0.1 | 0.0 to 0.2 | LR warmup fraction |
| `LR_SCHEDULER` | cosine | cosine, linear, constant | LR schedule shape |
| `MAX_GRAD_NORM` | 1.0 | 0.5 to 2.0 | Gradient clipping |
| `LOSS_TYPE` | dapo | grpo, dapo, dr_grpo, bnpo | Loss normalization strategy |
| `GENERATE_FROM_TEACHER` | off | "--generate_from_teacher" | Online SFT: teacher generates, student learns |

### Available in DistilConfig but not yet exposed

To use these, add CLI arg in `main.py:parse_args()`, wire into `DistilConfig(...)`, then add variable to `hpo_run.sh`:

| Parameter | Default | Range/Options | Notes |
|---|---|---|---|
| `epsilon` | 0.2 | 0.1 to 0.3 | Clipping parameter |
| `num_loss_tokens_to_skip` | 3 | 0 to 10 | Skip initial completion tokens in loss |
| `scale_rewards` | group | group, batch, none | Reward scaling strategy |
| `top_entropy_quantile` | 1.0 | 0.2 to 1.0 | Entropy-based token masking (1.0=off) |
| `ref_model_sync_steps` | 1 | 1 to 64 | How often to sync ref model |
| `max_completion_length` | 2048 | 1024 to 4096 | Max generated tokens |
| `max_prompt_length` | 2048 | 1024 to 4096 | Max prompt tokens |

---

## HPO Workflow (Step by Step)

1. **Modify hyperparameters** in `hpo_run.sh` (top section)
2. **Bump RUN_NAME** to next number (e.g., `sdft_idan_hpo_1` -> `sdft_idan_hpo_2`)
3. **Commit** the change: `git add hpo_run.sh && git commit -m "hpo: sdft_idan_hpo_N with <description of changes>"`
4. **Run** in tmux hpo window: `bash hpo_run.sh`
5. **Do NOT poll or wait** — the script has an EXIT trap that sends results via `tmux send-keys` to `idans_sdft:0.0` when it finishes (success or failure). The result will arrive in your pane automatically.
6. **Result appears** as a `# HPO ...` comment in your tmux pane, and is appended to `/tmp/sdft_hpo_results.log`
7. **Analyze** the result, check wandb for loss curves
8. **Decide next hyperparameters** based on results
9. **Repeat** from step 1

### WandB Dashboard
- Project: `amortize-maas`
- Entity: `ronny21`
- Each run is named by `RUN_NAME` (e.g., `sdft_idan_hpo_1`)

---

## Key Intuitions for HPO

1. **Early stopping matters most**: The best checkpoint so far (85.80%) was at step 113 (~1 epoch). Overfitting is the primary failure mode. With save_steps=50, you get checkpoints at 50, 100, 150... — evaluate multiple if needed.

2. **Learning rate**: Prior runs show LR=5e-6 gets lower loss than LR=2e-5, but both plateau at ~84.3% eval accuracy. The sdg_hub runs that hit 85.8% used unknown LR — exploring the 1e-6 to 1e-5 range is likely productive.

3. **KL direction (alpha)**: All prior runs used alpha=1.0 (reverse KL). Forward KL (alpha=0.0) or JSD (alpha=0.5) might behave differently — reverse KL is mode-seeking, forward KL is mean-seeking.

4. **Reference model tracking**: Currently ref_model_mixup_alpha=0.01 with sync every 1 step. This means the reference model tracks the student very closely. A higher alpha (e.g., 0.05-0.1) might provide stronger regularization against overfitting.

5. **Effective batch size**: Currently 16 * 2 = 32. Larger batch (e.g., 32 * 2 = 64) could stabilize training but reduces number of steps per epoch.

---

## Repository Structure

```
/home/rohan/1_Projects/idan_sdft/
  main.py              # Training entry point
  distil_config.py     # DistilConfig (extends TrainingArguments)
  distil_trainer.py    # DistilTrainer (the training loop)
  hpo_run.sh           # HPO experiment script (THIS IS WHAT YOU MODIFY)
  GOAL.md              # This file
  bench.sh             # 5-min benchmark script (reference only)
  data/
    tooluse_data/      # Training data (3,592 samples)
    science_data/      # Alternative dataset (not used for HPO)
    maas_data/         # Not wired into main.py

/home/rohan/1_Projects/maas-knowledge-eval/
  justfile             # Eval commands (just infer, just eval-claude)
  scripts/             # Inference and eval scripts
  data/                # Eval benchmark data (298 questions)
  eval_results_anthropic_sonnet/  # Eval results land here

/mnt/nvme7n1/rawhad/amortize_maas_rag/
  <RUN_NAME>/          # Checkpoints saved here
    checkpoint-50/
    checkpoint-100/
    ...
```
