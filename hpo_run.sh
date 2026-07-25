#!/bin/bash
# HPO Run Script for SDFT (Self-Distillation Fine-Tuning)
# =======================================================
# Trains Qwen3-8B with a 2-hour timeout, then evaluates ALL checkpoints.
# Sends results back to the opencode tmux pane when done.
#
# Usage: bash hpo_run.sh

set -uo pipefail

# ============================================================
# HYPERPARAMETERS — modify these between runs
# ============================================================
RUN_NAME="sdft_idan_hpo_4"
LEARNING_RATE=5e-6
NUM_EPOCHS=3
NUM_PROMPTS_PER_BATCH=16
PER_DEVICE_BATCH_SIZE=2
REF_MODEL_MIXUP_ALPHA=0.01
SAVE_STEPS=50
SEED=42
DATASET="tooluse"
ENABLE_THINKING=""   # set to "--enable_thinking" to enable

# Additional tunable hyperparameters (defaults match DistilConfig)
ALPHA=0.0              # KL direction: 0.0=forward, 0.5=JSD, 1.0=reverse
TEMPERATURE=1.0        # Generation sampling temperature
WARMUP_RATIO=0.1       # LR warmup fraction
LR_SCHEDULER="cosine"  # cosine, linear, constant
MAX_GRAD_NORM=1.0      # Gradient clipping norm
LOSS_TYPE="dapo"       # grpo, dapo, dr_grpo, bnpo
GENERATE_FROM_TEACHER="--generate_from_teacher"  # online SFT: teacher generates from teacher_prompt

# ============================================================
# FIXED CONFIG — generally don't change these
# ============================================================
TIMEOUT=7200                          # 2 hours in seconds
MODEL_NAME="Qwen/Qwen3-8B"
OUTPUT_BASE="/mnt/nvme7n1/rawhad/amortize_maas_rag"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
TRAIN_LOG="/tmp/sdft_hpo_${RUN_NAME}.log"
RESULTS_LOG="/tmp/sdft_hpo_results.log"
EVAL_DIR="/home/rohan/1_Projects/maas-knowledge-eval"
TRAIN_DIR="/home/rohan/1_Projects/idan_sdft"
TMUX_TARGET="idans_sdft:0.0"
TRAIN_GPUS="0,1,2,3"
INFER_GPU="${TRAIN_GPUS%%,*}"
SCRIPT_STATUS="UNKNOWN"

# Always notify via tmux on exit (success, error, or signal)
notify() {
  local elapsed=$(( $(date +%s) - START_TIME ))
  local msg="HPO ${RUN_NAME} | status=${SCRIPT_STATUS} | time=${elapsed}s"
  if [ -n "${RESULT_MSG:-}" ]; then
    msg="${RESULT_MSG}"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M')] ${msg}" >> "${RESULTS_LOG}"
  tmux send-keys -t "${TMUX_TARGET}" "# ${msg}" Enter
}
START_TIME=$(date +%s)
trap notify EXIT

# ============================================================
# TRAINING
# ============================================================
echo "==========================================="
echo "HPO Run: ${RUN_NAME}"
echo "==========================================="
echo "LR=${LEARNING_RATE}  epochs=${NUM_EPOCHS}  batch=${NUM_PROMPTS_PER_BATCH}x${PER_DEVICE_BATCH_SIZE}"
echo "ref_mixup=${REF_MODEL_MIXUP_ALPHA}  alpha=${ALPHA}  temp=${TEMPERATURE}"
echo "warmup=${WARMUP_RATIO}  scheduler=${LR_SCHEDULER}  grad_norm=${MAX_GRAD_NORM}"
echo "loss=${LOSS_TYPE}  save_steps=${SAVE_STEPS}  seed=${SEED}"
echo "thinking=${ENABLE_THINKING:-off}  teacher_gen=${GENERATE_FROM_TEACHER:-off}"
echo "dataset=${DATASET}"
echo "output: ${OUTPUT_DIR}"
echo "timeout: ${TIMEOUT}s ($(( TIMEOUT / 60 ))m)"
echo "log: ${TRAIN_LOG}"
echo "==========================================="
echo ""

cd "${TRAIN_DIR}"

# Run training with timeout
SCRIPT_STATUS="TRAINING"
timeout ${TIMEOUT} \
  env CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} \
      WANDB_PROJECT="amortize-maas" \
      WANDB_ENTITY="ronny21" \
      WANDB_NAME="${RUN_NAME}" \
  uv run python main.py \
    --learning_rate ${LEARNING_RATE} \
    --dataset_name ${DATASET} \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs ${NUM_EPOCHS} \
    --model_name ${MODEL_NAME} \
    --num_prompts_per_batch ${NUM_PROMPTS_PER_BATCH} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --ref_model_mixup_alpha ${REF_MODEL_MIXUP_ALPHA} \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --report_to wandb \
    --seed ${SEED} \
    --alpha ${ALPHA} \
    --temperature ${TEMPERATURE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type ${LR_SCHEDULER} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --loss_type ${LOSS_TYPE} \
    ${ENABLE_THINKING} \
    ${GENERATE_FROM_TEACHER} \
  2>&1 | tee "${TRAIN_LOG}" || true

TRAIN_EXIT=${PIPESTATUS[0]:-$?}
END_TIME=$(date +%s)
TRAIN_ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "Training finished in ${TRAIN_ELAPSED}s (exit code: ${TRAIN_EXIT})"
if [ "${TRAIN_EXIT}" -eq 124 ]; then
  echo "  (timed out after ${TIMEOUT}s)"
fi

# Count training steps from log
TRAIN_STEPS=$(grep -c "'loss':" "${TRAIN_LOG}" 2>/dev/null || echo "0")
echo "Training steps completed: ${TRAIN_STEPS}"

# ============================================================
# FIND ALL CHECKPOINTS
# ============================================================
echo ""
echo "=== Finding checkpoints ==="

ALL_CKPTS=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)

if [ -z "${ALL_CKPTS}" ]; then
  echo "ERROR: No checkpoint found in ${OUTPUT_DIR}"
  SCRIPT_STATUS="FAILED_NO_CKPT (${TRAIN_STEPS} steps)"
  exit 1
fi

CKPT_COUNT=$(echo "${ALL_CKPTS}" | wc -l)
echo "Found ${CKPT_COUNT} checkpoint(s):"
echo "${ALL_CKPTS}" | while read -r p; do echo "  $(basename "$p")"; done

# ============================================================
# EVALUATE ALL CHECKPOINTS
# ============================================================
cd "${EVAL_DIR}"

BEST_AVG="0"
BEST_CKPT=""
CKPT_SUMMARIES=""

for CKPT_PATH in ${ALL_CKPTS}; do
  CKPT_BASENAME=$(basename "${CKPT_PATH}")
  CKPT_NAME="${RUN_NAME}/${CKPT_BASENAME}"

  echo ""
  echo "--- Evaluating: ${CKPT_NAME} ---"

  # Inference
  SCRIPT_STATUS="INFERENCE (${CKPT_BASENAME})"
  if ! just infer "${INFER_GPU}" "${CKPT_NAME}"; then
    echo "WARNING: Inference failed for ${CKPT_NAME}, skipping"
    CKPT_SUMMARIES="${CKPT_SUMMARIES} | ${CKPT_BASENAME}=INFER_FAIL"
    continue
  fi

  # Evaluation
  SCRIPT_STATUS="EVALUATION (${CKPT_BASENAME})"
  if ! just eval-claude "${CKPT_NAME}"; then
    echo "WARNING: Eval failed for ${CKPT_NAME}, skipping"
    CKPT_SUMMARIES="${CKPT_SUMMARIES} | ${CKPT_BASENAME}=EVAL_FAIL"
    continue
  fi

  # Parse results for this checkpoint
  EVAL_RESULTS_DIR="${EVAL_DIR}/eval_results_anthropic_sonnet/v3_sdft/${CKPT_NAME}"
  ACCURACIES=""
  for i in 1 2 3; do
    SUMMARY_FILE=$(ls "${EVAL_RESULTS_DIR}"/rag_v2_eval_summary_*"-${i}.json" 2>/dev/null | head -1)
    if [ -f "${SUMMARY_FILE}" ]; then
      ACC=$(python3 -c "import json; d=json.load(open('${SUMMARY_FILE}')); print(d['accuracy'])")
      ACCURACIES="${ACCURACIES} ${ACC}"
      echo "  Run ${i}: accuracy=${ACC}"
    else
      echo "  Run ${i}: summary file not found"
    fi
  done

  CKPT_AVG=$(python3 -c "
accs = [float(x) for x in '''${ACCURACIES}'''.split()]
if accs:
    import statistics
    avg = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0
    print(f'{avg*100:.2f}%+/-{std*100:.2f}%')
else:
    print('N/A')
")
  echo "  => ${CKPT_BASENAME}: ${CKPT_AVG}"
  CKPT_SUMMARIES="${CKPT_SUMMARIES} | ${CKPT_BASENAME}=${CKPT_AVG}"

  # Track best
  CURR_AVG=$(python3 -c "
accs = [float(x) for x in '''${ACCURACIES}'''.split()]
print(f'{sum(accs)/len(accs):.6f}' if accs else '0')
")
  if python3 -c "exit(0 if float('${CURR_AVG}') > float('${BEST_AVG}') else 1)"; then
    BEST_AVG="${CURR_AVG}"
    BEST_CKPT="${CKPT_BASENAME}"
  fi
done

# Format best accuracy
BEST_PCT=$(python3 -c "print(f'{float(\"${BEST_AVG}\")*100:.2f}%')" 2>/dev/null || echo "N/A")

# ============================================================
# REPORT (trap handles tmux notify + log append)
# ============================================================
SCRIPT_STATUS="DONE"
RESULT_MSG="HPO ${RUN_NAME} | steps=${TRAIN_STEPS} | LR=${LEARNING_RATE} ep=${NUM_EPOCHS} batch=${NUM_PROMPTS_PER_BATCH}x${PER_DEVICE_BATCH_SIZE} mixup=${REF_MODEL_MIXUP_ALPHA} alpha=${ALPHA} temp=${TEMPERATURE} warmup=${WARMUP_RATIO} sched=${LR_SCHEDULER} loss=${LOSS_TYPE} | BEST=${BEST_CKPT} ${BEST_PCT}${CKPT_SUMMARIES} | baseline=84.7% target=85.8%"

echo ""
echo "==========================================="
echo "RESULT: ${RESULT_MSG}"
echo "==========================================="
