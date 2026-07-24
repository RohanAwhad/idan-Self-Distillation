#!/bin/bash
# 5-min benchmark: measure training steps in 300 seconds
# Usage: bash bench.sh

BENCH_LOG=/tmp/sdft_bench.log
GPU_LOG=/tmp/gpu_dmon.log
TIMEOUT=300

echo "Starting 5-min benchmark..."
echo "Train log: $BENCH_LOG"
echo "GPU log:   $GPU_LOG"

# Start GPU monitoring in background (2s interval, utilization + memory)
nvidia-smi dmon -d 2 -s um -i 4,5,6,7 > "$GPU_LOG" 2>&1 &
DMON_PID=$!

START_TIME=$(date +%s)

timeout $TIMEOUT \
  env CUDA_VISIBLE_DEVICES=4,5,6,7 \
      WANDB_MODE=disabled \
  uv run python main.py \
    --learning_rate 1e-5 \
    --dataset_name tooluse \
    --output_dir /tmp/sdft_bench \
    --num_train_epochs 2 \
    --model_name Qwen/Qwen3-8B \
    --num_prompts_per_batch 16 \
    --per_device_train_batch_size 2 \
    --report_to none \
    --save_strategy no \
  2>&1 | tee "$BENCH_LOG"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Stop GPU monitoring
kill $DMON_PID 2>/dev/null
wait $DMON_PID 2>/dev/null

echo ""
echo "=== BENCHMARK RESULT ==="
echo "Wall time: ${ELAPSED}s"
STEPS=$(grep -c "'loss':" "$BENCH_LOG")
echo "Steps completed: $STEPS"
if [ "$STEPS" -gt 0 ]; then
  echo "Avg: $(awk "BEGIN{printf \"%.1f\", $ELAPSED/$STEPS}")s/step"
fi
echo ""
# Show last tqdm progress line
grep -oP '\d+/\d+.*s/it' "$BENCH_LOG" | tail -1
echo ""
echo "GPU utilization log: $GPU_LOG"
