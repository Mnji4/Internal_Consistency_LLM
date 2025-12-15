#!/bin/bash
set -e

PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
LOG_DIR="$PROJECT_ROOT/logs"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

echo "=================================================="
echo "Starting Evaluation on Machine B"
echo "=================================================="

python3 $PROJECT_ROOT/src/evaluate/evaluate_checkpoint.py \
    --checkpoint_dir "$PROJECT_ROOT/checkpoints" \
    --test_file "$PROJECT_ROOT/data/train_prompts.jsonl" | tee $LOG_DIR/eval.log

echo "Evaluation Complete."
echo "Please git add logs/ output/ and push."
