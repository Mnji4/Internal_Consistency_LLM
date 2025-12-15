#!/bin/bash
set -e  # Exit on error

# Configuration
PROJECT_ROOT=$(dirname $(dirname $(realpath $0)))
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/output"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p $LOG_DIR
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Force GPU 0
export CUDA_VISIBLE_DEVICES=0

echo "=================================================="
echo "Starting Internal Consistency Pipeline on Machine B"
echo "Project Root: $PROJECT_ROOT"
echo "Date: $(date)"
echo "=================================================="

# 1. Environment Check (Restored to fix numpy)
echo "[1/4] Checking Environment..."
pip install -r $PROJECT_ROOT/requirements.txt | tee $LOG_DIR/install.log

# 2. Data Generation (Skipped)
# echo "[2/4] Generating Dataset..."
# python3 $PROJECT_ROOT/src/generate_dataset.py | tee $LOG_DIR/data_gen.log
# echo "Dataset generated at $DATA_DIR/train_prompts.jsonl"

# 3. Inference (Skipped)
# echo "[3/4] Running Inference (Validation/Filtering Phase)..."
# python3 $PROJECT_ROOT/src/inference/run_inference.py | tee $LOG_DIR/inference.log
# echo "Inference results at $OUTPUT_DIR/inference_results.jsonl"

# 4. Filtering (Skipped)
# echo "[4/4] Filtering & Creating SFT Dataset..."
# python3 $PROJECT_ROOT/src/analysis/filter_data.py | tee $LOG_DIR/filter.log
# echo "SFT Dataset at $DATA_DIR/sft_train.jsonl"

# 5. Training (Optional/Separate step usually, but included for 'All-in-one')
echo "[5/5] Starting SFT Training..."
# Note: Check arguments in source code for default paths
python3 $PROJECT_ROOT/src/train/run_sft.py \
    --num_epochs 1 \
    --batch_size 4 \
    --output_dir "$PROJECT_ROOT/checkpoints/ic_sft_v1" | tee $LOG_DIR/train.log

echo "=================================================="
echo "Pipeline Complete!"
echo "Results are ready to be pushed via git."
echo "Suggested commands:"
echo "  git add data/ output/ logs/"
echo "  git commit -m 'Run results from $(hostname) at $(date)'"
echo "  git push"
echo "=================================================="
