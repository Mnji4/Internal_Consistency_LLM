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

# 1. Environment Check
echo "[1/4] Checking Environment..."
pip install -r $PROJECT_ROOT/requirements.txt | tee $LOG_DIR/install.log

# 2. Data Generation
echo "[2/4] Generating Dataset for Machine B..."
python3 $PROJECT_ROOT/src/generate_dataset.py | tee $LOG_DIR/data_gen.log
echo "Dataset generated at $DATA_DIR/train_prompts.jsonl"

# 3. Inference
echo "[3/6] Running Inference (Validation/Filtering Phase)..."
# Using a slightly smaller batch size to be safe with OOM
# Note: config.BATCH_SIZE is used inside, usually 8.
python3 $PROJECT_ROOT/src/inference/run_inference.py | tee $LOG_DIR/inference.log
echo "Inference results at $OUTPUT_DIR/inference_results.jsonl"

# 4. Filtering
echo "[4/6] Filtering & Creating SFT Dataset..."
python3 $PROJECT_ROOT/src/analysis/filter_data.py | tee $LOG_DIR/filter.log
echo "SFT Dataset at $DATA_DIR/sft_train.jsonl"

# 5. Training
echo "[5/6] Starting SFT Training..."
python3 $PROJECT_ROOT/src/train/run_sft.py \
    --num_epochs 3 \
    --batch_size 4 \
    --output_dir "$PROJECT_ROOT/checkpoints/ic_sft_v1" | tee $LOG_DIR/train.log

# 6. Evaluation
echo "[6/6] Running Evaluation..."
python3 $PROJECT_ROOT/src/evaluate/evaluate_checkpoint.py \
    --checkpoint_dir "$PROJECT_ROOT/checkpoints/ic_sft_v1" \
    --test_file "$PROJECT_ROOT/data/train_prompts.jsonl" | tee $LOG_DIR/eval.log

echo "=================================================="
echo "Pipeline Complete!"
echo "Results are ready to be pushed via git."
echo "Suggested commands:"
echo "  git add data/ output/ logs/"
echo "  git commit -m 'Run results from $(hostname) at $(date)'"
echo "  git push"
echo "=================================================="
