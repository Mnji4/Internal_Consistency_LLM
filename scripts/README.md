# Internal Consistency Experiment Scripts

## Remote Execution Workflow (Machine A <-> Machine B)

### 1. Setup (Machine A)
Ensure you are on the `main` branch and have pushed the latest code.
```bash
git add .
git commit -m "Update code"
git push
```

### 2. Execution (Machine B)
Log in to Machine B, pull the code, and run the pipeline.

```bash
# On Machine B
cd /path/to/Internal_Consistency_LLM
git pull origin main

# Run the full pipeline (Data -> Inference -> Filter -> Train)
bash scripts/run_cycle_b.sh
```

**What this script does:**
1. Installs dependencies (`requirements.txt`).
2. Generates GSM8K prompt pairs (`src/generate_dataset.py`).
3. Runs inference on the model (`src/inference/run_inference.py`).
4. Filters data for consistency (`src/analysis/filter_data.py`).
5. Runs SFT training (`src/train/run_sft.py`).

### 3. Sync Results (Machine B -> Machine A)
After the script finishes, push the results back.

```bash
# On Machine B
git add data/ output/ logs/
git commit -m "Results from Machine B"
git push
```

### 4. Analysis (Machine A)
Pull the results locally.
```bash
# On Machine A
git pull
# Check logs or new data
cat logs/filter.log
```
