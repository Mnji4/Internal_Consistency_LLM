import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import config
from src.inference.run_inference import generate_batch, load_model
from src.analysis.filter_data import normalize_logic_answer
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(config.PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--base_model", type=str, default=config.MODEL_NAME)
    parser.add_argument("--test_file", type=str, default=os.path.join(config.DATA_DIR, "train_prompts.jsonl")) 
    # Ideally should use a held-out test set, but for this demo we can verify on the training set (or a subset) 
    # to see if it learned to correct the inconsistencies we found.
    # To be rigorous, we should have split train/test earlier. 
    # Let's inspect the 'inference_results.jsonl' to find indices of the 74 conflict samples and see if they are fixed.
    
    parser.add_argument("--output_file", type=str, default=os.path.join(config.OUTPUT_DIR, "eval_results.jsonl"))
    args = parser.parse_args()

    # 1. Load Data (Evaluation Set)
    # Let's verify specifically on the "Conflict" samples that we trained on (Training Accuracy)
    # And maybe some valid ones to ensure no regression.
    # For simplicity, let's load the inference_results, find the ones marked 'conflict' in our previous analysis.
    
    # Actually, let's just re-run inference on the original dataset (or a subset) using the LoRA model.
    print(f"Loading test data from {args.test_file}")
    with open(args.test_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Limit for speed? 
    # data = data[:100] 
    
    # 2. Load Base Model + LoRA Adapter
    print(f"Loading Base Model: {args.base_model}")
    model, tokenizer = load_model(args.base_model)
    
    print(f"Loading LoRA Adapter from: {args.checkpoint_dir}")
    try:
        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
        print("LoRA adapter loaded successfully.")
    except Exception as e:
        print(f"Error loading adapter (maybe raw model?): {e}")
        # proceed as is? No, point of eval is evaluating the fine-tuned model.
        return

    # 3. Inference on Complex Prompts
    # We want to see if Complex Prompt -> Simple Answer (Correct)
    complex_prompts = [item['complex_prompt'] for item in data]
    
    print("Running inference with Fine-tuned model on Complex prompts...")
    responses = generate_batch(model, tokenizer, complex_prompts, batch_size=config.BATCH_SIZE)
    
    # 4. Analysis
    results = []
    improved = 0
    total_conflict_in_baseline = 0
    regressed = 0
    
    # We need the baseline results to compare. 
    # We can assume the 'inference_results.jsonl' holds the baseline behavior.
    # Let's try to match them by ID if available, or just re-eval Base model? 
    # Re-evaluating base might be expensive. 
    # Let's just output the new results and do offline comparison or load previous results here.
    
    prev_results_path = os.path.join(config.OUTPUT_DIR, "inference_results.jsonl")
    prev_data_map = {}
    if os.path.exists(prev_results_path):
        with open(prev_results_path, 'r') as f:
            for line in f:
                j = json.loads(line)
                prev_data_map[j['id']] = j
    
    for i, item in enumerate(data):
        new_resp = responses[i]
        new_ans = normalize_logic_answer(new_resp)
        gt = item['ground_truth']
        
        # Check if it was a conflict before
        is_conflict_before = False
        prev_simple_ans = "N/A"
        prev_complex_ans = "N/A"
        
        if item['id'] in prev_data_map:
            prev = prev_data_map[item['id']]
            prev_simple_ans = normalize_logic_answer(prev['response_simple'])
            prev_complex_ans = normalize_logic_answer(prev['response_complex'])
            
            # Conflict definition from filter_data.py
            if prev_simple_ans == gt and prev_simple_ans != prev_complex_ans:
                is_conflict_before = True
                total_conflict_in_baseline += 1
        
        # Did we fix it?
        # Fixed if New Complex Answer == GT
        is_fixed = False
        if is_conflict_before:
            if new_ans == gt:
                improved += 1
                is_fixed = True
        
        results.append({
            "id": item['id'],
            "complex_prompt": item['complex_prompt'],
            "ground_truth": gt,
            "prediction": new_ans,
            "raw_response": new_resp,
            "is_fixed": is_fixed,
            "was_conflict": is_conflict_before
        })

    print("-" * 30)
    print(f"Total Samples: {len(data)}")
    print(f"Baseline Conflicts (Target for repair): {total_conflict_in_baseline}")
    print(f"Fixed Conflicts: {improved}")
    if total_conflict_in_baseline > 0:
        print(f"Recovery Rate: {improved / total_conflict_in_baseline:.2%}")
    print("-" * 30)
    
    with open(args.output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved evaluation results to {args.output_file}")

if __name__ == "__main__":
    main()
