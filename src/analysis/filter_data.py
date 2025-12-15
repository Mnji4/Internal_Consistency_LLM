import json
import os
import re
from src.config import config
from src.data.gsm8k_loader import GSM8KLoader

def extract_answer_model(text):
    """
    Heuristic extraction for model output.
    Adjust based on model behavior (e.g. looks for last number).
    """
    # 简单的策略：找也就是最后一个数字
    # 更鲁棒的策略可能需要解析 "The answer is X"
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if not numbers:
        return None
    return numbers[-1]

def normalize_answer(ans):
    if ans is None:
        return ""
    # Remove commas, evaluate simple fractions if needed? 
    # For now just strip string
    return str(ans).strip().replace(',', '')

def main():
    input_file = os.path.join(config.OUTPUT_DIR, "inference_results.jsonl")
    output_file = os.path.join(config.DATA_DIR, "sft_train.jsonl")
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    filtered_samples = []
    stats = {"total": len(data), "valid_simple": 0, "conflict": 0, "selected": 0}
    
    for item in data:
        # 1. Extract answers
        # GT is already clean from loader
        gt_raw = GSM8KLoader.extract_answer(item['ground_truth'])
        if gt_raw is None:
            # Maybe GT formatting is different or already extracted?
            # Check the raw string
            # If our loader extracted it before, it might be clean.
            # But in `generate_dataset.py` we saved `sample['answer']` which is the full string.
            pass

        ans_gt = normalize_answer(gt_raw)
        
        ans_simple_val = extract_answer_model(item['response_simple'])
        ans_complex_val = extract_answer_model(item['response_complex'])
        
        ans_simple = normalize_answer(ans_simple_val)
        ans_complex = normalize_answer(ans_complex_val)
        
        # 2. Check Validity of Simple Answer
        # Strategy B: We only trust Simple Answer if it matches Ground Truth
        if ans_simple == ans_gt and ans_simple != "":
            stats["valid_simple"] += 1
            
            # 3. Check Consistency
            if ans_simple != ans_complex:
                stats["conflict"] += 1
                
                # 4. Create Training Sample
                # Input: Complex Prompt
                # Output: Simple Response (The full text that led to the correct answer)
                
                # We want the model to output the chain of thought of the SIMPLE (correct) response,
                # but given the COMPLEX prompt.
                
                # Format for SFT (e.g. Chat format)
                training_entry = {
                    "messages": [
                        {"role": "user", "content": item['complex_prompt']},
                        {"role": "assistant", "content": item['response_simple']}
                    ]
                }
                filtered_samples.append(training_entry)
                stats["selected"] += 1

    # Save
    with open(output_file, 'w') as f:
        for entry in filtered_samples:
            f.write(json.dumps(entry) + "\n")
            
    print("Filtering complete.")
    print(json.dumps(stats, indent=2))
    print(f"Saved {len(filtered_samples)} samples to {output_file}")

if __name__ == "__main__":
    main()
