import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import config
from tqdm import tqdm
import argparse

def load_model(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer

def generate_batch(model, tokenizer, prompts, batch_size=8):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False, # Greedy decoding for consistency baseline
                pad_token_id=tokenizer.pad_token_id
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract the NEWly generated text (strip prompt)
        for prompt, full_text in zip(batch_prompts, decoded):
            # Simple stripping, might need to be more robust depending on model output format
            if full_text.startswith(prompt):
                generated = full_text[len(prompt):]
            else:
                generated = full_text # Fail safe
            results.append(generated)
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=os.path.join(config.DATA_DIR, "train_prompts.jsonl"))
    parser.add_argument("--output_file", type=str, default=os.path.join(config.OUTPUT_DIR, "inference_results.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Debug: limit number of samples")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 1. Load Data
    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if args.limit:
        data = data[:args.limit]
        
    print(f"Loaded {len(data)} samples.")

    # 2. Prepare Prompts
    # Flatten dataset to batch process: [Simple_1, Complex_1, Simple_2, Complex_2, ...]
    # Or process separately? Flattening is likely more efficient if batching.
    all_prompts = []
    for item in data:
        all_prompts.append(item['simple_prompt'])
        all_prompts.append(item['complex_prompt'])

    # 3. Load Model
    model, tokenizer = load_model(config.MODEL_NAME)

    # 4. Inference
    print("Starting inference...")
    responses = generate_batch(model, tokenizer, all_prompts, batch_size=config.BATCH_SIZE)

    # 5. Re-assemble and Save
    with open(args.output_file, 'w') as f:
        for i, item in enumerate(tqdm(data)):
            # simple is at 2*i, complex is at 2*i+1
            resp_simple = responses[2*i]
            resp_complex = responses[2*i+1]
            
            result_item = item.copy()
            result_item.update({
                'response_simple': resp_simple,
                'response_complex': resp_complex
            })
            f.write(json.dumps(result_item) + "\n")
            
    print(f"Done. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
