import json
import os
from src.data.gsm8k_loader import GSM8KLoader
from src.data.perturbation import PerturbationEngine
from src.config import config
from tqdm import tqdm

def main():
    # Ensure output dir exists
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    loader = GSM8KLoader(split="train")
    engine = PerturbationEngine()
    
    # Generate all pairs
    print("Generating prompt pairs...")
    # For dev/testing, maybe limit size? Or generate full. 
    # Let's generate full but maybe print info.
    all_data = []
    for i in tqdm(range(len(loader))):
        sample = loader[i]
        q = sample['question']
        
        # Determine number of variations? For now 1 simple, 1 complex per question.
        # We could generate MULTIPLE complex prompts per question to increase data.
        
        pair = {
            'id': i,
            'original_question': q,
            'ground_truth': sample['answer'],
            'simple_prompt': engine.create_simple_prompt(q),
            'complex_prompt': engine.create_complex_prompt(q)
        }
        all_data.append(pair)
        
    output_path = os.path.join(config.DATA_DIR, "train_prompts.jsonl")
    with open(output_path, "w") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved {len(all_data)} pairs to {output_path}")

if __name__ == "__main__":
    main()
