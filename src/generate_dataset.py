import json
import os
from src.data.logic_loader import LogicLoader
from src.data.perturbation import LogicPerturbationEngine
from src.config import config
from tqdm import tqdm

def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    loader = LogicLoader(split="train")
    engine = LogicPerturbationEngine()
    
    print("Generating Logic prompt pairs...")
    all_data = []
    
    for i in tqdm(range(len(loader))):
        sample = loader[i]
        
        # Folio structure: 'premises' (list), 'conclusion' (str), 'label' (str)
        # We treat 'conclusion' as the Question?
        # Folio: "Determine whether the conclusion follows."
        
        premises = LogicLoader.format_context(sample['premises'])
        conclusion = sample['conclusion']
        label = sample['label'] # True/False/Uncertain
        
        question = f"Based on the context, is the statement '{conclusion}' True, False, or Uncertain?"
        
        pair = {
            'id': i,
            'premises': premises,
            'conclusion': conclusion,
            'ground_truth': label,
            'simple_prompt': engine.create_simple_prompt(premises, question),
            'complex_prompt': engine.create_complex_prompt(premises, question)
        }
        all_data.append(pair)
        
    output_path = os.path.join(config.DATA_DIR, "train_prompts.jsonl")
    with open(output_path, "w") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Saved {len(all_data)} pairs to {output_path}")

if __name__ == "__main__":
    main()
