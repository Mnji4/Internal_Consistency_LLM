import random
from src.config import config

class PerturbationEngine:
    def __init__(self):
        self.templates = config.PERTURBATION_TEMPLATES

    def create_simple_prompt(self, question):
        """
        Constructs the simple prompt. 
        For now, it's just the clean question with a standard instruction.
        """
        # 可以加上明确的指令，确保模型处于答题模式
        return f"Question: {question}\nAnswer:"

    def create_complex_prompt(self, question):
        """
        Constructs a complex/perturbed prompt.
        """
        template = random.choice(self.templates)
        
        # 简单的策略：前缀干扰
        # 也可以考虑更复杂的逻辑
        perturbed_q = f"{template} {question}"
        
        return f"Question: {perturbed_q}\nAnswer:"

    def generate_pairs(self, samples):
        """
        Given a list of samples (dicts with 'question'), return list of dicts:
        {
            'original_id': ...,
            'simple_prompt': ...,
            'complex_prompt': ...,
            'ground_truth': ...
        }
        """
        pairs = []
        for i, sample in enumerate(samples):
            q = sample['question']
            pairs.append({
                'id': i,
                'simple_prompt': self.create_simple_prompt(q),
                'complex_prompt': self.create_complex_prompt(q),
                'ground_truth': sample.get('answer', '') # Keep full answer for reference
            })
        return pairs

if __name__ == "__main__":
    engine = PerturbationEngine()
    sample_q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    print("Simple:", engine.create_simple_prompt(sample_q))
    print("Complex:", engine.create_complex_prompt(sample_q))
