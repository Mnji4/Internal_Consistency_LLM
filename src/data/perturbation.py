import random
from src.config import config

class LogicPerturbationEngine:
    def __init__(self):
        self.templates = config.PERTURBATION_TEMPLATES
        # Some generic irrelevant rules/facts
        self.noise_facts = [
            "The moon is made of gray rock.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Cats are popular pets.",
            "Computers use binary code.",
            "The capital of France is Paris.",
            "Roses are red.",
            "Violets are blue.",
        ]

    def create_simple_prompt(self, context, question):
        """
        Standard Logic Prompt with CoT:
        Context: ...
        Question: ...
        Answer: Let's think step by step.
        """
        return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer: Let's think step by step."

    def create_complex_prompt(self, context, question):
        """
        Injects noise into the context.
        """
        # Mix in irrelevant facts at random positions in the context lines?
        # Or just append/prepend.
        # Interleaving is harder to detect for models.
        
        lines = context.split('\n')
        
        # Inject 1-2 noise facts
        num_noise = random.randint(1, 2)
        for _ in range(num_noise):
            noise = random.choice(self.noise_facts)
            insert_pos = random.randint(0, len(lines))
            lines.insert(insert_pos, noise)
            
        new_context = "\n".join(lines)
        
        # Also maybe add a perturbation instruction prefix?
        # For logic, changing the context is usually enough to test robustness.
        
        return f"Context:\n{new_context}\n\nQuestion:\n{question}\n\nAnswer:"

if __name__ == "__main__":
    engine = LogicPerturbationEngine()
    ctx = "All humans are mortal.\nSocrates is a human."
    q = "Is Socrates mortal?"
    print("Simple:\n", engine.create_simple_prompt(ctx, q))
    print("-" * 20)
    print("Complex:\n", engine.create_complex_prompt(ctx, q))
