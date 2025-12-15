from datasets import load_dataset
import re

class GSM8KLoader:
    def __init__(self, split="train"):
        self.split = split
        self.dataset = load_dataset("gsm8k", "main", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def extract_answer(answer_str):
        """
        Extract the numerical answer from the GSM8K solution string.
        Usually it ends with '#### <number>'
        """
        match = re.search(r"####\s*(-?[\d\.,]+)", answer_str)
        if match:
            return match.group(1).replace(',', '')
        return None

if __name__ == "__main__":
    loader = GSM8KLoader()
    print(f"Loaded {len(loader)} samples.")
    print("Sample 0:", loader[0])
    print("Extracted Answer:", loader.extract_answer(loader[0]['answer']))
