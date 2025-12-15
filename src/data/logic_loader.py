from datasets import load_dataset
import os

class LogicLoader:
    def __init__(self, dataset_name="yale-nlp/folio", split="train"):
        self.dataset_name = dataset_name
        self.split = split
        try:
            self.dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            self.dataset = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def format_context(premises):
        """
        Joins premises into a single context string.
        """
        return "\n".join(premises).strip()

    @staticmethod
    def extract_label(label_str):
        """
        Folio labels: 'True', 'False', 'Uncertain'
        """
        return str(label_str).strip()

if __name__ == "__main__":
    # Test loading
    loader = LogicLoader()
    if len(loader) > 0:
        print("Sample 0:", loader[0])
        print("Context:", loader.format_context(loader[0]['premises']))
