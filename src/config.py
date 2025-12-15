import os

class Config:
    # Environment
    # 增加 HF 镜像配置，解决连接问题
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # Model
    MODEL_NAME = "Qwen/Qwen3-0.6B" 
    
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    
    # Data
    DATASET_NAME = "folio" # or 'ruletaker' if available locally
    
    # Generation
    BATCH_SIZE = 8
    MAX_NEW_TOKENS = 64 # Logic answers are usually short (True/False/Uncertain)
    
    # Perturbation
    # Logic specific perturbations
    PERTURBATION_TEMPLATES = [
        # Irrelevant context injection
        "Also, the sky is blue and cats are cute.",
        "Note that in the year 2025, cars might fly.",
        "Ignore the fact that apples are red.",
        "It is irrelevant that the sun rises in the east.",
    ]
    
    # Labels
    LABELS = ["True", "False", "Uncertain"]

config = Config()
