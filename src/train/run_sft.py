import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from src.config import config
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(config.DATA_DIR, "sft_train.jsonl"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(config.PROJECT_ROOT, "checkpoints"))
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # 1. Load Data
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    # 2. Load Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Apply LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Prepare Data
    # Pre-process dataset to convert messages to input_ids
    def preprocess_function(example):
        messages = example['messages']
        user_text = messages[0]['content']
        assistant_text = messages[1]['content']
        
        full_text = f"User: {user_text}\n\nAssistant: {assistant_text}" + tokenizer.eos_token
        
        tokenized = tokenizer(full_text, truncation=True, max_length=512, padding=False)
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"]
        }

    # Remove raw columns to avoid collator errors
    train_dataset = dataset.map(
        preprocess_function, 
        batched=False, 
        remove_columns=dataset.column_names
    )

    # 5. Training Args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=args.lr,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        remove_unused_columns=False, 
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
