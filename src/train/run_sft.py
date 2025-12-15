import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
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

    # Load items
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], # Adjust based on model architecture (Qwen uses c_attn usually? No, q_proj/v_proj standard now)
        task_type="CAUSAL_LM",
    )

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
        group_by_length=True,
        lr_scheduler_type="constant",
    )



    # Need a formatting function for messages
    def formatting_prompts_func(example):
        output_texts = []
        for messages in example['messages']:
            # Manual chat template application if tokenizer doesn't support it perfectly or for control
            # Qwen uses ChatML usually: <|im_start|>user\n...<|im_end|><|im_start|>assistant\n...<|im_end|>
            # Let's use tokenizer.apply_chat_template if available, else manual
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except:
                # Fallback manual
                user = messages[0]['content']
                assistant = messages[1]['content']
                text = f"User: {user}\nAssistant: {assistant}"
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
        max_seq_length=config.MAX_NEW_TOKENS + 256 # ample space
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
