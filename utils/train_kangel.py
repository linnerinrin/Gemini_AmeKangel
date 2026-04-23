import os
import warnings
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
from pathlib import Path
root = Path(__file__).parent.parent
warnings.filterwarnings("ignore")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"{root}/models/qwen2.5",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=4,
    lora_dropout=0,
    bias="none",
)

dataset = load_dataset("json", data_files=f"{root}/data/kangel.json", split="train")
dataset = standardize_sharegpt(dataset)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5"
)


def format_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            c,
            tokenize=False,
            add_generation_prompt=False
        ) for c in convos
    ]
    return {"text": texts}

dataset = dataset.map(format_func, batched=True)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
        save_steps=100,
        save_total_limit=2,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer.train()

model.save_pretrained(f"{root}/models/lora_kangel")
tokenizer.save_pretrained(f"{root}/models/lora_kangel")