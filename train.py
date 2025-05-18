import torch
import os
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import (
    Trainer,
    TrainingArguments,
    PaliGemmaProcessor,
    BitsAndBytesConfig,
    AutoProcessor,
    PaliGemmaForConditionalGeneration
)
from huggingface_hub import login
from download_model import download_model

def main():
    # Set offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Download model if not already downloaded
    model_path = "models/paligemma-3b-pt-224"
    if not os.path.exists(model_path):
        model_path = download_model()

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset('caglarmert/small_riscm')
    
    # Split dataset
    split_ds = ds["train"].train_test_split(test_size=0.05)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = model_path  # Use local model path
    
    # Initialize processor
    print("Initializing processor...")
    processor = PaliGemmaProcessor.from_pretrained(model_id)

    def collate_fn(examples):
        texts = ["<image> <bos> describe this image." for example in examples]
        labels = [example['caption'] for example in examples]
        images = [example["image"].convert("RGB") for example in examples]
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest"
        )
        tokens = tokens.to(torch.bfloat16).to(device)
        return tokens

    # Initialize model with QLoRA
    print("Initializing model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0}
    )

    # Freeze vision tower and projector
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=2,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        output_dir="finetuned_paligemma_riscm_small",
        bf16=True,
        dataloader_pin_memory=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        data_collator=collate_fn,
        args=training_args
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save model locally instead of pushing to hub
    print("Saving model locally...")
    trainer.save_model("finetuned_paligemma_riscm_small")

if __name__ == "__main__":
    main() 