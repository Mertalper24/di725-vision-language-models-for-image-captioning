import torch
import os
import wandb
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

def collate_fn(examples, processor):
    """
    Custom collate function for the dataset
    """
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
    tokens = tokens.to(torch.bfloat16)
    return tokens

def compute_metrics(eval_preds, processor):
    """
    Compute metrics for evaluation
    """
    predictions, labels = eval_preds
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate metrics (BLEU, METEOR, CIDEr)
    bleu_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        # Simple word overlap as a proxy for BLEU
        pred_words = set(pred.split())
        label_words = set(label.split())
        overlap = len(pred_words.intersection(label_words))
        bleu_scores.append(overlap / max(len(pred_words), len(label_words)))
    
    return {
        'bleu': np.mean(bleu_scores)
    }

def main():
    # Initialize wandb
    wandb.init(project="paligemma-image-captioning", config={
        "learning_rate": 2e-5,  
        "batch_size": 2,        
        "num_epochs": 2,        
        "warmup_steps": 2,      
        "weight_decay": 1e-6,   
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "gradient_accumulation_steps": 4  
    })
    
    # Set offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Load model and processor
    model_path = "models/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset('caglarmert/small_riscm')
    
    # Split dataset
    split_ds = ds["train"].train_test_split(test_size=0.05)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model with QLoRA
    print("Initializing model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0}  
    )
    
    # Freeze vision tower and projector
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=wandb.config.lora_rank,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"finetuned_paligemma_riscm_{wandb.run.id}",
        learning_rate=wandb.config.learning_rate,
        per_device_train_batch_size=wandb.config.batch_size,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        num_train_epochs=wandb.config.num_epochs,
        weight_decay=wandb.config.weight_decay,
        warmup_steps=wandb.config.warmup_steps,
        logging_steps=2,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        bf16=True,
        dataloader_pin_memory=False,
        report_to=["wandb"],
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=lambda batch: collate_fn(batch, processor),
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, processor)
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main() 