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
    import random
    
    texts = ["<image> <bos> describe this image." for example in examples]
    
    # Handle multiple captions - randomly select one from caption_1 to caption_5
    labels = []
    for example in examples:
        caption_keys = ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']
        available_captions = [example[key] for key in caption_keys if example[key] is not None and example[key].strip()]
        if available_captions:
            selected_caption = random.choice(available_captions)
        else:
            selected_caption = "No caption available"  # Fallback
        labels.append(selected_caption)
    
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
        "learning_rate": 2e-5,          # Higher for LoRA
        "batch_size": 8,                # Increased for better stability  
        "num_epochs": 2,                # More epochs for larger dataset
        "warmup_steps": 500,            # ~1% of total steps
        "weight_decay": 1e-4,           # Slightly higher regularization
        "lora_rank": 16,                # Higher rank for better capacity
        "lora_alpha": 32,               # 2x rank (standard practice)
        "lora_dropout": 0.1,            # Keep same
        "gradient_accumulation_steps": 16  # Increased from 8
    })
    
    # COMMENT OUT this line to enable downloading
    # os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Load model and processor
    model_path = "google/paligemma-3b-pt-224"
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset('caglarmert/full_riscm')  # Changed from 'caglarmert/small_riscm'
    
    # Check the number of examples in the dataset
    print(f"Number of examples in the dataset: {len(ds['train'])}")

    # Split dataset
    split_ds = ds["train"].train_test_split(test_size=0.05)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]

    # Check the size of the test dataset
    print(f"Number of examples in the test dataset: {len(test_ds)}")
    
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
    

    # Set environment variable for memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Reduce evaluation batch size
    training_args = TrainingArguments(
        output_dir=f"finetuned_paligemma_riscm_{wandb.run.id}",
        learning_rate=wandb.config.learning_rate,
        per_device_train_batch_size=wandb.config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        num_train_epochs=wandb.config.num_epochs,
        weight_decay=wandb.config.weight_decay,
        warmup_steps=wandb.config.warmup_steps,
        logging_steps=2,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        bf16=True,
        dataloader_pin_memory=False,
        report_to=["wandb"],
        remove_unused_columns=False,
        evaluation_strategy="steps",
        eval_steps=1000,
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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main() 