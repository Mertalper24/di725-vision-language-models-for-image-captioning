# PaliGemma Fine-tuning for RISC-M Dataset

This repository contains code for fine-tuning the PaliGemma model on the RISC-M dataset using QLoRA (Quantized Low-Rank Adaptation).

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Hugging Face token as an environment variable:
```bash
export HF_TOKEN="your_huggingface_token"
```

## Training

To fine-tune the model on your dataset:

```bash
python train.py
```

The training script will:
- Load the RISC-M dataset
- Initialize PaliGemma with QLoRA
- Fine-tune the model
- Save the model to Hugging Face Hub

## Inference

To run inference with the fine-tuned model:

```bash
python inference.py
```

Make sure to update the `finetuned_model_id` in `inference.py` with your model's ID after training.

## Model Architecture

The implementation uses:
- PaliGemma-3B as the base model
- QLoRA for efficient fine-tuning
- 4-bit quantization for reduced memory usage
- LoRA adapters for parameter-efficient fine-tuning

## Dataset

The model is fine-tuned on the RISC-M dataset, which contains images and their corresponding captions.

