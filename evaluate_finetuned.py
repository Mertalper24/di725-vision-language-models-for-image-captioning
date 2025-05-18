import torch
import os
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import Counter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from datasets import load_dataset
import argparse
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

def calculate_cider(reference, candidate, n=4):
    """
    Calculate CIDEr score for a single reference-candidate pair
    """
    def compute_tf(tokens):
        return Counter(tokens)
    
    def compute_idf(tokens, all_tokens):
        return np.log(len(all_tokens) / (1 + sum(1 for t in all_tokens if tokens in t)))
    
    # Tokenize
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    # Get unique tokens
    all_unique_tokens = list(set(ref_tokens + cand_tokens))
    
    # Compute TF
    ref_tf = compute_tf(ref_tokens)
    cand_tf = compute_tf(cand_tokens)
    
    # Compute IDF
    all_tokens = [ref_tokens]  # In real scenario, this would be all references
    ref_idf = {token: compute_idf(token, all_tokens) for token in all_unique_tokens}
    
    # Compute TF-IDF vectors using the same vocabulary
    ref_vec = np.zeros(len(all_unique_tokens))
    cand_vec = np.zeros(len(all_unique_tokens))
    
    for i, token in enumerate(all_unique_tokens):
        ref_vec[i] = ref_tf.get(token, 0) * ref_idf.get(token, 0)
        cand_vec[i] = cand_tf.get(token, 0) * ref_idf.get(token, 0)
    
    # Compute cosine similarity
    if np.all(ref_vec == 0) or np.all(cand_vec == 0):
        return 0.0
    
    similarity = np.dot(ref_vec, cand_vec) / (np.linalg.norm(ref_vec) * np.linalg.norm(cand_vec))
    return similarity

def evaluate_model(model_path, processor, model, run_id):
    """
    Evaluate model on test data and return metrics
    """
    print(f"\nEvaluating fine-tuned model...")
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset('caglarmert/small_riscm')
    test_ds = ds["train"].train_test_split(test_size=0.05)["test"]
    
    metrics = {
        'bleu': [],
        'meteor': [],
        'cider': []
    }
    
    # Store some example predictions for visualization
    examples = []
    
    for idx, example in enumerate(tqdm(test_ds)):
        # Get image
        image = example["image"].convert("RGB")
        
        # Generate caption
        prompt = "<image> Please provide a detailed description of this image."
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_beams=5,
            length_penalty=1.0,
            repetition_penalty=1.2
        )
        
        # Clean up the generated caption
        generated_caption = processor.decode(output[0], skip_special_tokens=True)
        # Remove the prompt and any leading/trailing whitespace
        generated_caption = generated_caption.replace(prompt, "").strip()
        # Remove any remaining "Please provide a detailed description of this image" if it exists
        generated_caption = generated_caption.replace("Please provide a detailed description of this image", "").strip()
        # Remove any leading/trailing newlines
        generated_caption = generated_caption.strip()
        
        # Get reference caption
        reference_caption = example['caption']
        
        # Calculate metrics
        # BLEU
        reference_tokens = [reference_caption.split()]
        candidate_tokens = generated_caption.split()
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=SmoothingFunction().method1)
        
        # METEOR - tokenize both reference and generated captions
        ref_tokens = nltk.word_tokenize(reference_caption.lower())
        gen_tokens = nltk.word_tokenize(generated_caption.lower())
        meteor_score_val = meteor_score([ref_tokens], gen_tokens)
        
        # CIDEr
        cider_score = calculate_cider(reference_caption, generated_caption)
        
        metrics['bleu'].append(bleu_score)
        metrics['meteor'].append(meteor_score_val)
        metrics['cider'].append(cider_score)
        
        # Store example predictions (first 5 images)
        if idx < 5:
            examples.append({
                'image_path': f"example_{idx+1}.jpg",  # Just store a placeholder path
                'generated_caption': generated_caption,
                'reference_caption': reference_caption
            })
    
    # Calculate average metrics
    avg_metrics = {
        'bleu': np.mean(metrics['bleu']),
        'meteor': np.mean(metrics['meteor']),
        'cider': np.mean(metrics['cider'])
    }
    
    # Save results
    results = {
        'model_name': f'finetuned_{run_id}',
        'metrics': avg_metrics,
        'examples': examples
    }
    
    os.makedirs('evaluation_results', exist_ok=True)
    with open(f'evaluation_results/finetuned_metrics_{run_id}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return avg_metrics, examples

def load_pretrained_metrics():
    """
    Load the pretrained model metrics from the saved file
    """
    with open('evaluation_results/pretrained_metrics.json', 'r') as f:
        return json.load(f)

def get_latest_model_path():
    """
    Get the path of the most recently trained model
    """
    try:
        with open('latest_run_id.txt', 'r') as f:
            run_id = f.read().strip()
        return f"finetuned_paligemma_riscm_{run_id}"
    except FileNotFoundError:
        return "finetuned_paligemma_riscm_small"  # Default path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned model')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to the fine-tuned model directory. If not specified, uses the most recent model.')
    args = parser.parse_args()
    
    # If no model path specified, use the latest one
    if args.model_path is None:
        args.model_path = get_latest_model_path()
        print(f"No model path specified. Using the most recent model: {args.model_path}")
    
    # Set offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Get run ID from command line or use timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load pretrained metrics
    try:
        pretrained_results = load_pretrained_metrics()
        pretrained_metrics = pretrained_results['metrics']
        print("Loaded pretrained model metrics")
    except FileNotFoundError:
        print("Warning: Pretrained metrics not found. Please run evaluate_pretrained.py first.")
        return
    
    # Load processor from pretrained model
    pretrained_model_path = "models/paligemma-3b-pt-224"
    print(f"Loading processor from pretrained model at {pretrained_model_path}...")
    processor = AutoProcessor.from_pretrained(pretrained_model_path)
    
    # Load fine-tuned model
    finetuned_model_path = args.model_path
    print(f"Loading fine-tuned model from {finetuned_model_path}...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(finetuned_model_path)
    
    # Evaluate model
    finetuned_metrics, finetuned_examples = evaluate_model(
        finetuned_model_path, 
        processor, 
        model,
        run_id
    )
    
    # Print comparison
    print("\nResults Comparison:")
    print("\nPretrained Model:")
    for metric, value in pretrained_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nFine-tuned Model:")
    for metric, value in finetuned_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nImprovement:")
    for metric in pretrained_metrics.keys():
        improvement = ((finetuned_metrics[metric] - pretrained_metrics[metric]) / pretrained_metrics[metric]) * 100
        print(f"{metric.upper()}: {improvement:+.2f}%")
    
    # Print example comparisons
    print("\nExample Predictions:")
    for i, (pretrained_ex, finetuned_ex) in enumerate(zip(pretrained_results['examples'], finetuned_examples)):
        print(f"\nImage {i+1}:")
        print(f"Pretrained: {pretrained_ex['generated_caption']}")
        print(f"Fine-tuned: {finetuned_ex['generated_caption']}")
        print("Reference caption:")
        print(f"  {pretrained_ex['reference_caption']}")

if __name__ == "__main__":
    main() 