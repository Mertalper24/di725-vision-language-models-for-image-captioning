import json
import os
import wandb
from datetime import datetime

def parse_run_name(filename):
    """
    Parse hyperparameters from filename to create a descriptive run name
    """
    # Remove the prefix and extension
    name = filename.replace('finetuned_metrics_', '').replace('.json', '')
    
    # Split by double dashes to get individual parameters
    params = name.split('--')
    
    # Create a descriptive name
    run_name = "Fine-tuned"
    for param in params:
        if param.startswith('learning_rate'):
            run_name += f" LR={param.split('_')[-1]}"
        elif param.startswith('lora_rank'):
            run_name += f" Rank={param.split('_')[-1]}"
        elif param.startswith('num_epochs'):
            run_name += f" Epochs={param.split('_')[-1]}"
        elif param.startswith('batch_size'):
            run_name += f" BS={param.split('_')[-1]}"
        elif param.startswith('gradient_accumulation_steps'):
            run_name += f" GA={param.split('_')[-1]}"
    
    return run_name

def calculate_improvement(finetuned_metrics, baseline_metrics):
    """
    Calculate percentage improvement over baseline
    """
    improvements = {}
    for metric in baseline_metrics:
        baseline = baseline_metrics[metric]
        finetuned = finetuned_metrics[metric]
        improvement = ((finetuned - baseline) / baseline) * 100
        improvements[f"{metric}_improvement"] = improvement
    return improvements

def log_metrics_to_wandb(json_file_path, baseline_metrics=None, run_name=None):
    """
    Log metrics from a JSON file to WANDB
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    # Initialize wandb
    if run_name is None:
        if results['model_name'] == 'pretrained':
            run_name = "Pretrained Model (Baseline)"
        else:
            # Extract filename from path
            filename = os.path.basename(json_file_path)
            run_name = parse_run_name(filename)
    
    # Determine tags
    tags = []
    if results['model_name'] == 'pretrained':
        tags = ['baseline']
    else:
        tags = ['fine-tuned']
        # Add hyperparameter tags
        filename = os.path.basename(json_file_path)
        params = filename.replace('finetuned_metrics_', '').replace('.json', '').split('--')
        for param in params:
            if param.startswith('learning_rate'):
                tags.append(f"lr_{param.split('_')[-1]}")
            elif param.startswith('lora_rank'):
                tags.append(f"rank_{param.split('_')[-1]}")
            elif param.startswith('num_epochs'):
                tags.append(f"epochs_{param.split('_')[-1]}")
            elif param.startswith('batch_size'):
                tags.append(f"bs_{param.split('_')[-1]}")
            elif param.startswith('gradient_accumulation_steps'):
                tags.append(f"ga_{param.split('_')[-1]}")
    
    wandb.init(
        project="paligemma-image-captioning",
        name=run_name,
        tags=tags,
        config={
            "model_name": results['model_name'],
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    
    # Log metrics
    wandb.log(results['metrics'])
    
    # If this is a fine-tuned model and we have baseline metrics, log improvements
    if baseline_metrics and results['model_name'] != 'pretrained':
        improvements = calculate_improvement(results['metrics'], baseline_metrics)
        wandb.log(improvements)
        
        # Create a comparison table
        comparison_table = wandb.Table(columns=["Metric", "Baseline", "Fine-tuned", "Improvement"])
        for metric in baseline_metrics:
            comparison_table.add_data(
                metric.upper(),
                f"{baseline_metrics[metric]:.4f}",
                f"{results['metrics'][metric]:.4f}",
                f"{improvements[f'{metric}_improvement']:+.2f}%"
            )
        wandb.log({"comparison_with_baseline": comparison_table})
    
    # Log examples as a table
    examples_table = wandb.Table(columns=["Image", "Generated Caption", "Reference Caption"])
    for example in results['examples']:
        examples_table.add_data(
            example['image_path'],
            example['generated_caption'],
            example['reference_caption']
        )
    wandb.log({"examples": examples_table})
    
    # Close wandb
    wandb.finish()

def main():
    # Create evaluation_results directory if it doesn't exist
    os.makedirs('evaluation_results', exist_ok=True)
    
    # First, load baseline metrics
    baseline_metrics = None
    pretrained_json = 'evaluation_results/pretrained_metrics.json'
    if os.path.exists(pretrained_json):
        with open(pretrained_json, 'r') as f:
            baseline_results = json.load(f)
            baseline_metrics = baseline_results['metrics']
        print("Logging pretrained metrics to WANDB...")
        log_metrics_to_wandb(pretrained_json)
    
    # Log all fine-tuned metrics
    for filename in os.listdir('evaluation_results'):
        if filename.startswith('finetuned_metrics_') and filename.endswith('.json'):
            json_path = os.path.join('evaluation_results', filename)
            print(f"Logging {filename} to WANDB...")
            log_metrics_to_wandb(json_path, baseline_metrics)

if __name__ == "__main__":
    main() 