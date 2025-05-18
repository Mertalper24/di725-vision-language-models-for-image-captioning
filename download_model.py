import os
from huggingface_hub import snapshot_download, login

def download_model():
    print("Downloading PaliGemma model files...")
    
    # Login to Hugging Face
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Please enter your Hugging Face token: ")
    login(token=token)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the model files
    model_path = snapshot_download(
        repo_id="google/paligemma-3b-pt-224",
        repo_type="model",
        local_dir="models/paligemma-3b-pt-224",
        token=token  # Pass token explicitly
    )
    
    print(f"Model files downloaded to: {model_path}")
    return model_path

if __name__ == "__main__":
    download_model() 