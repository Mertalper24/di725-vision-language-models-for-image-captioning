import torch
import os
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

def main():
    # Set offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Load model and processor from local paths
    pretrained_model_path = "google/paligemma-3b-pt-224"
    finetuned_model_path = "/home/ai/di725-vision-language-models-for-image-captioning/finetuned_paligemma_riscm_4as6ggc3"  # Updated path to the new model
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(pretrained_model_path)
    
    print("Loading model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(finetuned_model_path)
    
    # Example inference
    def generate_caption(image_path):
        # Load and process image
        print(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        print(f"Image size: {image.size}")
        
        # Prepare input with proper image token and more specific prompt
        prompt = "<image> Please provide a detailed description of this image."
        print(f"Using prompt: {prompt}")
        
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        print("Input processed successfully")
        
        # Generate caption with improved parameters
        print("Generating caption...")
        output = model.generate(
            **inputs,
            max_new_tokens=100,  # Increased token limit
            do_sample=True,
            temperature=0.8,     # Slightly increased temperature
            top_p=0.95,         # Increased top_p
            num_beams=5,        # Added beam search
            length_penalty=1.0,  # Added length penalty
            repetition_penalty=1.2  # Added repetition penalty
        )
        caption = processor.decode(output[0], skip_special_tokens=True)
        print(f"Raw output: {caption}")
        
        # Remove the prompt from the output, preserving the complete caption
        caption = caption.replace(prompt, "").strip()
        # Remove any remaining "Please provide a detailed description of this image" if it exists
        caption = caption.replace("Please provide a detailed description of this image", "").strip()
        # Remove any leading/trailing newlines
        caption = caption.strip()
        return caption
    
    # Test with an example image
    test_image_path = "/home/ai/1.1/otopark.png"
    print(f"\nProcessing image: {test_image_path}")
    caption = generate_caption(test_image_path)
    print(f"\nFinal generated caption: {caption}")

if __name__ == "__main__":
    main() 