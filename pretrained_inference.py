import torch
import os
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

def main():
    # Set offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Load model and processor from local paths
    pretrained_model_path = "models/paligemma-3b-pt-224"
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(pretrained_model_path)
    
    print("Loading model...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(pretrained_model_path)
    
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
        return caption
    
    # Test with an example image
    test_image_path = "test-image2.jpeg"
    print(f"\nProcessing image: {test_image_path}")
    caption = generate_caption(test_image_path)
    print(f"\nFinal generated caption: {caption}")

if __name__ == "__main__":
    main() 