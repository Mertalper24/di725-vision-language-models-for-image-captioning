from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# Load processor and model
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
model = AutoModelForVision2Seq.from_pretrained("google/paligemma-3b-mix-224", torch_dtype=torch.bfloat16).to("cuda")

# Load your image
image = Image.open(r"C:\Users\Mert\Desktop\di725-vision-language-models-for-image-captioning\test-image.jpeg")  # replace with your image path
prompt = "Describe the image."

# Preprocess
inputs = processor(prompt, images=image, return_tensors="pt").to("cuda", torch.bfloat16)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n🧠 Model output:")
print(generated_text)
