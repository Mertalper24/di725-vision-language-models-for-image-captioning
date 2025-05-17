# 03_test_inference.py

from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
import torch

# Load the model and processor
model_id = "google/paligemma-3b-mix-224"
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

# Load and preprocess image
url = "https://huggingface.co/datasets/YiYiXu/test_assets/resolve/main/rock.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

prompt = "<image>\nwhat is in this image?\n"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# Generate output
generated_ids = model.generate(**inputs, max_new_tokens=20)
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Model response:", output)
