import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os, uuid
from config import MODEL_PATH, OUTPUT_DIR

# Carrega modelo pré-treinado (CartoonGAN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Transformação de entrada
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# Transformação de saída
postprocess = transforms.Compose([
    transforms.Normalize(mean=[-1,-1,-1], std=[2,2,2]),
    transforms.Lambda(lambda x: torch.clamp(x,0,1)),
    transforms.ToPILImage()
])

def convert_to_cartoon(image_bytes: bytes) -> Image.Image:
    """Converte imagem normal em cartoon usando CartoonGAN"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    cartoon_img = postprocess(output_tensor.squeeze(0).cpu())
    return cartoon_img

def save_cartoon(img: Image.Image, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    random_id = uuid.uuid4().hex[:8]
    filename = f"OnóxIA_{random_id}.png"
    path = os.path.join(output_dir, filename)
    img.save(path, format="PNG")
    return filename
