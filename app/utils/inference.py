
import torch
import torchvision
from typing import List
from PIL import Image
import io

from torchvision import transforms




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_bytes: bytes):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((64, 64)),  # assuming TinyVGG takes 64x64
        torchvision.transforms.ToTensor(),  # Converts to [0, 1] and CHW format
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(torchvision.transforms.ToTensor()(image) * 255)
    return image_tensor.unsqueeze(0)  # Add batch dim

def predict(model: torch.nn.Module, image_bytes: bytes, class_names: List[str]):
    image = preprocess_image(image_bytes).to(device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        preds = model(image)
        probs = torch.softmax(preds, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        return {
            "class": class_names[pred_label],
            "probability": round(probs[0][pred_label].item(), 3)
        }



def preprocess_image(image_bytes: bytes):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # converts to [0, 1]
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_image = transform(image)
    return tensor_image.unsqueeze(0)  # Add batch dimension
