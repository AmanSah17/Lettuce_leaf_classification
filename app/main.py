from fastapi import FastAPI, UploadFile, File
import torch
from app.utils.inference import preprocess_image
from app.utils.model import TinyVGG  # import your model class
import os

app = FastAPI()

# === Load class names ===
CLASSES_PATH = "app/utils/classes.txt"
with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === Load model ===
model_weights_path = "app\utils\TinyVGG_DNN_model_0_weights.pth"

# Ensure model architecture matches the saved state_dict
model = TinyVGG(input_shape=3, hidden_units=64, output_shape=len(class_names))
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Preprocess
    image = preprocess_image(image_bytes).to(torch.device("cpu"))

    # Predict
    with torch.inference_mode():
        preds = model(image)
        probs = torch.softmax(preds, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        result = {
            "class": class_names[pred_label],
            "probability": round(probs[0][pred_label].item(), 3)
        }

    return result
