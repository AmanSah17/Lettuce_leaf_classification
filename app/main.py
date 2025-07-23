from fastapi import FastAPI, UploadFile, File
import torch
import mlflow.pyfunc
import mlflow.pytorch
from app.utils.inference import preprocess_image
import os

app = FastAPI()

# === Load model from MLflow using run_id ===
RUN_ID = "cf73d0d7f39c4ea7b14d72c19f87d421"
MODEL_URI = f"runs:/{RUN_ID}/model"

# This loads the model (pytorch flavor)
model = mlflow.pytorch.load_model(model_uri=MODEL_URI)

# === Load class names ===
CLASSES_PATH = "app/utils/classes.txt"
with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Preprocess
    image = preprocess_image(image_bytes).to(torch.device("cpu"))

    # Predict
    model.eval()
    with torch.inference_mode():
        preds = model(image)
        probs = torch.softmax(preds, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        result = {
            "class": class_names[pred_label],
            "probability": round(probs[0][pred_label].item(), 3)
        }

    return result
