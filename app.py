from pathlib import Path
from contextlib import asynccontextmanager
import base64
import json
import io

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from model import load_model
from fastapi.staticfiles import StaticFiles


# ===============================
# PATH SETUP
# ===============================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "efficientnet_b1+cbam_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


# ===============================
# DEVICE SETUP
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ===============================
# LOAD CLASS NAMES
# ===============================
if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(f"class_names.json not found: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)
    CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
NUM_CLASSES = len(CLASS_NAMES)


# ===============================
# IMAGE TRANSFORM
# Must match training transform
# ===============================
IMAGE_SIZE = 240

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ===============================
# FASTAPI LIFESPAN
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.model = load_model(
        model_path=MODEL_PATH,
        num_classes=NUM_CLASSES,
        device=device,
    )
    print("Model loaded successfully.")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


# ===============================
# CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# HELPER FUNCTIONS
# ===============================
def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_gradcam(model, input_tensor, target_class):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Same as your project training code
    target_layer = model.features[-1]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(input_tensor)
        loss = output[:, target_class]

        model.zero_grad()
        loss.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM could not capture activations/gradients.")

        grads = gradients[0]
        acts = activations[0]

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * acts, dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam

    finally:
        forward_handle.remove()
        backward_handle.remove()


def build_gradcam_images(raw_img: Image.Image, cam: np.ndarray):
    img_np = np.array(raw_img.resize((IMAGE_SIZE, IMAGE_SIZE)))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.4 + img_np * 0.6
    overlay = overlay.astype(np.uint8)

    original_img = Image.fromarray(img_np)
    heatmap_img = Image.fromarray(heatmap)
    overlay_img = Image.fromarray(overlay)

    return {
        "original": pil_to_data_url(original_img),
        "heatmap": pil_to_data_url(heatmap_img),
        "overlay": pil_to_data_url(overlay_img),
    }


# ===============================
# ROOT ENDPOINT
# ===============================
@app.get("/")
def root():
    return {
        "message": "TongueCheck API is running",
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
    }


# ===============================
# HEALTH ENDPOINT
# ===============================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Backend is running",
        "model_loaded": True,
        "device": str(device),
    }


# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query(default="efficientnet_b1+cbam_model"),
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(device)

        model = app.state.model
        model.eval()

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_index].item())

        probability_dict = {
            CLASS_NAMES[i]: float(probabilities[i].item())
            for i in range(NUM_CLASSES)
        }

        return {
            "prediction": CLASS_NAMES[predicted_index],
            "pred_class": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "probabilities": probability_dict,
            "model_name": model_name,
        }

    except Exception as e:
        print("Prediction error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# GRAD-CAM ENDPOINT
# Returns object expected by GradCAMPanel.jsx:
# original, heatmap, overlay, pred_class, confidence
# ===============================
@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(...),
    model_name: str = Query(default="efficientnet_b1_cbam"),
):
    try:
        image_bytes = await file.read()
        raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = transform(raw_img).unsqueeze(0).to(device)

        model = app.state.model
        model.eval()

        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

        pred_class_index = int(torch.argmax(probabilities).item())
        confidence_percent = float(probabilities[pred_class_index].item()) * 100

        cam = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_class=pred_class_index,
        )

        images = build_gradcam_images(raw_img, cam)

        return {
            "pred_class": CLASS_NAMES[pred_class_index],
            "confidence": confidence_percent,
            "model_name": model_name,
            **images,
        }

    except Exception as e:
        print("Grad-CAM error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))