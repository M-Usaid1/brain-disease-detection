from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles
import shutil
import os

app = FastAPI(
    title="Brain Tumor Detection API",
    description="Backend API using YOLOv11",
    version="1.0.0"
)
PREDICTION_DIR = "runs/detect/predictions/result"

os.makedirs(PREDICTION_DIR, exist_ok=True)

app.mount(
    "/predictions",
    StaticFiles(directory=PREDICTION_DIR),
    name="predictions"
)

# Load trained model
model = YOLO("best.pt")

# Create folders if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("predictions", exist_ok=True)


@app.get("/")
def home():
    return {
        "status": "Running",
        "message": "Brain Tumor Detection API"
    }


@app.get("/health")
def health():
    return {
        "model": "Loaded Successfully",
        "status": "Healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image
    image_path = os.path.join("uploads", file.filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    results = model.predict(
    source=image_path,
    save=True,
    project="predictions",
    name="result",
    exist_ok=True
    )
    print("Save Directory:", results[0].save_dir)
    print("Results:", results)
    detections = []

    for result in results:
        for box in result.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3)
            })


    prediction_image = f"/predictions/{file.filename}"

    return JSONResponse({
    "message": "Prediction Successful",
    "detections": detections,
    "prediction_image": prediction_image
    })