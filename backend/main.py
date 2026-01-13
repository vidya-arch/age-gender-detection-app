from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import cv2
import multiprocessing

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is running 🚀"}

# 🔥 Load models ONCE
@app.on_event("startup")
def preload_models():
    from deepface import DeepFace
    DeepFace.build_model("Age")
    DeepFace.build_model("Gender")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    from deepface import DeepFace

    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))

    # 🔥 Resize for speed
    img = cv2.resize(img, (640, 480))

    result = DeepFace.analyze(
        img_path=img,
        actions=["age", "gender"],
        detector_backend="opencv",  # FAST
        enforce_detection=False
    )

    if isinstance(result, dict):
        result = [result]

    return {
        "no_of_persons": len(result),
        "predictions": [
            {"age": int(r["age"]), "gender": r["dominant_gender"]}
            for r in result
        ]
    }

if __name__ == "__main__":
    multiprocessing.freeze_support()
