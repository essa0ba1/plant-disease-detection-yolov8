from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import base64
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter as F

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ImageRequest(BaseModel):
    image: str
    confidence_threshold: float

model = YOLO("best.pt").to("cpu")

@app.post("/analyze_image")
def analyze_image(request: ImageRequest):
    image = request.image
    confidence_threshold = request.confidence_threshold
    
    image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
    image_filtered = image.filter(F.SHARPEN)
    image_np = np.array(image)
    
    result = model.predict(source=image_np, conf=confidence_threshold, save=False)
    names = result[0].names
    data = result[0].boxes.data.numpy()
    return {
        "names": names,
        "data": data.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)