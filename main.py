from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import base64
import numpy as np

app = FastAPI(title="AgriVision Disease Detection API")

# Load ONNX model
session = ort.InferenceSession("models/resnet50_plant_disease_fp16.onnx")
input_name = session.get_inputs()[0].name

# Labels (update based on your dataset)
LABELS = ["Apple___Apple_scab",
  "Apple___Black_rot",
  "Apple___Cedar_apple_rust",
  "Apple___healthy",

  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
  "Corn_(maize)___Common_rust_",
  "Corn_(maize)___Northern_Leaf_Blight",
  "Corn_(maize)___healthy",

  "Pepper,_bell___Bacterial_spot",
  "Pepper,_bell___healthy",

  "Potato___Early_blight",
  "Potato___Late_blight",
  "Potato___healthy",
  
  "Tomato___Bacterial_spot",
  "Tomato___Early_blight",
  "Tomato___Late_blight",
  "Tomato___Leaf_Mold",
  "Tomato___Septoria_leaf_spot",
  "Tomato___Spider_mites Two-spotted_spider_mite",
  "Tomato___Target_Spot",
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
  "Tomato___Tomato_mosaic_virus",
  "Tomato___healthy"]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img).astype("float16") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.post("/disease-scan")
async def disease_scan(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess(image_bytes)

        outputs = session.run(None, {input_name: input_tensor})
        scores = outputs[0][0]

        # 👉 Apply softmax to get probability
        probabilities = softmax(scores)
        idx = int(np.argmax(probabilities))
        confidence = float(probabilities[idx])

        return {
            "success": True,
            "label": LABELS[idx],
            "confidence": confidence
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/")
def root():
    return {"message": "AgriVision ML Backend Running"}
