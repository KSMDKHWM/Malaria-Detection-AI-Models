from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI(title="DenseNet50 Model API")

# Load your model
model = tf.keras.models.load_model("malaria_densenet50_cnn_model.h5")

class ImageRequest(BaseModel):
    image: str  # base64 encoded image

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def root():
    return {"status": "DenseNet50 API is running"}

@app.post("/predict")
def predict(request: ImageRequest):
    image_bytes = base64.b64decode(request.image)
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array)
    return {"predictions": preds.tolist()}
