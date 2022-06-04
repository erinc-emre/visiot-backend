import uvicorn
from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow import keras
from pydantic.main import BaseModel
from PIL import Image


app = FastAPI(title = "VISIOT ML", description = "Vision for IOT ML model", version = "0.1.0")

@app.get("/")
def home():
    return {"message": "Hello World"}



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = keras.models.load_model('/Users/erinc/Desktop/fast-api/app/keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(file.file)


    image = image.resize((224, 224))
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    return str(prediction)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)