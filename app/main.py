import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket
import numpy as np
from tensorflow import keras
from pydantic.main import BaseModel
from PIL import Image
import cv2
import time

app = FastAPI(title = "VISIOT ML", description = "Vision for IOT ML model", version = "0.1.0")

cap = cv2.VideoCapture("rtsp://admin:Leblebi2532356@192.168.2.108:554/cam/realmonitor?channel=1@subtype=1")
frame_rate = 1
prev = 0



@app.get("/")
def home():
    return {"message": "Hello World"}



@app.websocket("/predict")
async def predict(websocket: WebSocket):

    await websocket.accept()

    cap = cv2.VideoCapture("rtsp://admin:Leblebi2532356@192.168.2.108:554/cam/realmonitor?channel=1@subtype=1")
    model = keras.models.load_model('/Users/erinc/Desktop/fast-api/app/keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    frame_rate = 1
    prev = 0

    while True:
        try:
            time_elapsed = time.time() - prev
            res, frame = cap.read()

            if time_elapsed > 1./frame_rate:
                prev = time.time()

                ret, frame = cap.read()

                frame = frame.resize((224, 224))
                frame_array = np.asarray(frame)
                normalized_frame_array = (frame_array.astype(np.float32) / 127.0) - 1

                data[0] = normalized_frame_array

                prediction = model.predict(data)
                return str(prediction)
        except Exception as e:
            print('error:', e)
            break

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)