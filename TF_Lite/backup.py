import numpy as np
from tensorflow import keras
from pydantic.main import BaseModel
from PIL import Image
import cv2
import time
import random

def predict(image):
    start = time.time()
    # Load the model
    model = keras.models.load_model('/Users/erinc/Desktop/visiot-backend/app/keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    img2 = image.copy()
    # Make sure to resize all images to 224, 224 otherwise they won't fit in the array
    img2 = img2.resize((224, 224))
    image_array = np.asarray(img2)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    end = time.time()
    print(end - start)
    return prediction

cap = cv2.VideoCapture("rtsp://admin:Leblebi2532356@192.168.2.108:554/cam/realmonitor?channel=1@subtype=1")
frame_rate = 25
prev = 0    

while True:
    time_elapsed = time.time() - prev
    res, image = cap.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()

        ret, frame = cap.read()
        print(predict(frame))
        cv2.imshow("Capturing",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


