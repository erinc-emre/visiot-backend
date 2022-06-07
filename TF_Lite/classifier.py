
'''
import numpy as np
from tensorflow import keras
from pydantic.main import BaseModel
from PIL import Image
import time
import random
'''

from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time
import cv2



# Load an image to be classified.




def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]

data_folder = "/home/erinc/Desktop/visiot-backend/in_house/"

model_path = data_folder + "bird_model2.tflite"
label_path = data_folder + "label.txt"

interpreter = Interpreter(model_path)

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")





import threading

# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()

# Start the camera
camera = cv2.VideoCapture("rtsp://admin:Leblebi2532356@192.168.2.108:554/cam/realmonitor?channel=1@subtype=1")

# Start the cleaning thread
cam_cleaner = CameraBufferCleanerThread(camera)

# Use the frame whenever you want
while True:
    if cam_cleaner.last_frame is not None:
        frame =  cam_cleaner.last_frame

        frame = cv2.resize(frame, (320, 320))

        #image = Image.open(data_folder + "test.jpg").convert('RGB').resize((width, height))
        cv2.imshow('test',frame)
        # Classify the image.
        time1 = time.time()
        label_id, prob = classify_image(interpreter, frame)
        time2 = time.time()
        classification_time = np.round(time2-time1, 3)
        print("Classificaiton Time =", classification_time, "seconds.")

        # Read class labels.
        labels = load_labels(label_path)

        # Return the classification label of the image.
        #classification_label = labels[label_id]
        #print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")

    cv2.waitKey(10)
