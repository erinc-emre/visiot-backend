import cv2
import time
cap = cv2.VideoCapture("rtsp://admin:1234@192.168.2.108:554/cam/realmonitor?channel=1@subtype=1")

frame_rate = 1
prev = 0

while True:
    time_elapsed = time.time() - prev
    res, image = cap.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()

        ret, frame = cap.read()
        cv2.imshow("Capturing",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()