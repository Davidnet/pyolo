import object_detection
import cv2
import numpy

URL_PATH = "gs://vision-198622-production-models/object-detection/tiny-yolo/tiny-yolo.weights"


od_predictor = object_detection.TinyYOLODetector.get(url=URL_PATH, rm=True)

cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    ret, frame = cap.read()
    print(od_predictor.predict(image=[frame]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
