import object_detection
import cv2
import numpy


od_predictor = object_detection.FastObjectDetector()

cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    ret, frame = cap.read()
    print(od_predictor.predict(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
