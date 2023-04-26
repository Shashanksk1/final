'''import torch
# Loading in yolov5s - you can switch to larger models such as yolov5m or yolov5l, or smaller such as yolov5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
l=['zidane.jpg','bus.jpg','zidane.jpg','bus.jpg','zidane.jpg']
for i in range(5):
    img = l[i]  # or file, Path, PIL, OpenCV, numpy, list
    results = model(img)
    print("----->","{ ",results," }")'''

import cv2
import time
i=0
'''while i<10:
    # Capture image from default camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # Display the captured image
    cv2.imshow('frame', frame)

    # Wait for 1 second
    time.sleep(1)

    # Close the window and release the camera
    cv2.destroyAllWindows()
    cap.release()'''

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Display the captured image
cv2.imshow('frame', frame)

