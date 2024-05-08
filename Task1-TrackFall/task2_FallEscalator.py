import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time

model = YOLO('last.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right mouse button clicked at: ({x}, {y})")

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture("FallOnEscalator_1.avi")

with open("our_class_list.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

tracker = Tracker()

count = 0
alert_time = 0  # Variable to keep track of the time when alert started
alert_duration = 10  # Duration of alert in seconds
alert_active = False  # Flag to track if alert is active

while True:   
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (640,640))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")
    
    list = []
    person_fall_first = False
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'Fall' in c:
            person_fall_first=True
            list.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) 
            
    if person_fall_first and not alert_active:
        alert_active = True
        alert_time = time.time()
        
    if alert_active and time.time() - alert_time < alert_duration:
        cv2.rectangle(frame,(17, 3),(259, 40),(0,0,255,),thickness=cv2.FILLED)
        cv2.putText(frame, 'Fall Detected!', (20,30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)
        
    if alert_active and time.time() - alert_time >= alert_duration:
        alert_active = False
        alert_time = 0
        
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()