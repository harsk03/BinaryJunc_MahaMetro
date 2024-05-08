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

cap = cv2.VideoCapture("LeftObject_3.avi")

with open("our_class_list.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

tracker = Tracker()

area = np.array([(8, 373), (421, 229), (497, 241), (305, 492)], np.int32)

count = 0
motion_threshold = 1 
center_points_buffer = []

bag_last_position = {} 
abandoned_bag_duration_threshold = 20

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
    frame = cv2.resize(frame, (640, 640))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu().numpy()).astype("float")

    bag_centers = []
    list = []
    train_detected = False
    person_entered_roi = False  # Variable to track if person entered ROI for the first time
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'train' in c:
            train_detected = True
        if 'Baggage' in c:
            bag_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            bag_centers.append(bag_center)
            cv2.circle(frame, bag_center, 4, (0, 0, 255), -1)

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'Humans' in c:
            # Check if the person's bounding box intersects with the ROI
            results = cv2.pointPolygonTest(np.array(area, np.int32), ((x1 + x2) // 2, (y1 + y2) // 2), False)
            if results > 0:
                person_entered_roi = True  # Set flag to True if person enters ROI for the first time
                list.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # If person entered ROI for the first time, start alert
    if person_entered_roi and not alert_active:
        alert_active = True
        alert_time = time.time()

    # If alert is active and within duration, display alert
    if alert_active and time.time() - alert_time < alert_duration:
        cv2.putText(frame, 'Alert', (20,30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 1)

    # If alert duration is over, reset flags
    if alert_active and time.time() - alert_time >= alert_duration:
        alert_active = False
        alert_time = 0

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x1 + 0.15 * (x2 - x1))
        cy = int(y2) 
        # cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1) 

        results = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        if results > 0:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame, 'Alert', (20,30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 0), 3)

    for bag_center in bag_centers:
        if bag_center in bag_last_position:
            distance = np.sqrt((bag_center[0] - bag_last_position[bag_center]['position'][0])**2 + (bag_center[1] - bag_last_position[bag_center]['position'][1])**2)
            if distance <= motion_threshold:
                bag_last_position[bag_center]['duration'] += 1 
            else:
                bag_last_position[bag_center] = {'position': bag_center, 'duration': 0}
        else:
            bag_last_position[bag_center] = {'position': bag_center, 'duration': 0} 

        if bag_last_position[bag_center]['duration'] >= abandoned_bag_duration_threshold:
            cv2.putText(frame, 'Abandoned Bag!', bag_center, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)

    center_points_buffer.append(bag_centers)
    if len(center_points_buffer) > 5: 
        center_points_buffer.pop(0)

    if len(center_points_buffer) > 1:
        current_centers = center_points_buffer[-1]
        previous_centers = center_points_buffer[-2]
        for curr_center in current_centers:
            found_match = False
            for prev_center in previous_centers:       
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                if distance <= motion_threshold:
                    found_match = True
                    break
            if not found_match:
                cv2.putText(frame, 'Motion Detected!', curr_center, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1)
                cv2.circle(frame, curr_center, 2, (0, 0, 255), -1)
                print("Motion Detected for Backpack!")
    cv2.polylines(frame, [area], True, (255, 255, 0), 3)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
