from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
import random
import numpy as np
from utils.sort import *
import time

model = YOLO('YOLO_weights/yolov8l.pt')
# model.to('mps')
# may need to move to kaggle for better speed, mps is not working well with YOLO

# results = model ('cats_pic1.jpeg',show=True)
# cv2.waitKey(0)


cap = cv2.VideoCapture("car_videos2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = 'result.mp4'
#initialize a VideoWriter object for video writing 
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width,frame_height) )#,isColor=False)


random_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

# mask = cv2.imread('mask.png')

#add tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#add area of interest
area_1 = [(280,000), (750,0),(1100,280),(850,400)] 
vehicles_within_area = {}
vehicles_staying_time = {}
vehicles_stopped = 0


if not cap.isOpened():
    exit()
else: 
    print('cap is opened')

while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame,stream =True)
    detections = np.empty((0,5))

    #showing the area 
    cv2.polylines(frame, [np.array(area_1,np.int32)],True,(15,220,10),6)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding boxes
            x1, y1, x2,y2 = box.xyxy[0]
            x1 , y1, x2 , y2 = int(x1), int(y1), int(x2), int(y2)
            w , h = x2-x1 , y2-y1

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100

            # Class Name
            cls = int(box.cls[0])
            names = model.names
            currentClass = names[cls]

            if currentClass in ['car','bus','truck'] and conf >0.3:
                cv2.rectangle(frame,(x1,y1), (x2,y2),(128, 128, 128) ,thickness=1)
                currentArray = np.array ([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))       
                         
    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1 , y1, x2 , y2 = int(x1), int(y1), int(x2), int(y2)
        w , h = x2-x1 , y2-y1
        print(result)
        
        #getting the centre point of the object
        cx,cy = x1 +w//2 , y1+h//2

        #check if the point if the centre is in the polygon, 1 of yes , -1 for no ,0 for edge 
        check_centre = cv2.pointPolygonTest(np.array(area_1,np.int32), (int(cx),int(cy)), False)

        if check_centre >=0:
            
            if Id not in vehicles_within_area :
                vehicles_within_area[Id] = (cx,cy)
                vehicles_staying_time[Id] = 0
                
            else:
                if vehicles_staying_time[Id] + 1/fps > 1.0 and vehicles_staying_time[Id]<1.0:
                    vehicles_stopped += 1
                vehicles_staying_time[Id] += 1/fps

                cv2.rectangle(frame,(x1,y1), (x2,y2),random_colors[int(Id%10)],thickness=2)
                cv2.putText(frame, f'Car {Id} stayed {vehicles_staying_time[Id]:.2f}s', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)  
                cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)

                                            
    rect_x1 = frame_width - 800  # X coordinate
    rect_y1 = 20                 # Y coordinate
    rect_x2 = frame_width - 10    # X coordinate
    rect_y2 = 80
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), thickness=-1)
    # Add text on top of the rectangle
    cv2.putText(frame, f'Count of Vehicles stopped more than 1 second : {vehicles_stopped}', (rect_x1 + 10, rect_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=2)     
    
    out.write(frame)
    cv2.imshow('video',frame)
    if cv2.waitKey(int(1000/fps)) ==  ord('q'):
        break   

   
cap.release()
out.release()
cv2.destroyAllWindows()