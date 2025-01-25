from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap=cv2.VideoCapture(r"C:\Users\KIIT\Desktop\Other_essential\NOT_DELETE\pithon\video\cars.mp4")


#above one will set height and width of webcam
model=YOLO(r"C:\Users\KIIT\Desktop\Other_essential\NOT_DELETE\pithon\yolo_weights\yolov8n.pt")

#classnames--
class_names = [
    "person","bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "stop sign", "parking meter", "fire hydrant", "street light",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "laptop", "mouse", "remote control", "keyboard", "cell phone",
    "clock", "pot", "pan", "toaster oven", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier","toilet", "tub", "shower",
    "sink", "bed", "table", "chair", "sofa", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote control", "keyboard", "cell phone", "clock",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

mask=cv2.imread(r"C:\Users\KIIT\Desktop\Other_essential\NOT_DELETE\pithon\carcounter\mask.png")

#Tracking
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)  #maxage determine how many frames it will wait for a object to comeback

total_count=[]
#creatingline

limits=[400,297,673,297]


#for accessing webcam
while(True):
    success,img=cap.read()
    #masking on our original img
    imgregion=cv2.bitwise_and(img,mask)
    
    detections=np.empty((0,5))
    
    #sending only that to img region
    results=model(imgregion,stream=True)
    for r in results:
        boxes=r.boxes ##bounding box of each results
        for box in boxes:   #height and width 
            #yeah wla cvzone
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),(3))
            w,h=x2-x1,y2-y1
            
            #adding confidence
            
            conf=math.ceil((box.conf[0]*100))/100
            #findclassname
            cls=int(box.cls[0])
            
            currentclass=class_names[cls]
            
            if(currentclass=="car" or currentclass=="truck" or currentclass=="bus" or currentclass=="motorbike" and conf>0.3):
                x1_offset = 5
                y1_offset = 5
                # cvzone.cornerRect(img, (x1 + x1_offset, y1 + y1_offset, w, h), l=10, rt=2, t=3, colorR=(255, 0, 255), colorC=(230, 255, 210))
                # cvzone.putTextRect(img,f"{currentclass} {conf}",((max(0,x1+10),max(20,y1))),scale=1,thickness=1)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))
                # cvzone.cornerRect(img,(x1,y1,w,h),l=4,rt=4,t=3,colorR=(255,0,255),colorC=(0,255,0))
    
    
    
    
    
    
    result_tracker=tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    
    
    for results in result_tracker:
        x1,y1,x2,y2,id=results
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(results)
        w,h=x2-x1,y2-y1
        cx,cy=x1+w//2,y1+h//2
        if(limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15):
            if(total_count.count(id)==0):
                total_count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2,colorR=(30,20,89))

        
    cvzone.putTextRect(img,f"count:{len(total_count)}",(50,50))   
    cv2.imshow("Image",img)
    # cv2.imshow("ImageRegion",imgregion)
    cv2.waitKey(1)
