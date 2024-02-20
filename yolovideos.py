#bir video üzerinden görüntü işleme yapma.
from ultralytics import YOLO 
import time 
import math
import cv2
import cvzone 

cap=cv2.VideoCapture('videos\motorbikes-1.mp4') #burada videoyu yakalaması için path verdik. 
model=YOLO('yolov8n.pt') #modeli kurduk 
#bu sınıflar coco datasete ait 
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes #burada r.boxes kutucuk almak için yaptık.
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1 #uzunluk ve genişliği bu sekilde bulduk.
            cvzone.cornerRect(img, (x1, y1, w, h)) #koşeleri renkli olacak şekilde bir cerceve yaptık. bu kullanım cvzone ile yapılıyor.
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100 #burada ne kadar düzgün bir şekilde tahmin ediyor ona bakıyoruz. 
            #confidence değeri bu anlama geliyor. ne kadar 1 e yakınsa o kadar iyi anlamına gelir. 0 ile 1 arasında değer alıyor.
            # Class Name
            cls = int(box.cls[0]) 
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1) 
            #scale ile yazıları küçültebilirsin. buradaki ayarlardan özellikler değiştirilebilir.
            # bu puttextrect cvzone'dan gelen bir fonksıyon. çercevenin uzunluguna gore üstüne gelmesi gereken texti merkeze alıyor. 
             #nesne cerceve dısına cıksa bile onun confidence oranını vermeye devam edicek
 
    cv2.imshow("Image", img) #space tusuna basınca video kapanıcak.
    key=cv2.waitKey(1) 
    if key == ord(' '):
        break
   
