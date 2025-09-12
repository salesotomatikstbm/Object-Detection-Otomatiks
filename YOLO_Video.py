from ultralytics import YOLO
import cv2
import math
import cvzone
from email_alert import send_email_alert  

def video_detection(path_x):
    video_capture = path_x
   
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("YOLO-Weights/ppe.pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    email_sent = False  

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        violation_detected = False  

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.5:
                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        myColor = (0, 0, 255)  
                        violation_detected = True
                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)  
                    else:
                        myColor = (255, 0, 0)  

                
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                       colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

      
        if violation_detected and not email_sent:
            print("🚨 Violation detected! Sending email...")
            email_sent = send_email_alert()  
            if email_sent:
                print("✅ Email sent successfully!")
            else:
                print("❌ Email sending failed.")

       
        if not violation_detected:
            email_sent = False

        yield img 

    cap.release()
    cv2.destroyAllWindows()
