from ultralytics import YOLO
import cv2
import cvzone
import math
from email_alert import send_email_alert  

def ppe_detection(file):
    if file is None:
        cap = cv2.VideoCapture(0)  
        cap.set(3, 1280)
        cap.set(4, 720)
    else:
        cap = cv2.VideoCapture(file)  

    model = YOLO("best.pt")

    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                  'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

    myColor = (0, 0, 255)
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

        cv2.imshow("PPE Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
   
    file = r"F:\Computer_vision\PPE_detection_YOLO\Videos\ppe-2.mp4"
    ppe_detection(file)
