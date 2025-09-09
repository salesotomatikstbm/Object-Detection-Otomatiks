from ultralytics import YOLO
import cv2
import math
import cvzone
from email_alert import send_email_alert  # ✅ Import the send_email_alert function

def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("YOLO-Weights/ppe.pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    email_sent = False  # Prevent spamming multiple emails per run

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        violation_detected = False  # flag

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.5:
                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        myColor = (0, 0, 255)  # Red for violation
                        violation_detected = True
                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)  # Green for safe
                    else:
                        myColor = (255, 0, 0)  # Other classes

                    # Use cvzone to put text on the image
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                       colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        # ✅ Send Email if violation detected and not already sent
        if violation_detected and not email_sent:
            print("🚨 Violation detected! Sending email...")
            email_sent = send_email_alert()  # Call the email function
            if email_sent:
                print("✅ Email sent successfully!")
            else:
                print("❌ Email sending failed.")

        # Reset email alert if no violation
        if not violation_detected:
            email_sent = False

        yield img  # Return the processed frame

    cap.release()
    cv2.destroyAllWindows()
