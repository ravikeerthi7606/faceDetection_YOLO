import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)

    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            confidence = float(conf)

            text = f"{label} ({confidence:.2f})"

            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Live Detection (Hari vs Ravi)", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()