import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 'yolov5s' is a lightweight model
cap = cv2.VideoCapture(0)  # 0 = default webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 expects RGB images
    results = model(frame[..., ::-1])  # BGR (OpenCV) to RGB

    # Get predictions in [{xyxy, confidence, class, name}]
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        # Draw bounding boxes and put labels
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Show the frame
    cv2.imshow('Real-time Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()