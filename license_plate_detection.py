import cv2
from ultralytics import YOLO

# Load the YOLOv8 model from the specified local path
model = YOLO('/home/temoc/Desktop/checkpoint/yolov8_car_plate_model.pt')

# Set the correct video device (e.g., /dev/video0)
cap = cv2.VideoCapture('/dev/video0')  # Replace with the correct path

if not cap.isOpened():
    print("Error: Unable to open video capture")

while cap.isOpened():
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from video feed")
        break

    # Perform inference on the frame
    results = model(frame, verbose=False)

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.cpu().numpy()  # Get detected boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"License Plate {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 License Plate Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
