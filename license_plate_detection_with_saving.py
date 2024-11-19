import cv2
import os
import time
from ultralytics import YOLO

# Load the YOLOv8 model from the specified local path
model = YOLO('/home/temoc/Desktop/checkpoint/yolov8_car_plate_model.pt')

# Directory to save images
output_dir = './detected_plates'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Log file for the vehicle count
log_file = os.path.join(output_dir, 'vehicle_count.txt')

# Initialize vehicle count
vehicle_count = 0

# Function to check the size of the output directory
def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# Open video capture (adjust index or device path)
cap = cv2.VideoCapture('/dev/video0')

if not cap.isOpened():
    print("Error: Cannot open video capture")
    exit()

# Maximum directory size (5 GB)
max_size = 5 * 1024 * 1024 * 1024  # 5 GB in bytes

while cap.isOpened():
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    # Perform inference on the frame
    results = model(frame, verbose=False)

    # Check if any license plate is detected
    if results and len(results[0].boxes) > 0:
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Get detected boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score

                # Crop the detected license plate from the frame
                cropped_plate = frame[y1:y2, x1:x2]

                # Save the cropped license plate image with a timestamp
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_name = f"{timestamp}_plate_{vehicle_count+1}.jpg"
                image_path = os.path.join(output_dir, image_name)
                cv2.imwrite(image_path, cropped_plate)

                # Increment the vehicle count
                vehicle_count += 1

                # Update the log file with the current count
                with open(log_file, 'w') as f:
                    f.write(f"Vehicle Count: {vehicle_count}\n")

        # Draw bounding boxes and labels on the original frame
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"License Plate {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 License Plate Detection', frame)

    # Check the directory size
    if get_directory_size(output_dir) >= max_size:
        print("Reached maximum storage limit of 5 GB. Stopping...")
        break

    # Press 'q' to exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print final message
print(f"Detection completed. Total vehicles detected: {vehicle_count}")
