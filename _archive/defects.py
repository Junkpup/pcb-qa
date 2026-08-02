from ultralytics import YOLO
import cv2
import numpy as np

print("Starting program...")

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

# Initialize heatmap
heatmap = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working")
        break

    # Initialize heatmap size once
    if heatmap is None:
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    # Run detection
    results = model(frame, conf=0.2)

    # Get boxes
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Increase heat in detected region
            heatmap[y1:y2, x1:x2] += 0.3

    # Normalize heatmap
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # Convert to color heatmap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay heatmap on frame
    overlay = cv2.addWeighted(frame, 0.9, heatmap_color, 0.1, 0)
    cv2.imwrite("heatmap.jpg", heatmap_color) 	
    # Draw bounding boxes also
    annotated = results[0].plot()

    # Show both
    cv2.imshow("PCB Heatmap Overlay", overlay)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()