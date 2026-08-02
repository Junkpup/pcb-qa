import cv2
 
def get_available_cameras(max_cameras=10):
    """Check for available camera indices."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available

cameras = get_available_cameras()
if cameras:
    print(f"Available cameras: {cameras}")
else:
    print("No cameras found.") 
    