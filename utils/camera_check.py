import cv2


# Function to check if camera is available
def check_camera_available():
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        return False
    cap.release()
    return True
    


# def check_camera_available():
#     cap = cv2.VideoCapture(0)
#     if cap.isOpened():
#         cap.release()
#         return True
#     else:
#         return False

