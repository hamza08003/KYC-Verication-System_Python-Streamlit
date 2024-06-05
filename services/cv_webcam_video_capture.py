import cv2
import time



def capture_video_and_extract_frames(duration=5, interval=1):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frames = []
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
        time.sleep(interval)
    
    cap.release()
    cv2.destroyAllWindows()
    return frames
