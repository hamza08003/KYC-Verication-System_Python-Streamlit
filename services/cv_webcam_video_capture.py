import cv2
import time



def capture_video_and_extract_frames(feed_frame, duration=5, interval=1):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frames = []
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feed_frame.image(frame, channels='RGB')
        if not ret:
            continue
        frames.append(frame)
        time.sleep(interval)
    
    cap.release()
    cv2.destroyAllWindows()
    return frames
