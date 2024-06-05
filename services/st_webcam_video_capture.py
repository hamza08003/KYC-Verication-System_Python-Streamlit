# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# import av
# import cv2

# class VideoTransformer(VideoTransformerBase):
#     def __init__(self, id_face_encoding):
#         self.id_face_encoding = id_face_encoding
#         self.match_found = False

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         if not self.match_found:
#             face_encodings = extract_face_encodings(img)
#             if face_encodings is not None:
#                 res, dist = compare_faces(self.id_face_encoding, face_encodings)
#                 if res[0] and dist <= 0.55:
#                     self.match_found = True
#                     st.success("Face verification successful")
#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# def start_video_capture(id_face_encoding):
#     webrtc_streamer(key="example", video_transformer_factory=lambda: VideoTransformer(id_face_encoding))



# def capture_video_and_extract_frames(duration=5, interval=1):
#     cap = cv2.VideoCapture(0)
#     start_time = time.time()
#     frames = []
    
#     while (time.time() - start_time) < duration:
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         frames.append(frame)
#         time.sleep(interval)
    
#     cap.release()
#     cv2.destroyAllWindows()
#     return frames
