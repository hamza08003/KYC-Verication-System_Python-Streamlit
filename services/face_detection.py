import cv2
import face_recognition
import time
from google.cloud import vision


def perform_face_detection(client, binary_img_content):
    img = vision.Image(content=binary_img_content)
    res = client.face_detection(image=img)
    if res.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(res.error.message)
        )
    return res.face_annotations



def extract_face_encodings(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        return face_encodings[0]
    else:
        return None


def compare_faces(encoding1, encoding2):
    results = face_recognition.compare_faces([encoding1], encoding2)
    distance = face_recognition.face_distance([encoding1], encoding2)
    return results, distance
