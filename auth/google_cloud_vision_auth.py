from google.cloud import vision

def authenticate_vision_client(creds_file):
    return vision.ImageAnnotatorClient.from_service_account_json(creds_file)
