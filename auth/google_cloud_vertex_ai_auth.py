from google.cloud import aiplatform
from google.oauth2 import service_account

def authenticate_vertex_ai(creds_file, project_id, location):
    creds = service_account.Credentials.from_service_account_file(creds_file)
    aiplatform.init(project=project_id, location=location, credentials=creds)
