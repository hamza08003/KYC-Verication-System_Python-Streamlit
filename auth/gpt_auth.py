from openai import OpenAI

def authenticate_opeani_client(api_key):
        client = OpenAI(api_key=api_key)
        return client
