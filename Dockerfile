FROM python:3.11    
ENV SERVICE_ACCOUNT_CREDS_PATH  'configs/google_cloud_service_account_creds.json'
ENV OPENAI_API_KEY 'sk-proj-Us5P3cyWxikHzzQrShC3T3BlbkFJDMEc9LD9i7j64XXd9g7n'


WORKDIR /app
RUN apt-get update && apt-get install 
RUN apt-get install -y cmake

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install streamlit streamlit-option-menu streamlit-modal google-cloud-vision openai opencv-python-headless python-dotenv face-recognition pdf2image fuzzywuzzy

COPY . .

EXPOSE 8501

CMD [ "streamlit", "run", "app.py" ]
