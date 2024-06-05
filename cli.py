from auth.google_cloud_vision_auth import authenticate_vision_api
from auth.gpt_auth import authenticate_opeani_client
from services.ocr import pdf_to_image, read_image, perform_ocr
from utils.string_comparison import normalize_text, compare_names, compare_addresses
from services.face_detection import perform_face_detection, capture_video_and_extract_frames, extract_face_encodings, compare_faces
from utils.name_address_extraction import extract_name_and_address_gpt
from openai import OpenAI
from dotenv import load_dotenv
import cv2
import os



################################################################################
#                            CONFIGURATIOS                                     #
################################################################################

load_dotenv()

SERVICE_ACCOUNT_CREDS_FILEPATH = 'configs/google_cloud_service_account_creds.json'
PROJECT_ID = 'master-pager-425012-c8'
LOCATION = 'us-central1'

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# MODEL_SAFETY_CONFIG = {
#     generative_models.HarmCategory.HARM_CATEGORY_UNSPECIFIED: generative_models.HarmBlockThreshold.BLOCK_NONE,
#     generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
#     generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
#     generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
#     generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
# }

BILL_NAME_ADDRESS_EXTRACTION_PROMPT = "Please extract the full name and address (if address is present) from the following Bill's OCR text. Focus on identifying the holder's name and any associated address (if any) while ignoring any irrelevant or repetitive information. Make sure your output in the this format (very necessary): Name: <Full Name> Address: <Address> and don't write anything else"
ID_NAME_ADDRESS_EXTRACTION_PROMPT = "Please extract the name, surname (if any) and address from the following ID's OCR text. Focus on identifying the primary holder's name and any associated address ignoring any irrelevant or repetitive information. Make sure your output in the this format (very necessary): Name: <Full Name> Address: <Address> and don't write anything else"



################################################################################
#                            MAIN FUNCTION                                     #
################################################################################
def main():
    # Load Model safety config and name/address extraction prompt
    print("Loading Model Safety Config and Name/Address Extraction Model Prompt...")

    # Authenticate with Google Cloud APIs
    print("Authenticating with Google Cloud APIs...")
    vision_client = authenticate_vision_api(SERVICE_ACCOUNT_CREDS_FILEPATH)
    gpt_client = authenticate_opeani_client(OPEN_AI_API_KEY)


    ###################################################################################
    #                OCR on the documents & Address Verification                      #
    ###################################################################################

    # Input paths to ID document and bill
    id_front_image_path = input("Enter the path to the ID document Front image: ")
    id_back_image_path = input("Enter the path to the ID document Back image: ")
    bill_path = input("Enter the path to the bill image or pdf: ")

    # OCR on the ID document
    id_front_image_binary_content = read_image(id_front_image_path)
    id_back_image_binary_content = read_image(id_back_image_path)
    id_text_front, _ = perform_ocr(vision_client, id_front_image_binary_content)
    id_text_back, _ = perform_ocr(vision_client, id_back_image_binary_content)
    id_text = id_text_front + "\n" + id_text_back
    print("\nText Extracted from ID")

    # OCR on the bill
    if bill_path.endswith('.pdf'):
        bill_images = pdf_to_image(bill_path)
        bill_text = ""
        for i, bill_image in enumerate(bill_images):
            bill_text += perform_ocr(vision_client, bill_image)
        print("Text Extracted from Bill")

    elif bill_path.endswith('.jpg') or bill_path.endswith('.png'):
        bill_image_content = read_image(bill_path)
        bill_text, _ = perform_ocr(vision_client, bill_image_content)
        print("Text Extracted from Bill")

    else:
        print("Invalid file format. Please provide a pdf, jpg or png file.")
        bill_path = input("\nEnter the path to the bill image or pdf: ")

    # Extract name and address from the bill
    bill_name_address = extract_name_and_address_gpt(gpt_client, bill_text, BILL_NAME_ADDRESS_EXTRACTION_PROMPT)
    print(f"\nName and Address Extracted from bill:\n{bill_name_address}\n")

    # Extract name and address from ID
    id_name_address = extract_name_and_address_gpt(gpt_client, id_text, ID_NAME_ADDRESS_EXTRACTION_PROMPT)
    if 'Peach M.C' or 'Pechi M.C' or 'Pech IMC' or 'Peach MSC' or 'Phase MC' in id_name_address:
        id_name_address = id_name_address.replace('Peach M.C', 'HMC')
        id_name_address = id_name_address.replace('Pechi M.C', 'HMC')
        id_name_address = id_name_address.replace('Pech IMC', 'HMC')
        id_name_address = id_name_address.replace('Peach MSC', 'HMC')
        id_name_address = id_name_address.replace('Phase MC', 'HMC')
    print(f"\nExtracted the data from Front side of ID:\n{id_name_address}\n")

    # Parse names and addresses
    bill_name = bill_name_address.split("Address:")[0].replace("Name:", "").strip()
    bill_address = bill_name_address.split("Address:")[1].strip()
    id_name = id_name_address.split("Address:")[0].replace("Name:", "").strip()
    id_address = id_name_address.split("Address:")[1].strip()

    print(f"Bill Name: {bill_name}\n\nBill Address: {bill_address}")
    print(f"ID Name: {id_name}\n\nID Address: {id_address}")


    # Compare names and addresses using fuzzy matching
    name_match = compare_names(bill_name, id_name)
    address_match = compare_addresses(bill_address, id_address)

    if name_match and address_match:
        print("Name and address match between bill and ID. Verification successful.")
    else:
        print("Name and address do not match. Verification failed.")


    ##############################################################################
    #                Face Detection and Verification                             #
    ##############################################################################

    id_face_encoding = extract_face_encodings(id_front_image_path)
    if id_face_encoding is not None:
        frames = capture_video_and_extract_frames()
        match_found = False
        
        for i, frame in enumerate(frames):
            frame_path = f'frame_{i}.jpg'
            cv2.imwrite(frame_path, frame)
            live_face_encoding = extract_face_encodings(frame_path)
            if live_face_encoding is not None:
                res, dist = compare_faces(id_face_encoding, live_face_encoding)
                print(res)
                print(dist)
                if res[0] and dist <= 0.55:
                    print(f"Face verification successful on frame {i}")
                    match_found = True
                    break
        
        if not match_found:
            print("Face verification failed")
    else:
        print("No face detected in ID document")


if __name__ == "__main__":
    main()
