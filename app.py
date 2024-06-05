import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_modal import Modal
from auth.google_cloud_vision_auth import authenticate_vision_client
from auth.gpt_auth import authenticate_opeani_client
from services.ocr import pdf_to_image, read_image, perform_ocr
from utils.prompts_loader import load_prompts
from utils.name_address_extraction import extract_name_and_address_gpt
from utils.string_comparison import compare_names, compare_addresses
from utils.camera_check import check_camera_available
from services.face_detection import perform_face_detection, extract_face_encodings, compare_faces
from services.cv_webcam_video_capture import capture_video_and_extract_frames
from dotenv import load_dotenv
import cv2
import os



################################################################################
#                            CONFIGURATIOS                                     #
################################################################################

load_dotenv()

SERVICE_ACCOUNT_CREDS = os.getenv("SERVICE_ACCOUNT_CREDS_PATH")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

id_name_address_extraction_prompt, bill_name_address_extraction_prompt = load_prompts()


################################################################################
#                            MAIN FUNCTION                                     #
################################################################################
def main():
    # Authenticate with Google Cloud and OpenAI APIs
    @st.cache_resource
    def get_vision_client():
        return authenticate_vision_client(SERVICE_ACCOUNT_CREDS)

    @st.cache_resource
    def get_gpt_client():
        return authenticate_opeani_client(OPEN_AI_API_KEY)

    vision_client = get_vision_client()
    gpt_client = get_gpt_client()

    # # Streamlit App
    # st.title("KYC Verification System")

    # tabs = st.tabs(["About & Instructions", "Name and Address Verification", "Face Verification"])

    tabs = option_menu(
        menu_title= "KYC Verification System",
        options = ["Instructions", "Docs Verification", "Face Verification"],
        icons = ["info-circle-fill", "file-break-fill", "camera"],
        menu_icon="patch-check-fill",
        default_index = 0,
        orientation = "horizontal",
    )


    ###################################################################################
    #                            About & Instructions Tab                             #
    ###################################################################################

    if tabs == "Instructions":
        st.markdown("<h3 style='text-align: center;'>ⓘ About KYC Verification System</h3>", unsafe_allow_html=True)
        st.info("Welcome to the KYC Verification System. This application helps in verifying the identity of individuals by processing their documents and facial data. Follow the instructions in each tab to complete the verification process")

        st.subheader("Instructions")

        st.write("""    
        - **Name and Address Verification**: Upload the front and back images of your ID and any utility bill to verify your name and address. 
        - **Face Verification**: Capture a video of yourself to verify your face against the ID document.
        """)

        st.warning("Please ensure that all uploaded images/docs and captured video is clear and unobstructed to avoid verification errors.")

    ###################################################################################
    #             OCR on the documents - Name & Address Verification Tab              #
    ###################################################################################
    if tabs == "Docs Verification":

        st.markdown("<h3 style='text-align: center;'>📌 Name and Address Verification</h3>", unsafe_allow_html=True)

        id_front_image = st.file_uploader("Upload the front image of your ID document", type=["jpg", "png"], key="id_front")
        if id_front_image is not None:
            st.image(id_front_image, caption='Uploaded ID Front Image', use_column_width=True)

        id_back_image = st.file_uploader("Upload the back image of your ID document", type=["jpg", "png"], key="id_back")
        if id_back_image is not None:
            st.image(id_back_image, caption='Uploaded ID Back Image', use_column_width=True)

        bill_file = st.file_uploader("Upload your bill (PDF, JPG, PNG)", type=["pdf", "jpg", "png"], key="bill")
        if bill_file is not None:
            if bill_file.type == "application/pdf":
                bill_images = pdf_to_image(bill_file)
                for img in bill_images:
                    st.image(img, caption='Uploaded Bill Image', use_column_width=True)
            else:
                st.image(bill_file, caption='Uploaded Bill Image', use_column_width=True)


        if st.button("Verify Name and Address", use_container_width=True):
            if id_front_image and id_back_image and bill_file:

                with st.status("OCR on the documents...", expanded=True) as status:

                    # OCR on the ID document
                    st.write("Performing OCR on the ID ...")
                    id_front_image_binary_content = read_image(id_front_image)
                    id_back_image__binary_content = read_image(id_back_image)

                    id_front_ocr_text, _ = perform_ocr(vision_client, id_front_image_binary_content)
                    id_back_ocr_text, _ = perform_ocr(vision_client, id_back_image__binary_content)
                    id_ocr_text = id_front_ocr_text + "\n" + id_back_ocr_text

                    st.write("Performing OCR on the Bill...")
                    # OCR on the bill
                    if bill_file.type == "application/pdf":
                        bill_images = pdf_to_image(bill_file)
                        bill_ocr_text = ""
                        for img in bill_images:
                            bill_image_binary_content = read_image(img)
                            bill_ocr_text += perform_ocr(vision_client, bill_image_binary_content)
                    else:
                        bill_image_binary_content = read_image(bill_file)
                        bill_ocr_text, _ = perform_ocr(vision_client, bill_image_binary_content)
                    
                    status.update(label="OCR complete!", state="complete", expanded=False)

                    
                # Extract name and address
                with st.status("Extracting names and addresses...", expanded=True) as status:

                    st.write("Extracting name and address from the bill...")
                    bill_name_address = extract_name_and_address_gpt(gpt_client, bill_ocr_text, bill_name_address_extraction_prompt)

                    st.write("Extracting name and address from the ID...")
                    id_name_address = extract_name_and_address_gpt(gpt_client, id_ocr_text, id_name_address_extraction_prompt)

                    status.update(label="Extraction complete!", state="complete", expanded=False)
                    
                # Parse names and addresses
                bill_name = bill_name_address.split("Address:")[0].replace("Name:", "").strip()
                bill_address = bill_name_address.split("Address:")[1].strip()
                id_name = id_name_address.split("Address:")[0].replace("Name:", "").strip()
                id_address = id_name_address.split("Address:")[1].strip()

                st.write(f"Bill Name: {bill_name}")
                st.write(f"Bill Address: {bill_address}")
                st.write(f"ID Name: {id_name}")
                st.write(f"ID Address: {id_address}")
                        
                    
                # Compare names and addresses using fuzzy matching
                with st.spinner("Comparing names and addresses..."):
                    name_match = compare_names(bill_name, id_name)
                    address_match = compare_addresses(bill_address, id_address)

                    if name_match and address_match:
                        st.success("Name and address match between bill and ID. Verification successful.")
                    else:
                        st.error("Name and address do not match. Verification failed.")
            else:
                st.warning("Please upload all required documents.")


    ###################################################################################
    #                              Face Verification Tab                              #
    ###################################################################################
    if tabs == "Face Verification":
        check_camera_available()

        st.markdown("<h3 style='text-align: center;'>👤 Face Verification</h3>", unsafe_allow_html=True)

        if not check_camera_available():
            st.warning("No camera detected. Please connect a camera to continue.")
        else:
            id_face_image = st.file_uploader("Upload the front image of your ID document for face verification", type=["jpg", "png"], key="id_face")
            
            modal = Modal("Face Verification Instructions", key="face-verification-modal")
            open_modal = st.button("Start Face Verification", use_container_width=True)

            if open_modal:
                modal.open()

            if modal.is_open():
                with modal.container():
                    st.write("""
                    **Instructions for Face Verification:**
                    - Hold your face still in front of the camera.
                    - Ensure good lighting and avoid any background movement.
                    - We will capture a short video to verify your identity.
                    """)
                    confirm_button = st.button("Confirm and Start Video Capture")
                    if confirm_button:
                        modal.close()
                        if id_face_image:
                            with st.spinner("Processing..."):
                                # Extract face encoding from ID image
                                id_face_encoding = extract_face_encodings(id_face_image)
                                if id_face_encoding is not None:
                                    st.write("Face detected in ID document. Capturing video for live verification...")

                                    # Capture video and extract frames
                                    frames = capture_video_and_extract_frames(duration=5, interval=1)
                                    match_found = False

                                    for i, frame in enumerate(frames):
                                        frame_path = f'frame_{i}.jpg'
                                        cv2.imwrite(frame_path, frame)
                                        live_face_encoding = extract_face_encodings(frame_path)
                                        if live_face_encoding is not None:
                                            res, dist = compare_faces(id_face_encoding, live_face_encoding)
                                            if res[0] and dist <= 0.55:
                                                st.success(f"Face verification successful on frame {i}")
                                                match_found = True
                                                break

                                    if not match_found:
                                        st.error("Face verification failed")
                                else:
                                    st.error("No face detected in ID document")
                        else:
                            st.warning("Please upload the front image of your ID document.")

if __name__ == "__main__":
    main()
