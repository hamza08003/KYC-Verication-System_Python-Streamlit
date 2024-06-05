from google.cloud import vision
from pdf2image import convert_from_path, convert_from_bytes
import io



def pdf_to_image(pdf_file_path):
    if isinstance(pdf_file_path, str):
        images = convert_from_path(pdf_file_path)
    else:
        images = convert_from_bytes(pdf_file_path.getvalue())
    return images


def read_image(file_input):
    if isinstance(file_input, str):
        with io.open(file_input, 'rb') as img_file:
            content = img_file.read()
    else:
        content = file_input.getvalue()
    return content


def perform_ocr(client, binary_img_content):
    img = vision.Image(content=binary_img_content)
    res = client.text_detection(image=img)
    if res.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(res.error.message)
        )
    txt_annots = res.text_annotations
    description = ""
    for txt in txt_annots:
        description += txt.description + "\n"
    return description.strip(), txt_annots

