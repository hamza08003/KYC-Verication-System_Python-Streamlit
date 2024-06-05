import json


def load_prompts():
    with open('configs/prompt.json') as f:
        prompts = json.load(f)
        id_name_address_extraction_prompt = prompts['id_name_address_extraction_prompt']
        bill_name_address_extraction_prompt = prompts['bill_name_address_extraction_prompt']
    return id_name_address_extraction_prompt, bill_name_address_extraction_prompt
