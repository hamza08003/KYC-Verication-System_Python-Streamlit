# Function to extract name and address from noisy OCR text using GPT-3
def extract_name_and_address_gpt(client, text, prompt):

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system", 
            "content": "You are expert in extracting name and address from noisy OCR text"
        },
        {
            "role": "user", 
            "content": 
            f"""
            {prompt}\n\n
            {text}
            """
        }
    ]
    )
    name_address = completion.choices[0].message.content
    return name_address


# Function to extract name and address from noisy OCR text using Gemini
def extract_name_and_address_gemini(text, model, safety_config, prompt):
    res = model.generate_content(
        f"""
        {prompt}
        {text}
        """,
        safety_settings=safety_config,
    )

    name_address = res.candidates[0].content.parts[0].text
    return name_address
