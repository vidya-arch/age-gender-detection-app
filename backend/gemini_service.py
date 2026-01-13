import google.generativeai as genai
from PIL import Image
import os
import json

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-pro")

def analyze_image(image_path, prompt):
    image = Image.open(image_path)

    response = model.generate_content([
        prompt,
        image
    ])

    return json.loads(response.text)
