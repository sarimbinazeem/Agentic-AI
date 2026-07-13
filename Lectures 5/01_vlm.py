"""
vlm have a simple process

It can see images and talk about them

it returns text answer

"""

import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.1-flash-lite"

#wE OPEN THE IMAGE
image = Image.open("sample.jpg")

# Demo 1 -> Describe Image
print("DEMO 1: Describe the image")

response= client.models.generate_content(
    model=MODEL,
    content=[image,"Describe the image in 2 simple sentences"]
)
print(response.text)

# Demo 2  ->Ask question about image
print("DEMO 2: Ask a question about the image")

response= client.models.generate_content(
    model=MODEL,
    content=[image,"What colors do you see in the image? Is it Day or Night?"]
)
print(response.text)

# Demo 3 -> Compare the two images
print("DEMO 3: Compare two images")


image2 = Image.open("sample2.jpg")

response= client.models.generate_content(
    model=MODEL,
    content=[image,image2,"What is different between these two images?"]
)
print(response.text)

