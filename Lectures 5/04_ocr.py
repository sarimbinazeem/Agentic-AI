"""
OCR -> Optical Character Recognition

It reads text from image

Direct way-> Give image to GEMINI VLM

oLD SCHOOL way-> Tesseract does the reading and it gives text to LLM to THINK

"""

import os
import json
import pytesseract
import platform

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

#Check if it exists on a path of windows and store the .exe in variable
if platform.system()=="Windows":
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if os.path.exists(win_path):
        pytesseract.pytesseract.tesseract_cmd=win_path
        
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.1-flash-lite"

image = Image.open("receipt.jpg")

#STEP 1-> extract text from iamge
print("STEP 1: Tesseract OCR -> raw text")

raw_text=pytesseract.image_to_string(image)
print(raw_text)

#STEP 2 -> LLM tunrs text to clean json
print("STEP 2: LLM structures the text -> JSON  (no image, text only!)")

prompt = f"""Here is raw OCR text from a receipt (it may be messy or misread):

{raw_text}

Extract a JSON object with these keys:
  shop_name  (string)
  date       (string)
  items      (list of {{name, price}})
  total      (number)
"""

response=client.models.generate_content(
    model=MODEL,
    contents=[prompt],
    config=types.GenerateContentConfig(
        response_mime_type = "application/json" #this ensures that the output is in json
    )
)

#json into dictionary
data = json.loads(response.text)
print(json.dumps(data,indent=2))

#STEP 3 -> use and save data
print("STEP 3: Save the result")

print("Shop: ",data.get("shop_name"))
print("Total: ",data.get("total"))
print("Number of Items: ", len(data.get("items", [])))

#Open the file 
with open("receipt_output.json","w",encoding="utf-8") as f:
    json.dump(data,f,indent=2)
    
    