"""
Document Understandings 

OCR only reads text while docuents also have other strcutred data like
tables
header 
footers
sections
strcutures

We send whole PDF directyl with no page splitting

"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
client= genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-2.5-flash"

#Reading the file as PDF in bytes
with open("sample.pdf","rb") as f:
    pdf_bytes=f.read()
    
#Wrap it into a part so that the model knows it as a PDF
pdf_part= types.Part.from_bytes(data=pdf_bytes,mime_type="application/pdf")

#Helper function ask from pdf
def ask_pdf(question):
    response=client.models.generate_content(
        model=MODEL,
        contents=[pdf_part, question],
    )
    
    return response.text

# ---------- DEMO 1: Summary of the whole document ----------
print("=" * 50)
print("DEMO 1: Summarize the document")
print("=" * 50)
print(ask_pdf("Summarize this document in 3 simple sentences."))


# ---------- DEMO 2: Extract TABLES ----------
print("=" * 50)
print("DEMO 2: Extract all tables")
print("=" * 50)
print(ask_pdf("Extract every table in this document as a markdown table."))


# ---------- DEMO 3: Understand CHARTS ----------
print("=" * 50)
print("DEMO 3: Understand the charts")
print("=" * 50)
print(ask_pdf("Describe the charts in this document. Which quarter was the best and worst?"))


# ---------- DEMO 4: Headers, footers and structure ----------
print("=" * 50)
print("DEMO 4: Document structure")
print("=" * 50)
print(ask_pdf("""Describe the structure of this document:
    - What is written in the header of the pages?
- What is written in the footer of the pages?
- How many pages, tables and charts does it have?"""))