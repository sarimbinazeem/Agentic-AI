"""

Multimodal rag can send iamges as query then 
chunks->embed -> store
search -> similarity ->answer

Simple 4 steps:
1) Describe every iamge through caption by using gemini vlm
2) Turn every caption to embedding
3) Take query of user and mebed it the nsearch for simialrity
4) Then send iamge + text as query for answer generation
"""

import os
from google import genai
from PIL import Image
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

IMAGE_FOLDER = "images"

#STEP 1-> Caption every image
print("STEP 1: Creating captions for all images...")

#first we get iamges paths
image_paths=[
    os.path.join(IMAGE_FOLDER,name)  #image folder + name
    for name in os.listdir(IMAGE_FOLDER) #go through directory of images
    if name.endswith((".jpg",".png"))
]

captions=[]
for path in image_paths:
    img = Image.open(path)
    
    response = client.models.generate_content(
        model=MODEL,
        contents=[img, "Describe this image in one short sentence."],
    )
    
    captions.append(response.text)
    print(f"  {path}  ->  {response.text.strip()}")
    
#STEP 2 -> creates embeddings of captions
print("\nSTEP 2: Converting captions into embeddings...")

result=client.models.embed_content(model=EMBED_MODEL,contents=captions)
caption_embeddings=[e.values for e in result.embeddings]  #stroe embeddings versio nfo captions

print(f"  Done! Each caption is now a vector of {len(caption_embeddings[0])} numbers.")

#Step 3 -> Search through cosine simialrity
def cosine_similarity(a,b):
    a,b = np.array(a),np.array(b)
    return np.dot(a,b) / ((np.linalg.norm(a)) * (np.linalg.norm(b)))  #this is cosine theta formula linalg gives mangnitude of a and b

def find_best_image(question):
    #we make embedding of question for similairty searching
    q = client.models.embed_content(model=EMBED_MODEL, contents=question)
    q_embedding = q.embeddings[0].values
    
    #comparing the question embedding with everycaption
    scores = [cosine_similarity(q_embedding,emb) for emb in caption_embeddings]
    
    #extract the peak score's index
    best = int(np.argmax(scores))
    

    print(f"  Best match: {image_paths[best]}  (score {scores[best]:.2f})")  #.2f rounds off to 2 decimal places
    return image_paths[best]

#STEP 4-> MUltimodal Rag asking retrieivng
def multimodal_rag(question):
    print(f"\nQUESTION: {question}")
    
    #retrieving
    best_path=find_best_image(question)
    
    #open that image
    best_image=Image.open(best_path)
    response = client.models.generate_content(           # generation
        model=MODEL,
        contents=[best_image, question],
    )
    print("ANSWER:", response.text.strip())
    
multimodal_rag("Which image shows something round and red? Describe it.")
multimodal_rag("Find the shopping bill. What is the total amount?")