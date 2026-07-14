"""
Video Pipeline

Direct way is to hand the whole video and it reads frame by frame and list audio at once

Old-School way is to extract real values from the video and only handle the values to LLM

video goes through OpenCV / cv2 -> numbers per frame (brightness,color,motion)
numbers -> timeline of event (scene changes)
timeline goes to LLM which is a plain english story of video

THe LLM doesnt see pixel. It only reads the timeline of measurements. A video is just many frames in a row so we:
1) read frames by frame
2) measure each frame brightness color and motion
3) mark a scene change
4) LLM turn the number tiemline into description

"""

import os 
import json
import time

from dotenv import load_dotenv
from google import genai

import numpy as np
import cv2

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.1-flash-lite"    

VIDEO_FILE = "sample_video.mp4"

#color helper function that turns rgb into human word
def color_name(r,g,b):
    if max(r, g, b) < 45:
        return "black/dark"
    if min(r, g, b) > 200:
        return "white/bright"
    top = max(r, g, b)
    if r == top and r > g + 25 and r > b + 25:
        return "red"
    if g == top and g > r + 25 and g > b + 25:
        return "green"
    if b == top and b > r + 25 and b > g + 25:
        return "blue"
    if r > 150 and g > 150 and b < 100:
        return "yellow"
    return "mixed/grey"

if not os.path.exists(VIDEO_FILE):
    print(f"No '{VIDEO_FILE}' found. Run:  python 00_create_samples.py")
    raise SystemExit

#STEP 1 -> Read the Video's Basic Facts
print("STEP 1: OpenCV -> basic video facts (no AI here)")

cap = cv2.VideoCapture(VIDEO_FILE)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Duration
duration= (frames/fps) if fps else 0

#Store into list
facts = {
    "duration_seconds": round(duration, 1),
    "fps": round(fps, 1),
    "total_frames": frames,
    "resolution": f"{width}x{height}",
}
print(json.dumps(facts, indent=2))

#STEP 2 -> measure frames
print("STEP 2: measure frames (brightness, colour, motion) -> timeline")

#THERE ARE MANY FRAMES WE ONLY CALCULALTE 12 FRAMES.
step = max(1,frames//12) #// is division ceiling operator

timeline=[]
prev_small=None # Previous frame 
frame_idx=0

while True:
    ok,frame= cap.read()  #read() gives bool,frame 
    if not ok:
        break
    
    #Calculating Brightness (Open CV returns in B G R)
    avg_b,avg_g,avg_r= frame.reshape(-1,3).mean(axis=0) #respahe gives 3 colros and mean takes average of them
    brightness=int((avg_b+avg_g+avg_r)/3 /255 *100 ) #takes average of 3 vaiable then normalize it by dividing it by 255 and then mulitpling by 100 for percentage
    
    #Motion detection -> comparing previous and current frame
    small=cv2.cvtColor(
        cv2.resize(frame,(64,64)), #converts resolution to 64x64 because it is easier to work on
        cv2.COLOR_BGR2GRAY  #CONVERTS TO GRRAY
    )
    
    #if there is no previous frame then no comprison
    if prev_small is None:
        motion=0
    else:
        #take difference from both frames and then take avergae
            motion = int(np.mean(np.abs(small.astype(int) - prev_small.astype(int)))
                         / 255 * 100)
        
        #normalzie it and multiply by 100 for percentage
            prev_small=small
        
    #hchecking if the event is big or not
    event= "scene change" if motion >15 else "steady"
    
    timeline.append(
{            "time_s": round(frame_idx / fps, 1),
            "color": color_name(int(avg_r), int(avg_g), int(avg_b)),
            "brightness_0_100": brightness,
            "motion_0_100": motion,
            "event": event,}
    )
    
    frame_idx+=1
    #now go frame by frame
    
#releaase the video (Close it)
cap.release()
print(json.dumps(timeline,indent=2))

#Step 3-> NOW GIVE THE Measurements and Facts to LLM
print("STEP 3: LLM describes the video (it never sees a pixel)")

prompt = f"""You are given ONLY computer-vision measurements of a short video.
You cannot see the video itself.

Video facts:
{json.dumps(facts, indent=2)}

Timeline of sampled moments (time, dominant colour, brightness, motion, event):
{json.dumps(timeline, indent=2)}

Based only on these numbers, describe in 3 short lines what likely happens in
this video: how many scenes, their colours, and where the motion/action is.
"""

response=client.models.generate_content(
    model=MODEL,
    contet=[prompt],
)

print(response.text)

#STEP 4-> DIRECT UPLOADING THE VIDEO 
print("STEP 4: DIRECT — the model WATCHES the video itself (every pixel)")

#wE Upload the video to GEMINI API and it see the frames of video and hear the audio all at once

uploaded= client.files.upload(file=VIDEO_FILE)
#LET THE VIDEO BE PROCESS
while uploaded.state.name=="PROCESSING":
    time.sleep(2)
    uploaded=client.files.get(name=uploaded.name)
    
direct_response = client.models.generate_content(
    model=MODEL,
    contents=[
        uploaded,                               # <- the ACTUAL video, every pixel
        "Based on this video, describe in 3 short lines what happens: "
        "how many scenes, their colours, and where the motion/action is.",
    ],
)
print(direct_response.text)