"""
Audio Piepline

Direct Way -> let the model heaar the sound at once. 

Old School way-> python library pull the real values from the audio and we send those to the llm

wave + numpy gives numbers of audio (duration,loudness,silence)
CMU Sphinx gives a text and transcript of audio
we mix numbers and text to give it to LLM

LLM doesnt hear the audio but read the transcript and measurements of the audio

wave -> read .wav file
numpy -> calculates math on raw samples
CMU Sphinx -> it gives offline speech. it gives messy text and llm cleans it up
"""

import os
import wave 
import json

from dotenv import load_dotenv
from google import genai
import numpy as np

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.1-flash-lite"      

AUDIO_FILE = "sample_audio.wav"

#Step 1 -> gets measurements from wav file
print("STEP 1: wave + numpy -> signal measurements (no AI here)")

with wave.open(AUDIO_FILE,"rb") as wf:
    channels=wf.getnchannels() #1 for mono , 2 for stereo
    samp_width=wf.getsampwidth() #bytes per sample( 16 bit audio)
    sample_rate=wf.getframerate() #get samples per second
    n_frames=wf.getnframes() #get total frames
    raw_bytes=wf.readframes(n_frames) #it reads the frames into unreadable bytes
    
#gets the duration of audio
duration= n_frames/float(sample_rate)

#Turning raw byte into numpy array
dtype= {1: np.uint8, 2: np.uint16, 4: np.uint32}.get(samp_width,np.int16) 
#turns the audio into numbers:
samples= np.frombuffer(raw_bytes,dtype=dtype).astype(np.float32)

#If channel is stereo we convert to mono (Becayse mono is easier to decode for LLM)
if channels>1:
    samples=samples.reshape(-1,channels).mean(axis=1)  #we take average in the array to have one number to represent mono


#Scale the sample from -1 to 1 range because normally it is very big 
if dtype==np.uint8:
    samples= (samples-128)/128.0
    
else:
    samples= samples/float(2**(8*samp_width-1))  
    
#computing loudness peak and silence through formulaes
loudness=float(np.sqrt(np.mean(samples**2))) #ROOT MEAN SQUARE (0 to 1)
peak= float(np.max(np.abs(samples))) #gets maximum in absolute form
silence_ratio = float(np.mean(np.abs(samples) < 0.02)) #silence is considered when it is lower than 0.02
#we take MEAN of true,false array whteer it is lesser or greater than 0.02 . Then the mean tells how much % it is silent

#Now we make energy overtime array after chunking sample into 10 pieces
chunks=np.array_split(samples,10)
energy_over_time= [int(np.sqrt(np.mean(c**2)) *100) for c in chunks if c.size]  #store RMS of loudness if it exists in the chunks

#storing all the audio detail in dictioanry
measurements = {
    "duration_seconds": round(duration, 1),
    "sample_rate_hz": sample_rate,
    "channels": channels,
    "loudness_0_to_100": int(loudness * 100),
    "peak_0_to_100": int(peak * 100),
    "silence_percent": int(silence_ratio * 100),
    "energy_over_time": energy_over_time,
}
print(json.dumps(measurements, indent=2))

#STEP 2-> Creating transcript OFFLINE
print("STEP 2: CMU Sphinx (offline, classic) -> transcript")

transcript=""

try:
    import speech_recognition  as sr
    
    recognizer=sr.Recognizer()
    #opening file in audio and reading it then storing it in transcript
    
    with sr.AudioFile(AUDIO_FILE) as source:
        audio_data = recognizer.record(source)
    transcript=recognizer.recognize_speech(audio_data)
    
    print(transcript)
    
except Exception as e:
    # No Sphinx installed, or it failed — that's fine, we keep going with
    # just the numbers from STEP 1. (Install: pip install SpeechRecognition pocketsphinx)
    print(f"(Sphinx not available: {e})")
    print("Skipping transcript — the LLM will work from the measurements only.")
    
#Step 3-> LLM interprets the answer
print("STEP 3: LLM interprets the values (it never hears the audio)")

prompt = f"""You are given ONLY classic audio measurements and a rough,
possibly-wrong offline transcript of an audio clip. You cannot hear the audio.

Signal measurements:
{json.dumps(measurements, indent=2)}

Rough offline transcript (may contain mistakes, may be empty):
"{transcript}"

Questions:
Based only on this, answer in 2 short lines:
1) What is this audio probably about?
2) What is the likely mood / energy of the speaker?
"""

response=client.models.generate_content(
    model=MODEL,
    contents=[prompt]
)

print(response.text)