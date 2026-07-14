"""
Here we use whisper API that is modern and gives acurate answer

we put audio into whisper that gives text/transcript of the audio
we put that transcript into LLM for answere generation


"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

STT_MODEL = "whisper-large-v3"          # the EARS:  speech -> text
LLM_MODEL = "llama-3.3-70b-versatile"   # the BRAIN: text  -> answer

AUDIO_FILE = "sample_audio.wav"


#Put audio into whisper
print("STEP 1: Whisper transcribes the audio -> text")

with open(AUDIO_FILE,"rb") as f:
    transcription = client.audio.transcriptions.create(
        file=(AUDIO_FILE,f.read()),
        model=STT_MODEL,   
    )
    
print(transcription.text)

#Put text into LLm
print("STEP 2: LLM understands the text (never hears the audio)")

completion = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": (
                "Here is a transcript of an audio clip:\n\n"
                f"{transcription.text}\n\n"
                "What is it about? What is the mood of the speaker? "
                "Answer in 2 short lines."
            ),
        }
    ],
)
print(completion.choices[0].message.content)