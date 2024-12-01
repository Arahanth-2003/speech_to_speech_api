import os
import random
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from googletrans import Translator
from gtts import gTTS
from fastapi.responses import FileResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

app = FastAPI()

# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Initialize the Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")  # Load API key securely
)

def remove_file(file_path: str):
    """Background task to remove the file after sending the response."""
    try:
        os.remove(file_path)
        print(f"File {file_path} removed successfully.")
    except Exception as e:
        print(f"Error while deleting file {file_path}: {e}")

@app.post("/translate/")
async def generate_translated_audio(audio_url: str, lang: str, background_tasks: BackgroundTasks) -> FileResponse:
    try:
        # Download the audio file from the URL
        response = requests.get(audio_url)
        response.raise_for_status()

        # Save the audio file temporarily
        uploaded_file = f"uploaded_audio_{random.randint(1000,9999)}.mp3"
        with open(uploaded_file, "wb") as buffer:
            buffer.write(response.content)

        # Perform speech-to-text using Groq Whisper API
        with open(uploaded_file, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(uploaded_file, file.read()),
                model="whisper-large-v3",
            )
        speech_to_text = transcription.text

        # Translate the text
        translator = Translator()
        translation = translator.translate(speech_to_text, dest=lang)
        translated_text = translation.text

        # Generate translated audio using gTTS
        output_file = f"translated_audio_{random.randint(1000,9999)}.mp3"
        tts = gTTS(text=translated_text, lang=lang)
        tts.save(output_file)

        # Add background task to remove the output file after sending the response
        background_tasks.add_task(remove_file, output_file)

        # Clean up the uploaded file immediately
        os.remove(uploaded_file)

        # Return the audio file directly to the client
        return FileResponse(output_file, media_type="audio/mp3", filename=os.path.basename(output_file))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
