from fastapi import FastAPI
from fastapi.responses import FileResponse
from maestro import generate_full_piece
import os

app = FastAPI()

MODEL_PATH = "models/music_rnn_epoch_20.pth"
SOUNDFONT_PATH = "soundfonts/GeneralUser-GS.sf2"
os.makedirs("output/audio", exist_ok=True)

@app.get("/")
def home():
    return {"message": "🎶 Music Raga Generator Backend is running!"}

@app.get("/generate")
def generate(raga: str, tala: str):
    """Generate audio for given raga and tala."""
    audio_path = generate_full_piece(MODEL_PATH, raga, tala)
    return FileResponse(audio_path, media_type="audio/wav",
                        filename=os.path.basename(audio_path))
