from fastapi import FastAPI
import torch
import os
import pretty_midi
import numpy as np
import json
import torch.nn.functional as F
from fastapi.responses import FileResponse

# --- CONFIG ---
MODEL_PATH = "models/music_rnn_epoch_20.pth"
SOUNDFONT_PATH = "soundfonts/GeneralUser-GS.sf2"
PROCESSED_DATA_DIR = "data/processed"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="🎵 Maestro AI Music Generator")

# --- MODEL DEFINITION (SAME AS TRAINING) ---
class MusicRNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=3, dropout=0.3):
        super(MusicRNN, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.fc(out)
        return out, hidden

# --- LOAD MODEL ---
with open(os.path.join(PROCESSED_DATA_DIR, 'vocab.json'), 'r') as f:
    vocab = json.load(f)
token_to_int = vocab['token_to_int']
int_to_token = vocab['int_to_token']
vocab_size = len(token_to_int)

model = MusicRNN(vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --- SEQUENCE GENERATION FUNCTION ---
def generate_sequence(start_tokens, max_len=512):
    seq = [token_to_int[t] for t in start_tokens if t in token_to_int]
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.LongTensor([seq])
            output, _ = model(input_tensor)
            next_token = torch.argmax(output[0, -1, :]).item()
            seq.append(next_token)
    return [int_to_token[str(i)] for i in seq]

@app.get("/")
def root():
    return {"message": "Welcome to Maestro AI 🎵"}

@app.get("/generate/{raga}/{tala}")
def generate_music(raga: str, tala: str):
    events = generate_sequence([f"raga_{raga}", f"tala_{tala}"])
    midi_path = os.path.join(OUTPUT_DIR, f"ai_{raga}_{tala}.mid")

    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    current_time = 0
    for event in events:
        if "note_on_" in event:
            pitch = int(event.split("_")[-1])
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=current_time, end=current_time + 0.3))
            current_time += 0.3
    midi.instruments.append(inst)
    midi.write(midi_path)

    return FileResponse(midi_path, media_type="audio/midi", filename=f"ai_{raga}_{tala}.mid")
