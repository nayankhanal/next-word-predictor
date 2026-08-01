import json

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import SimpleRNN
from preprocess import text_to_indices

app = FastAPI(title="Next Word QA Prediction API")

# allow Streamlit (running on a different port/origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load vocab + model once, at startup ----
with open("vocab.json", "r") as f:
    vocab = json.load(f)

idx_to_word = {idx: word for word, idx in vocab.items()}

model = SimpleRNN(len(vocab))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()


class Question(BaseModel):
    question: str
    threshold: float = 0.5


class Prediction(BaseModel):
    answer: str
    confidence: float


@app.get("/")
def health_check():
    return {"status": "ok", "vocab_size": len(vocab)}


@app.post("/predict", response_model=Prediction)
def predict(payload: Question):
    numerical_question = text_to_indices(payload.question, vocab)
    question_tensor = torch.tensor(numerical_question).unsqueeze(0)

    with torch.no_grad():
        output = model(question_tensor)
        probs = F.softmax(output, dim=1)
        value, index = torch.max(probs, dim=1)

    confidence = value.item()
    predicted_word = idx_to_word[index.item()]

    if confidence < payload.threshold:
        return Prediction(answer="I don't know", confidence=confidence)

    return Prediction(answer=predicted_word, confidence=confidence)