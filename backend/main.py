import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model.sentiment_model import SentimentModel
from data_preprocessing import preprocess_text
from transformers import AutoTokenizer
from pydantic import BaseModel


app = FastAPI()

class TextData(BaseModel):
    text: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokeniser = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = SentimentModel(tokeniser.vocab_size, num_classes=3)
model.load_state_dict(torch.load('models/model1.pth', map_location=device))
model = model.to(device)
model.eval()

map_sentiment = {0: 'Negative', 1: 'Neutral', 2: 'Positive' }


@app.post("/")
async def prediction(data: TextData):
    try:
        text = preprocess_text(data.text)
        encoded_text = tokeniser(text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(encoded_text['input_ids'])
            probs = nn.functional.softmax(logits, dim=-1)
            predicted_sentiment = probs.argmax(dim=-1).item()
        
        probs = [round(prob * 100, 2) for prob in probs.squeeze(0).tolist()]
        return {
            "sentiment": map_sentiment[predicted_sentiment],
            "probabilities": probs,
        }
    except Exception as e:
        return {"message": f"An error has occurred: {str(e)}"}
