from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from pydantic import BaseModel

app = FastAPI()

class EmbedRequest(BaseModel):
    inputs: List[str]

tokenizer = AutoTokenizer.from_pretrained("Omartificial-Intelligence-Space/GATE-AraBert-v1")
model = AutoModel.from_pretrained("Omartificial-Intelligence-Space/GATE-AraBert-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/embed")
async def embed(request: EmbedRequest):
    sentences = request.inputs

    if not sentences:
        return {"error": "No input sentences provided."}
    
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    token_embeddings = model_output[0]
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask

    return {"embeddings": sentence_embeddings.tolist()}