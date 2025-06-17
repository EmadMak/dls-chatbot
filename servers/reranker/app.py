from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
from pydantic import BaseModel

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("Omartificial-Intelligence-Space/ARA-Reranker-V1")
model = AutoModelForSequenceClassification.from_pretrained("Omartificial-Intelligence-Space/ARA-Reranker-V1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

@app.post("/rerank")
async def rerank(request: RerankRequest):

    if not request.query or not request.documents:
        return {"error": "Query and documents must be provided."}
    
    pairs = [[request.query, doc] for doc in request.documents]

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    return {"scores": scores.tolist()}