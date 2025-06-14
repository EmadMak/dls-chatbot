import torch
import numpy as np
from typing import List, Tuple, Any
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from pydantic import Field

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
from transformers import AutoTokenizer

class ONNXEmbeddings(Embeddings):
    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        print(f"Loading ONNX embedding model from: {model_path}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            provider="CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        )
        print("ONNX embedding model loaded successfully.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(
            model_output=model_output,
            attention_mask=encoded_input["attention_mask"]
        )

        return sentence_embeddings.cpu().numpy().tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class ONNXReranker(BaseDocumentCompressor):
    """
    LangChain-compatible document compressor that uses an ONNX cross-encoder
    model for reranking. Correctly handles Pydantic initialization.
    """
    top_n: int
    model: Any
    tokenizer: Any
    device: Any

    class Config:
        arbitrary_types_allowed = True
        fields = {
            'model': {'exclude': True},
            'tokenizer': {'exclude': True},
            'device': {'exclude': True},
        }

    def __init__(self, model_path: str, top_n: int = 10, device: str = "cpu", **kwargs: Any):
        """Initializes the ONNX-based reranker."""
        # Step 1: Initialize objects
        print(f"Loading ONNX reranker model from: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
            
        device_obj = torch.device(device)
        tokenizer_obj = AutoTokenizer.from_pretrained(model_path)
        model_obj = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            provider="CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider",
        )
        print("ONNX reranker model loaded successfully.")

        # Step 2: Assemble data for Pydantic validation
        init_data = {
            "top_n": top_n,
            "device": device_obj,
            "tokenizer": tokenizer_obj,
            "model": model_obj,
            **kwargs,
        }

        # Step 3: Call the parent constructor
        super().__init__(**init_data)

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks = None,
    ) -> List[Document]:
        if not documents:
            return []

        doc_texts = [doc.page_content for doc in documents]
        pairs: List[List[str]] = [[query, doc_text] for doc_text in doc_texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            scores = self.model(**inputs).logits.sigmoid().squeeze().cpu().numpy()

        if scores.ndim == 0:
            scores = np.array([scores])

        scored_docs: List[Tuple[Document, float]] = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:self.top_n]]