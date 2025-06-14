import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma

_, _, file_names = next(os.walk("data/txt"))

print("Loading documents...")
documents = []
for file_name in file_names:
    file_path = os.path.join("data/txt", file_name)
    with open(file_path, "r") as f:
        text = f.read()
        documents.append(Document(page_content=text))

print(f"Loaded {len(documents)} documents.")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1", model_kwargs={"device": "cuda"})
splitter = SemanticChunker(embeddings=embeddings)

print("Chunking documents...")
chunks = splitter.split_documents(documents=documents)

print(f"Number of chunks: {len(chunks)}")

print("Loading embeddings to ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings,
    persist_directory="./chroma_storage")

print("Successfully loaded embeddings.")