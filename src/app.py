from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from onnx_wrappers import ONNXEmbeddings, ONNXReranker

embeddings = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1", model_kwargs={"device": "cpu"})
#embeddings = ONNXEmbeddings(model_path="onnx/GATE-AraBert-v1-onnx")

vectorstore = Chroma(
    persist_directory="./chroma_storage",
    embedding_function=embeddings
)

vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

reranker = HuggingFaceCrossEncoder(model_name="Omartificial-Intelligence-Space/ARA-Reranker-V1", model_kwargs={"device": "cpu"})
compressor = CrossEncoderReranker(model=reranker, top_n=5)
#compressor = ONNXReranker(model_path="onnx/ARA-Reranker-V1-onnx", top_n=10)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vectorstore_retriever
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    temperature=0
)

template = """
<|system|>
انت مساعد ذكي تجيب على الاسئلة باللغة العربية و بشكل واضح
context: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template=template)
output_parser = StrOutputParser()

qa_chain = (
    {"context": compression_retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

query = input("Enter your query: ")
result = qa_chain.invoke(query)
print(result)
