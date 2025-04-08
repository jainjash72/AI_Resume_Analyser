from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def create_vector_store(chunks):
    # Build the FAISS vector store from the text chunks
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_store
