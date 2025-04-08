from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_split_pdf(file_path):
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the document into chunks with custom separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return documents, chunks
