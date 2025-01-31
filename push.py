# Install required packages
# pip install PyPDF2 python-dotenv langchain pinecone-client sentence-transformers

import os
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize embedding model
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# 1. PDF Processing
def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# 2. Text Splitting
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    return text_splitter.split_text(text)

# 3. Initialize Pinecone
def initialize_pinecone(index_name):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if not exists (384 is dimension for all-MiniLM-L6-v2)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    return pc.Index(index_name)

# 4. Process and Store Embeddings
def process_and_store(pdf_path, index_name):
    # Extract text
    raw_text = extract_pdf_text(pdf_path)
    
    # Split text
    chunks = split_text(raw_text)
    
    # Initialize Pinecone index
    index = initialize_pinecone(index_name)
    
    # Create embeddings and store in Pinecone
    batch_size = 100
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:batch_end]
        
        # Generate embeddings
        embeds = EMBEDDING_MODEL.encode(batch_chunks).tolist()
        
        # Prepare vectors
        vectors = [(f"{i+idx}", embed, {"text": chunk}) for idx, (chunk, embed) in enumerate(zip(batch_chunks, embeds))]
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        print(f"Processed {batch_end}/{total_chunks} chunks")

    print(f"Successfully uploaded {total_chunks} chunks to Pinecone")

# Run the processing
if __name__ == "__main__":
    PDF_PATH = r"C:\Users\sharo\Desktop\SEMProjects\project phase\finLLM\data\finance_encyclopedia.pdf"
    INDEX_NAME = "financial-encyclopedia"
    
    process_and_store(PDF_PATH, INDEX_NAME)
