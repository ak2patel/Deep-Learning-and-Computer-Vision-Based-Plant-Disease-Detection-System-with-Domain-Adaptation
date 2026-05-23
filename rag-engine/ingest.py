import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
DB_FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

def create_vector_db():
    # Step 1: Load PDFs from the documents folder
    print(f"[*] Scanning for PDFs in: {DOCS_DIR}")
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print(f"[ERROR] No documents found in '{DOCS_DIR}'. Please add some agricultural PDFs first.")
        return
        
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    raw_documents = loader.load()
    print(f"[OK] Loaded {len(raw_documents)} pages from PDFs.")
    
    # Step 2: Split text into manageable chunks
    # Chunking strategy: 
    # - chunk_size=1000 characters (approx. 150-200 words per chunk)
    # - chunk_overlap=200 characters (ensures context isn't split across borders)
    print("[*] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)
    print(f"[OK] Created {len(documents)} text chunks.")
    
    # Step 3: Initialize the embedding model
    # We use all-MiniLM-L6-v2 which runs completely offline/locally on CPU.
    # It converts text sentences into 384-dimensional dense vectors.
    print("[*] Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Step 4: Build FAISS Vector Store and save it locally
    print("[*] Generating embeddings and building FAISS index (this might take a moment)...")
    db = FAISS.from_documents(documents, embeddings)
    
    print(f"[*] Saving FAISS database to: {DB_FAISS_PATH}")
    db.save_local(DB_FAISS_PATH)
    print("[SUCCESS] FAISS vector database created and saved successfully!")

if __name__ == "__main__":
    create_vector_db()
