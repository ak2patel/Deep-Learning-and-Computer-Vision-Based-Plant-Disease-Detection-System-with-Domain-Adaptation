import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables (like GROQ_API_KEY)
# Check current directory first, then search in parent directory (workspace root)
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    parent_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(parent_env_path):
        load_dotenv(parent_env_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

def get_rag_chain():
    # 1. Check for Groq API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not found in environment or .env file.")
        print("Please set your GROQ_API_KEY. You can get one for free at https://console.groq.com/")
        sys.exit(1)

    # 2. Load the embeddings model (must match the one used during ingestion)
    print("[*] Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 3. Load the local FAISS database
    print(f"[*] Loading FAISS index from: {DB_FAISS_PATH}")
    if not os.path.exists(DB_FAISS_PATH):
        print(f"[ERROR] FAISS index not found at '{DB_FAISS_PATH}'. Please run 'python ingest.py' first.")
        sys.exit(1)
        
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 matching chunks
    
    # 4. Initialize Groq Chat LLM
    print("[*] Connecting to Groq (Llama-3.1)...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2, # Low temperature for factual, grounded responses
        api_key=api_key
    )

    # 5. Define prompt template enforcing grounding
    system_prompt = (
        "You are an expert AI agricultural assistant. "
        "Answer the user's question regarding plant diseases, treatments, or prevention using ONLY the context provided below. "
        "Do not invent information. If the context does not contain the answer, say "
        "'I cannot find treatment details for this in the provided agricultural manuals.'\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Create the chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain, retriever

def ask_question(query: str, rag_chain, retriever):
    print(f"\n[QUERY] {query}")
    print("-" * 50)
    
    # Retrieve & show the documents for learning/inspection
    print("[*] Retrieving relevant chunks from vector database...")
    relevant_docs = retriever.invoke(query)
    for i, doc in enumerate(relevant_docs, 1):
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 'unknown')
        print(f"  -> Chunk {i} from [Source: {source}, Page: {page}]")
        # Print a snippet of the chunk
        snippet = doc.page_content.replace('\n', ' ')[:100]
        print(f"     Text snippet: \"{snippet}...\"")
        
    print("[*] Synthesizing answer using Llama-3...")
    response = rag_chain.invoke({"input": query})
    
    print("\n[ANSWER]")
    print(response["answer"])
    print("-" * 50)

def main():
    rag_chain, retriever = get_rag_chain()
    
    # Check if user passed query as a command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        ask_question(query, rag_chain, retriever)
    else:
        # Otherwise run interactive chat loop
        print("\n=== AgriVision RAG Chatbot ===")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                query = input("\nAsk about plant diseases: ").strip()
                if not query:
                    continue
                if query.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                ask_question(query, rag_chain, retriever)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
