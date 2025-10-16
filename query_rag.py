# query_rag_groq.py (High-Speed RAG with Groq and Local Embeddings)

# ==============================================================================
#  CUSTOM RAG QUERY TOOL POWERED BY GROQ
# ==============================================================================
#  This script uses the high-speed Groq API for LLM inference and a local
#  Hugging Face model for embeddings, making it fast and free to run.
#
#  Prerequisites:
#  1. A .env file with your GROQ_API_KEY.
#  2. All libraries from requirements.txt installed.
#

# --- Core Libraries ---
import os
import sys
from dotenv import load_dotenv

# --- LangChain Libraries ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# ==============================================================================
#  STEP 1: CONFIGURATION
# ==============================================================================
class Config:
    """Configuration class for all paths and model names."""
    PDF_KNOWLEDGE_BASE_PATH = "knowledge_base/"
    CSV_KNOWLEDGE_BASE_PATH = "csvs/"
    VECTOR_STORE_PATH = "faiss_index_groq"  # Use a new folder for the Groq index
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # A fast, popular, and effective local model
    LLM_MODEL = "llama-3.3-70b-versatile"            # A great, fast model available on Groq

# ==============================================================================
#  STEP 2: BUILD OR LOAD THE RAG PIPELINE
# ==============================================================================
def get_rag_pipeline():
    """
    Builds the RAG pipeline using Groq and local embeddings.
    Saves and loads the vector store to avoid re-processing.
    """
    print("🧠 Initializing the RAG pipeline with Groq and local embeddings...")

    # --- Use local Hugging Face model for embeddings ---
    # This runs on your CPU/GPU, not via an API call.
    print(f"  -> Loading local embedding model: '{Config.EMBEDDING_MODEL}'")
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

    if os.path.exists(Config.VECTOR_STORE_PATH):
        print(f"  -> Loading existing vector store from '{Config.VECTOR_STORE_PATH}'...")
        vectorstore = FAISS.load_local(Config.VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("  ✅ Vector store loaded successfully.")
    else:
        print("  -> No existing vector store found. Building a new one...")
        try:
            pdf_loader = PyPDFDirectoryLoader(Config.PDF_KNOWLEDGE_BASE_PATH)
            pdf_docs = pdf_loader.load()
            csv_files = [os.path.join(Config.CSV_KNOWLEDGE_BASE_PATH, f) for f in os.listdir(Config.CSV_KNOWLEDGE_BASE_PATH) if f.endswith('.csv')]
            csv_docs = [doc for file in csv_files for doc in CSVLoader(file_path=file).load()]
            all_docs = pdf_docs + csv_docs
        except Exception as e:
            print(f"❌ ERROR loading documents: {e}")
            return None

        print(f"    -> Splitting {len(all_docs)} documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_documents(all_docs)

        print(f"    -> Creating embeddings and FAISS index locally... (This might be slow the first time)")
        vectorstore = FAISS.from_documents(splits, embeddings)

        print(f"    -> 💾 Saving new vector store to '{Config.VECTOR_STORE_PATH}'...")
        vectorstore.save_local(Config.VECTOR_STORE_PATH)

    # --- Define LLM and Prompt ---
    # Use the Groq API for high-speed chat
    llm = ChatGroq(temperature=0.2, model_name=Config.LLM_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt_template = """
    You are an intelligent assistant for a SAGIN researcher.
    Answer the user's question based ONLY on the provided context. If the context doesn't contain the answer, say so.
    Provide a clear and helpful answer.

    CONTEXT FROM KNOWLEDGE BASE:
    {context}

    USER'S QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # --- Create the RAG Chain ---
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("\n✅ RAG pipeline is fully built and ready to use.")
    return rag_chain

# ==============================================================================
#  STEP 3: INTERACTIVE QUERY SESSION
# ==============================================================================
def start_interactive_session(rag_chain):
    """Starts an interactive loop to ask questions to the RAG chain."""
    print("\n" + "="*50)
    print("🚀 SAGIN Knowledge Base Assistant (Groq Edition) is ready.")
    print("   Type your question and press Enter.")
    print("   Type 'exit' or 'quit' to end the session.")
    print("="*50)
    while True:
        try:
            query = input("\n> ")
            if query.lower().strip() in ['exit', 'quit']:
                print("👋 Session ended. Goodbye!")
                break
            if not query: continue

            print("\n⚡️ Thinking (using Groq)...")
            response = rag_chain.invoke(query)
            
            print("\n--- Answer ---")
            print(response)
            print("--------------")
        except KeyboardInterrupt:
            print("\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

# ==============================================================================
#  MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("❌ ERROR: GROQ_API_KEY not found in .env file. Please create one.")
        sys.exit(1)

    rag_chain_instance = get_rag_pipeline()
    if rag_chain_instance:
        start_interactive_session(rag_chain_instance)