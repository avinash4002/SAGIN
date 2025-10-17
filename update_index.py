# update_index.py

# ==============================================================================
#  FAISS INDEX UPDATER
# ==============================================================================
#  This script adds new documents to an existing FAISS vector store without
#  rebuilding the entire index from scratch.
#
#  Workflow:
#  1. Place new PDF/CSV files into the 'NewKnowledgeFiles' folder.
#  2. Run this script: `python update_index.py`
#  3. The script will load the existing index, process ONLY the new files,
#     merge them, and save the updated index.
#  4. Move the processed files out of 'NewKnowledgeFiles' to avoid re-adding.
#

# --- Core Libraries ---
import os
import sys

# --- LangChain Libraries ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
class Config:
    """Configuration class for paths and model names."""
    # This must be the folder with your existing FAISS index
    VECTOR_STORE_PATH = "faiss_index_groq"
    
    # This folder should ONLY contain the new files to be added
    NEW_DOCS_PATH = "new_knowldegebase/"
    
    # IMPORTANT: This MUST be the same embedding model used to create the original index
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ==============================================================================
#  MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Starting FAISS Index Update Process...")

    # --- 1. Sanity Checks ---
    if not os.path.exists(Config.VECTOR_STORE_PATH):
        print(f"❌ ERROR: Existing vector store not found at '{Config.VECTOR_STORE_PATH}'.")
        print("   Please run the main query script first to create the initial index.")
        sys.exit(1)

    if not os.path.exists(Config.NEW_DOCS_PATH) or not os.listdir(Config.NEW_DOCS_PATH):
        print(f"✅ No new documents found in '{Config.NEW_DOCS_PATH}'. Index is already up-to-date.")
        sys.exit(0)

    # --- 2. Load the New Documents ---
    print(f"  -> Loading new documents from '{Config.NEW_DOCS_PATH}'...")
    try:
        # Load any new PDFs
        pdf_loader = PyPDFDirectoryLoader(Config.NEW_DOCS_PATH)
        new_pdf_docs = pdf_loader.load()
        
        # Load any new CSVs
        csv_files = [os.path.join(Config.NEW_DOCS_PATH, f) for f in os.listdir(Config.NEW_DOCS_PATH) if f.endswith('.csv')]
        new_csv_docs = [doc for file in csv_files for doc in CSVLoader(file_path=file).load()]

        new_docs = new_pdf_docs + new_csv_docs
        if not new_docs:
            print("✅ No processable new documents found. Exiting.")
            sys.exit(0)
            
        print(f"  -> Found {len(new_docs)} new document pages/rows to add.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load new documents. Details: {e}")
        sys.exit(1)
        
    # --- 3. Split the New Documents ---
    print("  -> Splitting new documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    new_splits = text_splitter.split_documents(new_docs)

    # --- 4. Load Existing Vector Store and Add New Documents ---
    try:
        print("  -> Loading existing FAISS index...")
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(Config.VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

        print(f"  -> Merging {len(new_splits)} new chunks into the index...")
        # This is the key step: adding new documents to the loaded index
        vectorstore.add_documents(new_splits)

    except Exception as e:
        print(f"❌ ERROR: Failed during index loading or merging. Details: {e}")
        sys.exit(1)

    # --- 5. Save the Updated Vector Store ---
    print(f"  -> 💾 Saving the updated index back to '{Config.VECTOR_STORE_PATH}'...")
    vectorstore.save_local(Config.VECTOR_STORE_PATH)

    print("\n🎉 SUCCESS: The FAISS index has been updated with the new documents.")
    print("   IMPORTANT: Remember to move the processed files out of the 'NewKnowledgeFiles' folder.")