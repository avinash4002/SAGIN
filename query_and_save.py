# query_and_save.py

# ==============================================================================
#  CUSTOM RAG QUERY TOOL WITH MARKDOWN EXPORT
# ==============================================================================
#  This script uses the Groq API and local embeddings to answer questions.
#  After each response, it saves the query and its answer to a timestamped
#  .md file in the 'QueryResponses' folder.
#

# --- Core Libraries ---
import os
import sys
import datetime
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
    VECTOR_STORE_PATH = "faiss_index_groq"
    OUTPUT_PATH = "QueryResponses/"  # Folder to save the .md files
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "openai/gpt-oss-120b"

# ==============================================================================
#  STEP 2: BUILD OR LOAD THE RAG PIPELINE
# ==============================================================================
def get_rag_pipeline():
    """Builds or loads the RAG pipeline from the saved FAISS index."""
    print("🧠 Initializing the RAG pipeline with Groq and local embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

    if not os.path.exists(Config.VECTOR_STORE_PATH):
        print(f"❌ ERROR: Vector store not found at '{Config.VECTOR_STORE_PATH}'.")
        print("   Please run the script that builds the index first.")
        return None
        
    print(f"  -> Loading existing vector store from '{Config.VECTOR_STORE_PATH}'...")
    vectorstore = FAISS.load_local(Config.VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGroq(temperature=0.2, model_name=Config.LLM_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt_template = """
    You are an intelligent assistant for a SAGIN researcher.
    Answer the user's question based ONLY on the provided context.
    Format your answer clearly using Markdown.

    CONTEXT FROM KNOWLEDGE BASE:
    {context}

    USER'S QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    
    print("\n✅ RAG pipeline is fully built and ready to use.")
    return rag_chain

# ==============================================================================
#  STEP 3: HELPER FUNCTION TO SAVE RESPONSE
# ==============================================================================
def save_to_markdown(query, response):
    """Saves the query and response to a timestamped Markdown file."""
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        
        # Generate a unique filename using the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.md"
        filepath = os.path.join(Config.OUTPUT_PATH, filename)
        
        # Format the content for the Markdown file
        md_content = f"# Query\n\n`{query}`\n\n---\n\n# RAG Response\n\n{response}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        print(f"📄 Response saved to: {filepath}")
        
    except Exception as e:
        print(f"❌ ERROR: Could not save response to file. Details: {e}")

# ==============================================================================
#  STEP 4: INTERACTIVE QUERY SESSION
# ==============================================================================
def start_interactive_session(rag_chain):
    """Starts an interactive loop to ask questions and saves each response."""
    print("\n" + "="*50)
    print("🚀 SAGIN Knowledge Base Assistant (with Auto-Save)")
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
            
            # Save the response to a file
            save_to_markdown(query, response)

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