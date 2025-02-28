import streamlit as st
import os
from functools import lru_cache
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from concurrent.futures import ThreadPoolExecutor

# Constants
PDF_FOLDER = "cancer_documents/"  # Folder containing cancer-related PDFs
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100  # Reduced overlap
MAX_DOCS_RETURNED = 3  # Limit number of docs for context

# Initialize models with session state to avoid reloading
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")

if "language_model" not in st.session_state:
    st.session_state.language_model = OllamaLLM(
        model="deepseek-r1:1.5b",
        num_predict=512,  # Limit token generation
        temperature=0.1   # Lower temperature for more focused responses
    )

if "document_store" not in st.session_state:
    st.session_state.document_store = InMemoryVectorStore(st.session_state.embedding_model)
#
# Enhanced prompt template for better responses
PROMPT_TEMPLATE = """
You are an AI specialized in answering questions about cancer-related research.
Answer the query based on ONLY the provided context. Be concise and factual.
If the context doesn't contain relevant information, state "I don't have enough information about that in my current database."

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# Cache mechanism for PDF loading
@lru_cache(maxsize=32)
def get_pdf_path(filename):
    return os.path.join(PDF_FOLDER, filename)

def process_single_pdf(filename):
    """Process a single PDF file and return its chunks"""
    if not filename.endswith(".pdf"):
        return []
    
    try:
        file_path = get_pdf_path(filename)
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]  # Optimized separators
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add source metadata for better traceability
        for chunk in chunks:
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = filename
        
        return chunks
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        return []

def load_and_process_pdfs():
    """Loads PDFs using parallel processing"""
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        st.info(f"Created folder {PDF_FOLDER}. Please add PDF documents.")
        return
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    
    if not pdf_files:
        st.info("No PDF files found in the documents folder.")
        return
    
    # Show loading progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_documents = []
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(pdf_files))) as executor:
        futures = {executor.submit(process_single_pdf, pdf_file): pdf_file for pdf_file in pdf_files}
        
        for i, future in enumerate(futures):
            chunks = future.result()
            all_documents.extend(chunks)
            progress_value = (i + 1) / len(pdf_files)
            progress_bar.progress(progress_value)
            status_text.text(f"Processed {i+1}/{len(pdf_files)} files")
    
    if all_documents:
        st.session_state.document_store.add_documents(all_documents)
        status_text.text(f"Indexed {len(all_documents)} chunks from {len(pdf_files)} files")
    else:
        status_text.text("No content was extracted from the PDFs.")
    
    progress_bar.empty()

# Load documents if not already loaded
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# Initialize the cache
@st.cache_resource
def get_conversation_chain():
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | st.session_state.language_model

# Streamlit UI
st.title("🩺 Cancer Research Chatbot")
st.markdown("### Ask me questions about cancer research.")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    if st.button("Reload Documents"):
        st.session_state.docs_loaded = False
        st.session_state.document_store = InMemoryVectorStore(st.session_state.embedding_model)
    
    max_docs = st.slider("Max documents for context", 1, 10, MAX_DOCS_RETURNED)
    
    st.markdown("---")
    st.markdown("### Document Statistics")
    if hasattr(st.session_state.document_store, "_collection"):
        doc_count = len(st.session_state.document_store._collection)
        st.write(f"Total chunks indexed: {doc_count}")

# Load documents if needed
if not st.session_state.docs_loaded:
    load_and_process_pdfs()
    st.session_state.docs_loaded = True

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍⚕️" if message["role"] == "assistant" else None):
        st.write(message["content"])

# User input
user_input = st.chat_input("Enter your question about cancer...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.spinner("Analyzing cancer research papers..."):
        try:
            # Get relevant documents
            relevant_docs = st.session_state.document_store.similarity_search(
                user_input, 
                k=max_docs
            )
            
            # Prepare context
            if relevant_docs:
                context_texts = []
                for i, doc in enumerate(relevant_docs):
                    source = doc.metadata.get("source", "Unknown")
                    context_texts.append(f"[Document {i+1} from {source}]\n{doc.page_content}")
                
                context_text = "\n\n".join(context_texts)
            else:
                context_text = "No relevant documents found."
            
            # Get response chain
            response_chain = get_conversation_chain()
            
            # Generate response
            ai_response = response_chain.invoke({
                "user_query": user_input, 
                "document_context": context_text
            })
            
            # Add sources information if available
            if relevant_docs:
                sources = set(doc.metadata.get("source", "Unknown") for doc in relevant_docs)
                source_text = "\n\n*Sources: " + ", ".join(sources) + "*"
                ai_response += source_text
                
        except Exception as e:
            ai_response = f"Sorry, I encountered an error: {str(e)}"
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        st.write(ai_response)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})