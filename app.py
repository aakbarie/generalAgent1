import re
import os
import pickle
import logging
import time
from datetime import datetime
from typing import List

import streamlit as st
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Updated imports for deprecated modules
from langchain_community.chat_models import ChatOllama  # Updated import
from langchain_community.document_loaders import PyPDFLoader  # Updated import

# Remaining imports from main langchain package
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Ensure NLTK data is downloaded
import nltk
nltk.download('punkt')

# Streamlit page configuration
st.set_page_config(page_title="Provider Manual Assistant", layout="wide")

# Custom CSS with Jeff Koons Moon Phases color palette
st.markdown(
    """
    <style>
    /* Color Palette */
    /* #212A31 - Dark Blue */
    /* #2E3944 - Slate Gray */
    /* #124E66 - Teal */
    /* #748D92 - Light Slate Gray */
    /* #D3D9D4 - Light Gray */

    /* Main content area */
    .main {
        background-color: #D3D9D4; /* Light Gray */
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #212A31; /* Dark Blue */
    }
    [data-testid="stSidebar"] * {
        color: #D3D9D4; /* Light Gray text for contrast */
    }
    /* Button styling in the sidebar */
    [data-testid="stSidebar"] button {
        background-color: #124E66 !important; /* Teal */
        color: #D3D9D4 !important; /* Light Gray text */
    }
    /* Button styling in the main content */
    .main button {
        background-color: #D3D9D4 !important; /* Light Gray */
        color: #212A31 !important; /* Dark Blue text */
        border: 2px solid #124E66 !important; /* Teal border */
        border-radius: 5px; /* Optional: Add border radius for rounded corners */
    }
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #212A31; /* Dark Blue */
    }
    /* Text styling */
    p, div, span, label {
        color: #2E3944; /* Slate Gray */
    }
    /* Input field styling */
    .stTextInput > div > div > input {
        color: #2E3944; /* Slate Gray text */
        background-color: #FFFFFF; /* White background */
    }
    /* Radio button labels */
    .stRadio label {
        font-size: 16px;
        color: #2E3944; /* Slate Gray */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Persistent storage paths
SESSION_HISTORY_DIR = "saved_sessions"

# Ensure the directory for saved sessions exists
if not os.path.exists(SESSION_HISTORY_DIR):
    os.makedirs(SESSION_HISTORY_DIR)

def clean_text(text):
    """Cleans the input text by removing extra whitespace."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_to_disk(obj, path):
    """Saves an object to disk using pickle."""
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Saved object to {path}")
    except Exception as e:
        logging.error(f"Error saving object to {path}: {e}")

def load_from_disk(path):
    """Loads an object from disk using pickle."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logging.info(f"Loaded object from {path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {path}: {e}")
        return None

def vector_db_exists(VECTOR_DB_PATH, BM25_PATH, CHUNKS_PATH) -> bool:
    """Checks if the vector database files exist."""
    return os.path.exists(VECTOR_DB_PATH) and os.path.exists(BM25_PATH) and os.path.exists(CHUNKS_PATH)

def setup_retrieval_system(documents: List[Document], doc_id: str):
    """Sets up the retrieval system for the given document ID."""
    doc_dir = os.path.join("documents", doc_id)
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)

    VECTOR_DB_PATH = os.path.join(doc_dir, "vector_db.pkl")
    BM25_PATH = os.path.join(doc_dir, "bm25.pkl")
    CHUNKS_PATH = os.path.join(doc_dir, "chunks.pkl")

    if vector_db_exists(VECTOR_DB_PATH, BM25_PATH, CHUNKS_PATH):
        chunks = load_from_disk(CHUNKS_PATH)
        bm25 = load_from_disk(BM25_PATH)
        db = load_from_disk(VECTOR_DB_PATH)
        if chunks is None or bm25 is None or db is None:
            st.error("Failed to load retrieval system from disk.")
            return None, None, None
        st.info(f"Loaded vector database for '{doc_id}' from disk.")
    else:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=182)
            chunks = text_splitter.split_documents(documents)
            tokenized_segments = [word_tokenize(chunk.page_content.lower()) for chunk in chunks]
            bm25 = BM25Okapi(tokenized_segments)

            # Create FAISS vector store
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            db = FAISS.from_documents(chunks, embeddings)

            # Save to disk for future use
            save_to_disk(chunks, CHUNKS_PATH)
            save_to_disk(bm25, BM25_PATH)
            save_to_disk(db, VECTOR_DB_PATH)
            st.info(f"Created new vector database for '{doc_id}'.")
        except Exception as e:
            logging.error(f"Error setting up retrieval system for '{doc_id}': {e}")
            st.error("Failed to set up the retrieval system.")
            return None, None, None

    return chunks, bm25, db

@st.cache_resource
def setup_llm_model():
    """Initializes the LLM model."""
    try:
        return ChatOllama(model="llama3.2", temperature=0.7, top_k=10, top_p=0.95)
    except Exception as e:
        logging.error(f"Error setting up LLM model: {e}")
        st.error("Failed to initialize the LLM model.")
        return None

def combined_retrieve_multi(query, documents_data):
    """Combines BM25 and FAISS retrieval results from multiple documents."""
    bm25_results = []
    faiss_results = []

    for doc_id, data in documents_data.items():
        bm25 = data['bm25']
        chunks = data['chunks']
        db = data['db']

        # BM25 retrieval
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)
        top_k_indices = bm25_scores.argsort()[-5:][::-1]
        bm25_results.extend([chunks[i] for i in top_k_indices])

        # FAISS similarity search
        faiss_results.extend(db.similarity_search(query, k=5))

    # Combine results, avoiding duplicates
    seen = set()
    combined_results = []
    for doc in bm25_results + faiss_results:
        if doc.page_content not in seen:
            combined_results.append(doc)
            seen.add(doc.page_content)
    return combined_results

def chain_invoke_multi(question, llm, documents_data, conversation_history):
    """Generates an answer using the LLM, including context and conversation history, from multiple documents."""
    context_docs = combined_retrieve_multi(question, documents_data)
    context = "\n".join([doc.page_content for doc in context_docs])

    # Build the conversation history in reverse order (most recent first)
    conversation = ""
    for turn in reversed(conversation_history):
        conversation += f"User: {turn['question']}\nAssistant: {turn['response']}\n"

    # Update the prompt to include conversation history
    prompt = f"""
You are an assistant helping answer questions based on the provided context.

Context:
{context}

Conversation history:
{conversation}

Current question: {question}

Answer the question based ONLY on the provided context and conversation history.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
"""

    try:
        response = llm.invoke(prompt)
        return response.content, context_docs  # Return the response and the context docs
    except Exception as e:
        logging.error(f"Error invoking LLM: {e}")
        st.error("Failed to generate a response from the LLM.")
        return "", []

def save_session(conversation_history, session_name, doc_key):
    """Saves the conversation history to a file."""
    filename = os.path.join(SESSION_HISTORY_DIR, f"{session_name}_{doc_key}.pkl")
    try:
        with open(filename, "wb") as f:
            pickle.dump(conversation_history, f)
        st.success(f"Session '{session_name}' for documents '{doc_key}' saved successfully.")
    except Exception as e:
        logging.error(f"Error saving session: {e}")
        st.error("Failed to save the session.")

def load_saved_sessions():
    """Loads the list of saved sessions."""
    sessions = []
    try:
        for filename in os.listdir(SESSION_HISTORY_DIR):
            if filename.endswith(".pkl"):
                sessions.append(filename[:-4])  # Remove the .pkl extension
        return sessions
    except Exception as e:
        logging.error(f"Error loading saved sessions: {e}")
        return []

def load_session(session_name):
    """Loads a saved conversation history."""
    filename = os.path.join(SESSION_HISTORY_DIR, f"{session_name}.pkl")
    try:
        with open(filename, "rb") as f:
            conversation_history = pickle.load(f)
        return conversation_history
    except Exception as e:
        logging.error(f"Error loading session: {e}")
        st.error("Failed to load the session.")
        return []

def process_uploaded_pdfs(uploaded_files):
    """Processes uploaded PDF files and sets up the retrieval system for each."""
    for uploaded_file in uploaded_files:
        doc_id = os.path.splitext(uploaded_file.name)[0]
        if doc_id not in st.session_state['documents']:
            doc_dir = os.path.join("documents", doc_id)
            if not os.path.exists(doc_dir):
                os.makedirs(doc_dir)
            # Paths for storing data
            VECTOR_DB_PATH = os.path.join(doc_dir, "vector_db.pkl")
            BM25_PATH = os.path.join(doc_dir, "bm25.pkl")
            CHUNKS_PATH = os.path.join(doc_dir, "chunks.pkl")

            # Save the uploaded file to the doc_dir
            pdf_path = os.path.join(doc_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load the PDF using PyPDFLoader
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                # Now, set up the retrieval system
                chunks, bm25, db = setup_retrieval_system(documents, doc_id)
                if chunks is None or bm25 is None or db is None:
                    continue
                # Store the retrieval components per document
                st.session_state['documents'][doc_id] = {
                    'chunks': chunks,
                    'bm25': bm25,
                    'db': db,
                    'pdf_path': pdf_path
                }
                st.sidebar.success(f"Processed and indexed document: {uploaded_file.name}")
            except Exception as e:
                logging.error(f"Error processing uploaded PDF {uploaded_file.name}: {e}")
                st.sidebar.error(f"Failed to process the uploaded PDF {uploaded_file.name}.")
        else:
            st.sidebar.info(f"Document '{uploaded_file.name}' already processed.")

def get_conversation_history(selected_doc_ids):
    """Retrieves or initializes the conversation history for the selected documents."""
    doc_key = "_".join(sorted(selected_doc_ids))
    if doc_key not in st.session_state['conversation_histories']:
        st.session_state['conversation_histories'][doc_key] = []
    return st.session_state['conversation_histories'][doc_key], doc_key

def main():
    # Initialize session state variables
    if 'documents' not in st.session_state:
        st.session_state['documents'] = {}
    if 'conversation_histories' not in st.session_state:
        st.session_state['conversation_histories'] = {}
    if 'question_input' not in st.session_state:
        st.session_state['question_input'] = ''
    if "context_docs" not in st.session_state:
        st.session_state["context_docs"] = []

    # Sidebar: File Uploader, Document Selection, and Saved Sessions
    with st.sidebar:
        st.header("Provider Manual Assistant")
        st.markdown("---")

        # 1. File Uploader for PDFs
        uploaded_files = st.file_uploader("📁 Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

        # Process uploaded PDFs
        if uploaded_files:
            process_uploaded_pdfs(uploaded_files)

        st.markdown("---")

        # 2. Document Selection
        doc_ids = list(st.session_state['documents'].keys())
        if doc_ids:
            selected_doc_ids = st.multiselect("📄 Select Documents to Chat With:", doc_ids)
        else:
            st.warning("Please upload at least one PDF document.")
            selected_doc_ids = []  # No documents selected

        st.markdown("---")

        # 3. Saved Sessions
        st.header("Saved Sessions")
        saved_sessions = load_saved_sessions()
        if saved_sessions:
            selected_session = st.selectbox("🔄 Load a Saved Session:", saved_sessions)
            if st.button("Load Session"):
                # Extract doc_key from session name
                session_parts = selected_session.split('_')
                if len(session_parts) >= 2:
                    session_name = session_parts[0]
                    doc_key = '_'.join(session_parts[1:])
                    missing_docs = [doc_id for doc_id in doc_key.split('_') if doc_id not in st.session_state['documents']]
                    if not missing_docs:
                        st.session_state['conversation_histories'][doc_key] = load_session(selected_session)
                        st.sidebar.success(f"Session '{session_name}' for documents '{doc_key}' loaded.")
                    else:
                        st.sidebar.error(f"The following documents associated with this session are not uploaded: {', '.join(missing_docs)}.")
                else:
                    st.sidebar.error("Invalid session name format.")
        else:
            st.write("No saved sessions available.")

    # Main Content Area
    if selected_doc_ids:
        st.markdown("---")
        llm = setup_llm_model()
        if llm is None:
            st.stop()

        # Get the conversation history for the selected documents
        conversation_history, doc_key = get_conversation_history(selected_doc_ids)

        # Collect the document data for the selected documents
        selected_documents_data = {doc_id: st.session_state['documents'][doc_id] for doc_id in selected_doc_ids}

        # 4. Question Input
        question = st.text_input("❓ Enter your question:")

        if st.button("🔍 Search Documents"):
            if question:
                # Loading screen
                loading_messages = [
                    "Analyzing documents...",
                    "Gathering information...",
                    "Searching across multiple files...",
                    "Compiling responses...",
                    "Almost there..."
                ]
                loading_placeholder = st.empty()
                for msg in loading_messages:
                    loading_placeholder.markdown(f"### {msg}")
                    time.sleep(0.5)
                with st.spinner("Generating answer..."):
                    response, context_docs = chain_invoke_multi(
                        question, llm,
                        selected_documents_data,
                        conversation_history
                    )
                    loading_placeholder.empty()
                    st.markdown(f"### 📝 Answer")
                    st.write(response)

                    # Save context docs to session state for display in the expander
                    st.session_state["context_docs"] = context_docs

                    # Add the latest interaction to conversation history
                    conversation_history.append({"question": question, "response": response})
            else:
                st.warning("Please enter a question to search the documents.")

        # 5. Show Sources using st.expander
        if "context_docs" in st.session_state and st.session_state["context_docs"]:
            with st.expander("📚 Show Sources"):
                st.header("Chunks Used for Response")
                for idx, doc in enumerate(st.session_state["context_docs"], start=1):
                    st.markdown(f"**Chunk {idx}:**\n{doc.page_content}")
        else:
            pass  # Do nothing if no context docs are available

        # 6. Save Session
        st.markdown("---")
        session_name = st.text_input("💾 Enter a name to save this session:")
        if st.button("💾 Save Session"):
            if session_name.strip():
                save_session(conversation_history, session_name.strip(), doc_key)
            else:
                st.warning("Please enter a valid session name to save.")

        # 7. Display Conversation History
        if conversation_history:
            st.markdown("---")
            st.subheader("🗨️ Conversation History")
            for turn in reversed(conversation_history):
                st.markdown(f"**User:** {turn['question']}")
                st.markdown(f"**Assistant:** {turn['response']}")
                st.markdown("---")
    else:
        st.info("Awaiting document selection to begin chatting.")

if __name__ == "__main__":
    main()
