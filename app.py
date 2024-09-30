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

# Updated imports from langchain_community
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import nltk

# Prevent repeated downloads of 'punkt'
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Attempt to import tiktoken for accurate token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

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
        border-radius: 5px; /* Rounded corners */
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

    /* Targeting the dropdown in the sidebar */
span[data-baseweb="select"] {
    background-color: #D3D9D4 !important; /* Light gray background */
    color: #212A31 !important; /* Dark text for contrast */
    border: 1px solid #124E66; /* Teal border */
    border-radius: 5px; /* Optional: Rounded corners */
}

/* When the dropdown is in its expanded (opened) state */
ul[data-baseweb="menu"] {
    background-color: #FFFFFF !important; /* White background for options */
    color: #212A31 !important; /* Dark text for options */
}

/* Styling individual items in the dropdown */
li[data-baseweb="menu-item"] {
    background-color: #FFFFFF !important; /* White background */
    color: #212A31 !important; /* Dark text */
}

/* When hovering over dropdown items */
li[data-baseweb="menu-item"]:hover {
    background-color: #D3D9D4 !important; /* Light gray background on hover */
    color: #124E66 !important; /* Teal text on hover */
}

/* Tag styling for selected items */
span[data-baseweb="tag"][role="button"] {
    background-color: #2E3944 !important; /* Slate gray for selected item */
    color: #FFFFFF !important; /* White text for selected item */
}
    </style>
    """,
    unsafe_allow_html=True,
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Persistent storage paths
SESSION_HISTORY_DIR = "saved_sessions"
DOCUMENTS_DIR = "documents"

# Ensure the directories for saved sessions and documents exist
for directory in [SESSION_HISTORY_DIR, DOCUMENTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def clean_text(text):
    """Cleans the input text by removing extra whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def sanitize_filename(filename):
    """Sanitizes the filename to prevent filesystem issues."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

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
    doc_dir = os.path.join(DOCUMENTS_DIR, doc_id)
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

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Counts the number of tokens in a given text using tiktoken.

    Args:
        text (str): The text to tokenize.
        model (str): The model name for encoding. Adjust if using a different model.

    Returns:
        int: The number of tokens.
    """
    if not tiktoken:
        return approximate_token_count(text)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding
    tokens = encoding.encode(text)
    return len(tokens)

def approximate_token_count(text: str) -> int:
    """
    Approximates the number of tokens in a given text.

    Args:
        text (str): The text to tokenize.

    Returns:
        int: The approximate number of tokens.
    """
    # Approximation: 1 token ~= 0.75 words
    return int(len(text.split()) / 0.75)

def combined_retrieve_multi(query, documents_data, bm25_top_k=3, faiss_top_k=3):
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
        top_k_indices = bm25_scores.argsort()[-bm25_top_k:][::-1]
        bm25_results.extend([chunks[i] for i in top_k_indices])

        # FAISS similarity search
        faiss_results.extend(db.similarity_search(query, k=faiss_top_k))

    # Combine results, avoiding duplicates
    seen = set()
    combined_results = []
    for doc in bm25_results + faiss_results:
        if doc.page_content not in seen:
            combined_results.append(doc)
            seen.add(doc.page_content)
    return combined_results

def chain_invoke_multi(question, llm, documents_data, conversation_history, max_tokens=2048):
    """
    Generates an answer using the LLM, including context and conversation history, from multiple documents.

    Args:
        question (str): The user's question.
        llm: The language model instance.
        documents_data (dict): Data of selected documents.
        conversation_history (list): List of past interactions.
        max_tokens (int): Maximum allowed tokens.

    Returns:
        tuple: (response_text, context_docs)
    """
    context_docs = combined_retrieve_multi(question, documents_data)
    context = "\n".join([doc.page_content for doc in context_docs])

    # Build the conversation history in reverse order (most recent first)
    conversation = ""
    for turn in reversed(conversation_history[-10:]):  # Limit to last 10 interactions
        conversation += f"User: {turn['question']}\nAssistant: {turn['response']}\n"

    # Construct initial prompt
    prompt = f"""
You are an assistant helping answer questions based on the provided context.

Context:
{context}

Conversation history:
{conversation}

Current question: {question}

If the user refers to previous responses or seeks corrections, use the conversation history to provide accurate and context-aware answers.

Answer the question based ONLY on the provided context and conversation history.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
"""

    # Count tokens
    token_count = count_tokens(prompt)

    # Check if token count exceeds the limit
    if token_count > max_tokens:
        excess_tokens = token_count - max_tokens

        # Strategy:
        # 1. Trim conversation history first
        # 2. If still exceeding, trim context

        # 1. Trim conversation history
        while excess_tokens > 0 and len(conversation_history) > 0:
            # Remove the oldest turn
            conversation_history.pop(0)
            # Rebuild conversation
            conversation = ""
            for turn in reversed(conversation_history[-10:]):
                conversation += f"User: {turn['question']}\nAssistant: {turn['response']}\n"
            # Reconstruct prompt
            prompt = f"""
You are an assistant helping answer questions based on the provided context.

Context:
{context}

Conversation history:
{conversation}

Current question: {question}

If the user refers to previous responses or seeks corrections, use the conversation history to provide accurate and context-aware answers.

Answer the question based ONLY on the provided context and conversation history.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
"""
            # Re-count tokens
            token_count = count_tokens(prompt)
            excess_tokens = token_count - max_tokens

        # 2. Trim context
        if token_count > max_tokens:
            # Calculate how much to trim context
            # Allow some buffer for the prompt structure
            buffer_tokens = 100
            conversation_tokens = count_tokens(f"""
You are an assistant helping answer questions based on the provided context.

Conversation history:
{conversation}

Current question: {question}

If the user refers to previous responses or seeks corrections, use the conversation history to provide accurate and context-aware answers.

Answer the question based ONLY on the provided context and conversation history.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
""") if tiktoken else approximate_token_count(f"""
You are an assistant helping answer questions based on the provided context.

Conversation history:
{conversation}

Current question: {question}

If the user refers to previous responses or seeks corrections, use the conversation history to provide accurate and context-aware answers.

Answer the question based ONLY on the provided context and conversation history.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
""")

            allowed_context_tokens = max_tokens - buffer_tokens - conversation_tokens

            # Estimate tokens per chunk
            if tiktoken:
                tokens_per_chunk = [count_tokens(doc.page_content) for doc in context_docs]
            else:
                tokens_per_chunk = [approximate_token_count(doc.page_content) for doc in context_docs]

            # Select chunks that fit within allowed_context_tokens
            trimmed_context = []
            current_tokens = 0
            for doc, tokens in zip(context_docs, tokens_per_chunk):
                if current_tokens + tokens > allowed_context_tokens:
                    break
                trimmed_context.append(doc.page_content)
                current_tokens += tokens

            # Reconstruct context
            context = "\n".join(trimmed_context)

            # Reconstruct prompt
            prompt = f"""
You are an assistant helping answer questions based on the provided context.

Context:
{context}

Conversation history:
{conversation}

Current question: {question}

If the user refers to previous responses or seeks corrections, use the conversation history to provide accurate and context-aware answers.

Answer the question based ONLY on the provided context and conversation history.
If you don't know the answer, just say that you don't know; don't try to make up an answer.
"""

            # Re-count tokens
            token_count = count_tokens(prompt)

            # If still exceeding, consider further trimming or notify the user
            if token_count > max_tokens:
                st.warning("Your request is too long and cannot be processed. Please simplify your question or reduce the number of selected documents.")
                return "", context_docs

    try:
        response = llm.invoke(prompt)
        return response.content, context_docs  # Return the response and the context docs
    except Exception as e:
        logging.error(f"Error invoking LLM: {e}")
        st.error("Failed to generate a response from the LLM.")
        return "", []

def save_session(conversation_history, session_name, doc_key):
    """Saves the conversation history to a file."""
    sanitized_session_name = sanitize_filename(session_name.strip())
    filename = os.path.join(SESSION_HISTORY_DIR, f"{sanitized_session_name}_{doc_key}.pkl")
    try:
        with open(filename, "wb") as f:
            pickle.dump(conversation_history, f)
        st.success(f"Session '{sanitized_session_name}' for documents '{doc_key}' saved successfully.")
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

def is_valid_pdf(file) -> bool:
    """Validates if the uploaded file is a PDF and within size limits."""
    return file.type == "application/pdf" and file.size < 10 * 1024 * 1024  # 10 MB limit

def process_uploaded_pdfs(uploaded_files):
    """Processes uploaded PDF files and sets up the retrieval system for each."""
    for uploaded_file in uploaded_files:
        if not is_valid_pdf(uploaded_file):
            st.sidebar.error(f"Invalid file: {uploaded_file.name}. Ensure it's a PDF and less than 10MB.")
            continue

        doc_id = os.path.splitext(sanitize_filename(uploaded_file.name))[0]
        if doc_id not in st.session_state['documents']:
            doc_dir = os.path.join(DOCUMENTS_DIR, doc_id)
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

def load_existing_documents():
    """Loads existing documents from the documents directory into session state."""
    for doc_id in os.listdir(DOCUMENTS_DIR):
        doc_dir = os.path.join(DOCUMENTS_DIR, doc_id)
        if os.path.isdir(doc_dir) and doc_id not in st.session_state['documents']:
            VECTOR_DB_PATH = os.path.join(doc_dir, "vector_db.pkl")
            BM25_PATH = os.path.join(doc_dir, "bm25.pkl")
            CHUNKS_PATH = os.path.join(doc_dir, "chunks.pkl")
            PDF_PATH = None

            # Find the PDF file in the directory
            for file in os.listdir(doc_dir):
                if file.endswith(".pdf"):
                    PDF_PATH = os.path.join(doc_dir, file)
                    break

            if not PDF_PATH:
                logging.warning(f"No PDF found in {doc_dir}. Skipping.")
                continue

            # Load retrieval data
            chunks = load_from_disk(CHUNKS_PATH)
            bm25 = load_from_disk(BM25_PATH)
            db = load_from_disk(VECTOR_DB_PATH)

            if chunks is None or bm25 is None or db is None:
                logging.warning(f"Incomplete retrieval data for '{doc_id}'. Skipping.")
                continue

            # Store the retrieval components per document
            st.session_state['documents'][doc_id] = {
                'chunks': chunks,
                'bm25': bm25,
                'db': db,
                'pdf_path': PDF_PATH
            }
            logging.info(f"Loaded existing document '{doc_id}' into session state.")

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

    # Load existing documents from the documents directory
    load_existing_documents()

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
                        loaded_history = load_session(selected_session)
                        if loaded_history:
                            st.session_state['conversation_histories'][doc_key] = loaded_history
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

        # 4. Question Input with Header
        st.header("Enter Your Question")
        question = st.text_input("Question", key="question", label_visibility="hidden")

        if st.button("🔍 Search Documents"):
            if question:
                # Enhanced loading screen with dynamic messages lasting 5 seconds each
                loading_messages = [
                    # Einstein quotes
    "Imagination is more important than knowledge. – Albert Einstein",
    "Life is like riding a bicycle. To keep your balance, you must keep moving. – Albert Einstein",
    "The important thing is not to stop questioning. Curiosity has its own reason for existence. – Albert Einstein",
    
    # Alan Turing quotes
    "We can only see a short distance ahead, but we can see plenty there that needs to be done. – Alan Turing",
    "Machines take me by surprise with great frequency. – Alan Turing",
    "Sometimes it is the people no one can imagine anything of who do the things no one can imagine. – Alan Turing",
    
    # Goethe quotes
    "Knowing is not enough; we must apply. Willing is not enough; we must do. – Johann Wolfgang von Goethe",
    "Whatever you can do or dream you can, begin it. Boldness has genius, power, and magic in it. – Johann Wolfgang von Goethe",
    "Doubt can only be removed by action. – Johann Wolfgang von Goethe",
    
    # Richard Feynman quotes
    "I would rather have questions that can't be answered than answers that can't be questioned. – Richard Feynman",
    "The first principle is that you must not fool yourself and you are the easiest person to fool. – Richard Feynman",
    "What I cannot create, I do not understand. – Richard Feynman",
                ]
                loading_placeholder = st.empty()
                for msg in loading_messages:
                    loading_placeholder.markdown(f"### {msg}")
                    time.sleep(5)  # Display each message for 5 seconds
                with st.spinner("Generating answer..."):
                    response, context_docs = chain_invoke_multi(
                        question, llm,
                        selected_documents_data,
                        conversation_history,
                        max_tokens=2048
                    )
                    loading_placeholder.empty()  # Remove the loading message
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
        session_name = st.text_input("💾 Enter a name to save this session:", key="session_name")
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
