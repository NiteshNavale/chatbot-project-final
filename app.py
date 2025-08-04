import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO
from PyPDF2 import PdfReader
import docx
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DocuBot",
    page_icon="🤖",
    layout="wide"
)

# --- ENVIRONMENT AND API KEY LOADING ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- UI STYLING (CSS) ---
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #F0F2F6;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: #FFFFFF;
        border-right: 2px solid #E0E0E0;
    }
    /* Chat message containers */
    [data-testid="chat-message-container"] {
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
    }
    /* User message styling */
    [data-testid="chat-message-container-user"] {
        background-color: #E1F5FE; /* Light blue */
    }
    /* Bot message styling */
    [data-testid="chat-message-container-assistant"] {
        background-color: #FFFFFF; /* White */
    }
    /* Table styling */
    .stDataFrame, .stTable {
        border-radius: 8px;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);
        border: 1px solid #E0E0E0;
    }
    /* Title styling */
    h1 {
        color: #1E1E1E;
        font-weight: 600;
    }
    /* Subheader styling */
    h2, h3 {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# --- CORE FUNCTIONS ---

def get_docs_text(docs):
    """Extracts text with the most robust and forgiving CSV handling."""
    text = ""
    for doc in docs:
        try:
            doc.seek(0)
            file_extension = os.path.splitext(doc.name).lower()
            
            if file_extension == '.pdf':
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            elif file_extension == '.docx':
                document = docx.Document(doc)
                for para in document.paragraphs:
                    text += para.text + "\n"
            elif file_extension == '.pptx':
                prs = Presentation(doc)
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text += shape.text + "\n"
            elif file_extension == '.txt':
                text += doc.getvalue().decode("utf-8", errors='ignore') + "\n"

            # --- DEFINITIVE CSV HANDLING ---
            elif file_extension == '.csv':
                try:
                    # MASTER TRY: Attempt to parse as a structured table with forgiveness.
                    # We use the 'python' engine because it fully supports 'on_bad_lines'.
                    common_kwargs = {'on_bad_lines': 'skip', 'engine': 'python'}
                    try:
                        # Attempt 1: Standard UTF-8
                        doc.seek(0)
                        df = pd.read_csv(doc, encoding='utf-8', **common_kwargs)
                        text += df.to_string(index=False) + "\n\n"
                    except UnicodeDecodeError:
                        # Attempt 2: Forgiving Latin-1 for Excel files
                        doc.seek(0)
                        df = pd.read_csv(doc, encoding='latin-1', **common_kwargs)
                        text += df.to_string(index=False) + "\n\n"
                except Exception as e:
                    # ULTIMATE FALLBACK: If even the forgiving parser fails, read as raw text.
                    st.warning(f"Could not parse CSV '{doc.name}' even with forgiving settings. Reading as raw text. Error: {e}")
                    doc.seek(0)
                    text += doc.getvalue().decode("utf-8", errors='ignore') + "\n"
                    
        except Exception as e:
            st.error(f"An unexpected error occurred while processing {doc.name}: {e}")
            continue
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False

def get_conversational_chain():
    """Initializes the conversational QA chain."""
    map_prompt_template = """
    Based on the following context, answer the question. Extract all relevant details.
    Context: {context}
    Question: {question}
    Answer:
    """
    map_prompt = PromptTemplate.from_template(map_prompt_template)
    combine_prompt_template = """
    You are an expert assistant. You will be given a question and a set of extracted text from a document.
    Synthesize these into a single, coherent final response.
    CRITICAL INSTRUCTION: If the user's question asks for a comparison, a list of items, a summary of features,
    or any other structured data, YOUR FINAL ANSWER MUST be formatted as a Markdown table.
    Use columns and rows appropriately. Do not just list items; structure them in a table.
    For all other questions, provide a clear, well-formatted text answer.
    Question: {question}
    Set of Answers: {summaries}
    Final Answer:
    """
    combine_prompt = PromptTemplate.from_template(combine_prompt_template)
    model = ChatGroq(model_name="llama3-8b-8192", temperature=0.2, api_key=groq_api_key)
    return load_qa_chain(llm=model, chain_type="map_reduce", return_intermediate_steps=False, question_prompt=map_prompt, combine_prompt=combine_prompt)

def handle_user_input(user_question):
    """Processes user questions and displays the bot's response."""
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("Please upload and process documents before asking a question.")
        return

    with st.spinner("Analyzing documents..."):
        try:
            chain = get_conversational_chain()
            docs = st.session_state.vector_store.similarity_search(user_question, k=4)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            output = response["output_text"]
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": output})
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- MAIN APP LAYOUT ---

def main():
    """Main function to run the Streamlit app."""
    st.title("🤖 DocuBot: Your Intelligent Document Assistant")
    st.write("Upload your documents, and I'll help you find the answers you need.")

    # --- SESSION STATE INITIALIZATION AND VALIDATION ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Check if history is in the old tuple format and reset if it is
    if st.session_state.chat_history and not isinstance(st.session_state.chat_history, dict):
        st.session_state.chat_history = []
        st.rerun()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🛠️ Setup Panel")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found. Please set it in your environment.")
            st.stop()
        
        st.subheader("1. Upload Your Documents")
        uploaded_docs = st.file_uploader("Supports PDF, DOCX, PPTX, TXT, CSV", accept_multiple_files=True, type=['pdf', 'docx', 'pptx', 'txt', 'csv'])

        st.subheader("2. Process Documents")
        if st.button("Process", use_container_width=True):
            if uploaded_docs:
                with st.spinner("Reading, chunking, and embedding... please wait."):
                    raw_text = get_docs_text(uploaded_docs)
                    if not raw_text.strip():
                        st.warning("No text extracted. Check document content.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        if get_vector_store(text_chunks):
                            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                            st.success("Knowledge base is ready!")
                            st.session_state.chat_history = []
                            st.rerun()
            else:
                st.warning("Please upload at least one document.")

    # --- MAIN CHAT INTERFACE ---
    st.header("💬 Chat with DocuBot")
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.info("Please process your documents in the sidebar to begin the chat.")

    # Display chat history from session state
    for message in st.session_state.chat_history:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            content = message["content"]
            if '|' in content and '---' in content:
                try:
                    table_data = content[content.find('|'):]
                    df = pd.read_csv(StringIO(table_data), sep='|', header=0, skipinitialspace=True).dropna(axis=1, how='all').iloc[1:].rename(columns=lambda x: x.strip())
                    st.table(df.reset_index(drop=True))
                except Exception:
                    st.markdown(content)
            else:
                st.markdown(content)
    
    # Handle user input
    if user_question := st.chat_input("Ask a question about the content of your documents..."):
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
