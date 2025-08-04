import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import docx
from pptx import Presentation
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Set up the page configuration for the Streamlit app
st.set_page_config(page_title="Chat with Documents", layout="wide")

# Load environment variables from the .env file
load_dotenv()

def get_docs_text(docs):
    """
    Extracts text content from a list of uploaded document files.

    Args:
        docs (list): A list of uploaded file objects.

    Returns:
        str: The concatenated text from all document files.
    """
    text = ""
    for doc in docs:
        try:
            if doc.name.endswith('.pdf'):
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            elif doc.name.endswith('.docx'):
                document = docx.Document(doc)
                for para in document.paragraphs:
                    text += para.text + "\n"
            elif doc.name.endswith('.pptx'):
                prs = Presentation(doc)
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text += shape.text + "\n"
            elif doc.name.endswith('.txt'):
                text += doc.getvalue().decode("utf-8") + "\n"
            elif doc.name.endswith('.csv'):
                # First, try to read with standard UTF-8 encoding
                try:
                    df = pd.read_csv(doc)
                    text += df.to_string() + "\n"
                except UnicodeDecodeError:
                    # If UTF-8 fails, reset the file pointer and try 'latin-1'
                    doc.seek(0)
                    df = pd.read_csv(doc, encoding='latin-1')
                    text += df.to_string() + "\n"
        except Exception as e:
            st.error(f"Error processing file {doc.name}: {e}")
            # Optionally, you can log the error for debugging
            # print(f"Could not process file {doc.name}. Error: {e}")
            continue # Move to the next file
    return text
def get_text_chunks(text):
    """
    Splits the input text into smaller chunks for processing.

    Args:
        text (str): The input text.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates and saves a FAISS vector store from text chunks.

    Args:
        text_chunks (list): A list of text chunks.
    """
    # Use a popular open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create the vector store using FAISS
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store locally (optional, for persistence)
    vector_store.save_local("faiss_index")
    st.session_state.vector_store_created = True


def get_conversational_chain():
    """
    Initializes and returns a conversational QA chain using the Groq model.

    Returns:
        A LangChain conversational chain object.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
    If the answer is not in the provided context, just say, "The answer is not available in the context".
    Do not provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    # Initialize the Groq model
    model = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Create the QA chain with a custom prompt
    prompt = PromptTemplate.from_template(prompt_template)
    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )
    return chain

def user_input(user_question):
    """
    Handles user input, queries the vector store, and gets a response from the LLM.

    Args:
        user_question (str): The question asked by the user.
    """
    if "vector_store" not in st.session_state or not st.session_state.vector_store:
        st.warning("Please upload and process your documents first.")
        return

    # Use a popular open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Perform a similarity search in the vector store
    docs = st.session_state.vector_store.similarity_search(user_question, k=3)

    # Get the conversational chain
    chain = get_conversational_chain()

    # Get the response from the chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Add the current interaction to the chat history
    st.session_state.chat_history.append(("You", user_question))
    st.session_state.chat_history.append(("Bot", response["output_text"]))

def main():
    """
    The main function that runs the Streamlit application.
    """
    # Initialize session state variables if they don't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "vector_store_created" not in st.session_state:
        st.session_state.vector_store_created = False

    # App Title
    st.title("📄 Chat with Your Documents. Developed by Nitesh")
    st.write("Upload your PDF, DOCX, PPTX, TXT, or CSV documents, and ask any questions about their content!")

    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("Settings")

        # Check if API key is available
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found. Please set it in your .env file.")
            st.stop()
        else:
            st.success("Groq API Key loaded successfully.")

        st.subheader("Your Documents")
        uploaded_docs = st.file_uploader(
            "Upload your files here and click on 'Process'",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'pptx', 'txt', 'csv']
        )

        if st.button("Process Documents"):
            if uploaded_docs:
                with st.spinner("Processing your documents... This may take a moment."):
                    # 1. Extract text from documents
                    raw_text = get_docs_text(uploaded_docs)

                    # 2. Split text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # --- ADDED CHECK ---
                    # 3. Check if text was extracted
                    if not text_chunks:
                        st.warning("Could not extract any text from the documents. Please ensure the files are not empty or image-based scans.")
                    else:
                        # 4. Create and save vector store
                        get_vector_store(text_chunks)

                        # 5. Load the vector store into session state
                        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                        st.success("Documents processed successfully! You can now ask questions.")

            else:
                st.warning("Please upload at least one document file.")

    # Main chat interface
    st.header("Chatbot")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role.lower()):
                st.write(message)

    # Handle user input at the bottom
    if user_question := st.chat_input("Ask a question about your documents..."):
        user_input(user_question)

        # Rerun to display the latest chat messages
        st.rerun()

if __name__ == "__main__":
    main()
