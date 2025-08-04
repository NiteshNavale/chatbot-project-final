import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# Set up the page configuration for the Streamlit app
st.set_page_config(page_title="Chat with Documents", layout="wide")

# Load environment variables from the .env file
load_dotenv()

def get_pdf_text(pdf_docs):
    """
    Extracts text content from a list of uploaded PDF files.

    Args:
        pdf_docs (list): A list of uploaded PDF file objects.

    Returns:
        str: The concatenated text from all PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
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
    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=st.session_state.prompt_template_class.from_template(prompt_template)
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
    
    # Needed for the prompt template in the chain
    from langchain.prompts import PromptTemplate
    st.session_state.prompt_template_class = PromptTemplate

    # App Title
    st.title("📄 Chat with Your Documents. Developed by Nitesh")
    st.write("Upload your PDF documents, and ask any questions about their content!")

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
        pdf_docs = st.file_uploader(
            "Upload your PDF files here and click on 'Process'",
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing your documents... This may take a moment."):
                    # 1. Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Create and save vector store
                    get_vector_store(text_chunks)
                    
                    # 4. Load the vector store into session state
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

                st.success("Documents processed successfully! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

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
