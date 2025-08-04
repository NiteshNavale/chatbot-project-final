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
            # Move the file pointer to the beginning before processing
            doc.seek(0)
            
            if doc.name.endswith('.pdf'):
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
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
                text += doc.getvalue().decode("utf-8", errors='ignore') + "\n"
            elif doc.name.endswith('.csv'):
                # --- MODIFIED FOR ROBUSTNESS ---
                # Read the CSV as a plain text file, line by line.
                # This avoids all pandas parsing errors.
                text += doc.getvalue().decode("utf-8", errors='ignore') + "\n"

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
    # Use a smaller chunk size to avoid exceeding the model's context window
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
    This function is updated to use the 'map_reduce' chain type correctly.
    """
    
    # Prompt for the MAP step: processes each chunk of text
    map_prompt_template = """
    Based on the following context, answer the question.
    Provide a detailed and comprehensive answer extracting all relevant information from the text.
    If the context does not contain the answer, state that the information is not available.
    
    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """
    map_prompt = PromptTemplate.from_template(map_prompt_template)

    # Prompt for the REDUCE step: combines the answers from the map step
    combine_prompt_template = """
    You are given a question and a set of answers from different sections of a document.
    Your task is to synthesize these answers into a single, final, and coherent response.
    Do not add any information that is not present in the provided answers.
    If the answers indicate that the information is not available, then state that the answer is not available in the context.
    
    Question:
    {question}
    
    Set of Answers:
    {summaries}
    
    Final Answer:
    """
    combine_prompt = PromptTemplate.from_template(combine_prompt_template)
    
    # Initialize the Groq model
    model = ChatGroq(
        model_name="llama3-8b-8192", 
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Create the QA chain with the separate map and combine prompts
    chain = load_qa_chain(
        llm=model,
        chain_type="map_reduce",
        return_intermediate_steps=False,
        question_prompt=map_prompt,  # Renamed from map_prompt for clarity in older versions
        combine_prompt=combine_prompt
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
