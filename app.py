import streamlit as st
import os
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader

# Import Groq's chat model and other core langchain components
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# CORRECT import line for the refined chain type

from langchain_core.prompts import PromptTemplate

def extract_text_from_file(file):
    """Extracts text from an uploaded file object."""
    file_extension = os.path.splitext(file.name)[1].lower()
    text = ""
    
    if file_extension == '.docx':
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_extension == '.pdf':
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            text = ""
    elif file_extension in ['.xlsx', '.xls', '.csv']:
        try:
            df = pd.read_excel(file) if file_extension in ['.xlsx', '.xls'] else pd.read_csv(file)
            text = df.to_string(index=False)
        except Exception as e:
            st.error(f"Error reading Excel/CSV: {e}")
            text = ""
    else:
        st.warning(f"Unsupported file type: {file_extension}")
        text = ""

    return text

def get_vector_store(text):
    """
    Splits text into chunks, embeds them using a free Hugging Face model,
    and stores them in a vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Initialize a free Hugging Face embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    """
    Creates a conversational Q&A chain using the Groq chat model
    and a custom refine chain.
    """
    llm = ChatGroq(
        temperature=0.7,
        model_name="mixtral-8x7b-32768", 
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )
    
    # Define the prompts for the refine chain
    question_prompt_template = """
    Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}

    Question: {question}
    """
    question_prompt = PromptTemplate.from_template(question_prompt_template)

    refine_prompt_template = """
    The original question is as follows: {question}
    We have provided an existing answer: {existing_answer}
    We have the opportunity to refine the existing answer with some more context below.
    ------------
    {context}
    ------------
    Given the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer.
    """
    refine_prompt = PromptTemplate.from_template(refine_prompt_template)
    
    # Create the refine document chain
    doc_chain = load_qa_chain(
        llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
    )
    
    # Create the ConversationalRetrievalChain by passing the components
    conversation_chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        question_generator=llm,
        combine_docs_chain=doc_chain,
    )
    
    return conversation_chain

def handle_user_input(user_question):
    """Handles user questions and displays the chatbot's response."""
    if "conversation" in st.session_state and "chat_history" in st.session_state:
        response = st.session_state.conversation({'question': user_question, 'chat_history': st.session_state.chat_history})
        st.session_state.chat_history = response['chat_history']
        st.write("You:", user_question)
        st.write("Bot:", response['answer'])

def main():
    st.set_page_config(page_title="Document Chatbot", page_icon=":books:")
    st.title("Chat with Your Documents")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    with st.sidebar:
        st.subheader("Upload your documents")
        uploaded_file = st.file_uploader(
            "Upload files and click 'Process'",
            type=["pdf", "docx", "xlsx", "csv"],
            accept_multiple_files=False
        )
        if st.button("Process Document"):
            if uploaded_file:
                with st.spinner("Processing..."):
                    raw_text = extract_text_from_file(uploaded_file)
                    if raw_text:
                        vector_store = get_vector_store(raw_text)
                        st.session_state.conversation = get_conversation_chain(vector_store)
                        st.success("Document processed successfully!")
                    else:
                        st.error("Failed to process the document.")
            else:
                st.error("Please upload a document first.")
    
    st.subheader("Ask a question about your document:")
    user_question = st.text_input("Your question:")
    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()
