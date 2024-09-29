import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# Load the GROQ API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

# Title of the app with a styled header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Llama-RAG Research Assistant with Ollama</h1>", unsafe_allow_html=True)
st.markdown("Welcome to the **RAG Document Q&A System**. Upload research papers and ask questions for instant responses using cutting-edge Groq AI and LangChain!")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

# Sidebar for document embedding
st.sidebar.title("Document Embedding")
st.sidebar.markdown("Click below to process your research papers.")
if st.sidebar.button("Embed Documents"):
    with st.spinner("Creating document embeddings..."):
        if "vectors" not in st.session_state:
            st.session_state.embeddings = OllamaEmbeddings()  
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data Ingestion step
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    st.sidebar.success("Vector Database is ready!")

# Main section for user query input
st.markdown("<h3>Ask a question about the research papers</h3>", unsafe_allow_html=True)
user_prompt = st.text_input("Enter your query from the research papers")

if user_prompt:
    import time
    with st.spinner("Searching for answers..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        end = time.process_time()

        st.markdown(f"### Response Time: {round(end - start, 2)} seconds")
        st.success(response['answer'])

        # Expandable section for document similarity search
        with st.expander("Document Similarity Search", expanded=False):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.markdown('---')

# Footer
st.markdown("<hr style='border:1px solid #ccc' />", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Developed using Groq, LangChain, and Streamlit</p>", unsafe_allow_html=True)
