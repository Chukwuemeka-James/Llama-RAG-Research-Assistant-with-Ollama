# Llama-RAG Research Assistant with Ollama

**Llama-RAG Research Assistant** provides a powerful Retrieval-Augmented Generation (RAG) system that allows users to upload research papers and ask questions about them. Using Groq AI's Llama model and LangChain, this app retrieves relevant document sections and generates highly accurate answers.

## Features

- **Document Embedding**: Upload research papers (PDF format), and the app will create document embeddings for quick and efficient retrieval.
- **Question-Answering**: Ask questions about the uploaded documents, and the app will retrieve relevant information and generate answers based on the content.
- **Document Similarity Search**: Provides a list of document sections that are most similar to the user's query, allowing users to dive deeper into the research material.

## Technologies Used

- **Groq API**: Powered by the Llama3-8b model for intelligent response generation.
- **LangChain**: Handles document retrieval, embeddings, and text processing.
- **Ollama Embeddings**: Used to create vector embeddings of the research documents.
- **FAISS**: Vector database used for fast document similarity searches.
- **Streamlit**: Provides an easy-to-use interface for interacting with the system.

## How It Works

1. **Upload Research Papers**: Upload PDF documents to the app for analysis.
2. **Document Embedding**: The app will split the documents into chunks and create vector embeddings using Ollama.
3. **Ask a Question**: Input your question in the text box, and the app will search through the documents to find the most relevant answers.
4. **Get Instant Results**: The app provides a direct answer to your query and shows relevant document sections.
