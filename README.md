# RPSAT – Research Paper Summarizer and Audio Translator (RAG based)
RPSAT is a lightweight Streamlit application that summarizes research papers and PDFs using a Retrieval-Augmented Generation (RAG) pipeline.  
It extracts text from PDFs, chunks it, stores embeddings in ChromaDB, retrieves the most relevant sections, and generates concise summaries in multiple languages with optional text-to-speech output.

## 🔥 Features
- RAG-based summarization (LangChain + ChromaDB)
- PDF text extraction using PyMuPDF
- Chunking & semantic retrieval
- AI-powered summarization
- Multilingual translation
- Text-to-Speech support

## 🛠 Tech Stack
Python • Streamlit • LangChain • ChromaDB • PyMuPDF • gTTS • deeptranslator

## 🚀 How to Run
### 1. Install dependencies
pip install -r requirements.txt

### 2. Run the Streamlit app
streamlit run app.py

### 3. Upload a PDF and generate:
Summaries
Translations
Audio output
