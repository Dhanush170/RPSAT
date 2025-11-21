import streamlit as st
import os
import time
import chromadb
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# AI Imports
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import MultiQueryRetriever

# Import from our custom utils file
from utils import (
    extract_text_from_pdf_stream,
    preprocess_for_llm,
    chunk_document,
    get_embedding_model,
    summarize_single_section,
    translate_text,
    text_to_audio,
    SECTION_RETRIEVAL_PROMPTS
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Research Paper Audio Summarizer",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables if .env exists
load_dotenv()

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: white;
        border-radius: 5px 5px 0 0;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
    div.stButton > button:first-child {
        background-color: #2C3E50;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # --- Sidebar: Settings & Keys ---
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        st.subheader("1. API Keys")
        # Try to get keys from .env, otherwise ask user
        env_google = os.getenv("GOOGLE_API_KEY")
        env_groq = os.getenv("GROQ_API_KEY")
        
        google_key = st.text_input("Google Gemini Key", value=env_google if env_google else "", type="password")
        groq_key = st.text_input("Groq API Key", value=env_groq if env_groq else "", type="password")
        
        st.markdown("---")
        
        st.subheader("2. Output Settings")
        languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Hindi": "hi",
            "Japanese": "ja",
            "Chinese": "zh-CN"
        }
        selected_lang_name = st.selectbox("Translation Language", list(languages.keys()))
        target_lang_code = languages[selected_lang_name]
        
        st.info("Upload a PDF to start.")

    # --- Main Content Area ---
    st.title("📄 Research Paper Summarizer & 🎧 Audio Player")
    st.markdown("Extract insights from PDFs, translate them, and listen on the go.")

    uploaded_file = st.file_uploader("Upload your Research Paper (PDF)", type=['pdf'])

    # --- Logic Flow ---
    if uploaded_file and google_key and groq_key:
        
        # 1. Initialize Session State to hold data
        if "summaries" not in st.session_state:
            st.session_state.summaries = {}
        if "current_filename" not in st.session_state or st.session_state.current_filename != uploaded_file.name:
            # Reset if new file
            st.session_state.summaries = {}
            st.session_state.current_filename = uploaded_file.name

        # 2. Process Button
        if not st.session_state.summaries:
            if st.button("🚀 Analyze Document"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # A. Setup Models
                    status_text.text("Initializing AI Models...")
                    genai.configure(api_key=google_key)
                    llm_model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    query_gen_llm = ChatGroq(
                        temperature=0, 
                        groq_api_key=groq_key, 
                        model_name="llama-3.3-70b-versatile"
                    )
                    embedding_model = get_embedding_model() # Loaded from utils, cached
                    progress_bar.progress(10)

                    # B. PDF Processing
                    status_text.text("Reading and cleaning PDF...")
                    file_bytes = uploaded_file.read()
                    raw_pages = extract_text_from_pdf_stream(file_bytes)
                    for page in raw_pages:
                        page['cleaned_text'] = preprocess_for_llm(page['text'])
                    final_chunks = chunk_document(raw_pages)
                    progress_bar.progress(30)

                    # C. Vector Store Creation
                    status_text.text("Creating Knowledge Base (ChromaDB)...")
                    client = chromadb.Client()
                    # Create unique collection name
                    collection_name = f"doc_{int(time.time())}"
                    vectorstore = Chroma.from_documents(
                        documents=final_chunks,
                        embedding=embedding_model,
                        collection_name=collection_name,
                        client=client
                    )
                    progress_bar.progress(50)

                    # D. Retriever Setup
                    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    multi_query_retriever = MultiQueryRetriever.from_llm(
                        retriever=base_retriever,
                        llm=query_gen_llm
                    )

                    # E. Parallel Summarization
                    status_text.text("Generating Summaries (Parallel Processing)...")
                    sections_to_process = list(SECTION_RETRIEVAL_PROMPTS.keys())
                    
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        future_to_section = {
                            executor.submit(summarize_single_section, sec, multi_query_retriever, llm_model): sec
                            for sec in sections_to_process
                        }
                        
                        completed_count = 0
                        for future in future_to_section:
                            sec = future_to_section[future]
                            try:
                                result = future.result()
                                st.session_state.summaries[sec] = result
                            except Exception as exc:
                                st.session_state.summaries[sec] = f"Error: {exc}"
                            
                            completed_count += 1
                            progress = 50 + int((completed_count / len(sections_to_process)) * 50)
                            progress_bar.progress(progress)

                    status_text.text("Analysis Complete!")
                    time.sleep(1)
                    st.rerun() # Refresh to show results

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

        # 3. Display Results
        if st.session_state.summaries:
            st.success("Document Analyzed Successfully!")
            
            # Create tabs for sections
            section_names = list(st.session_state.summaries.keys())
            tabs = st.tabs(section_names)

            for i, section in enumerate(section_names):
                with tabs[i]:
                    original_summary = st.session_state.summaries[section]
                    
                    # Layout: Summary on Left, Translation/Audio on Right
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(f"{section} Summary")
                        st.write(original_summary)
                    
                    with col2:
                        st.subheader("Translation & Audio")
                        st.caption(f"Target Language: {selected_lang_name}")
                        
                        # Use a unique key for each button so they don't conflict
                        if st.button(f"Translate & Listen", key=f"btn_{section}"):
                            with st.spinner("Translating..."):
                                # Translate
                                trans_text = translate_text(original_summary, target_lang_code)
                                st.markdown(f"**{selected_lang_name}:**")
                                st.info(trans_text)
                                
                                # Audio
                                audio_file = text_to_audio(trans_text, target_lang_code)
                                if audio_file:
                                    st.audio(audio_file, format='audio/mp3')
    
    elif not uploaded_file:
        st.info("waiting for file upload...")
    elif not (google_key and groq_key):
        st.warning("Please provide your API Keys in the sidebar.")

if __name__ == '__main__':
    main()