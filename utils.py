import re
import fitz  # PyMuPDF
import chromadb
from typing import List, Dict, Union
from io import BytesIO

# AI & LangChain Imports
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Translation & Audio Imports
from gtts import gTTS
from deep_translator import GoogleTranslator

# --- Constants & Prompts ---

SECTION_RETRIEVAL_PROMPTS = {
    "Abstract": "Extract only the text under the section titled 'Abstract' from the document. Do not include content from any other section. Return the extracted text exactly as it appears.",
    "Introduction": "Extract only the text under the section titled 'Introduction' from the document. Exclude any text from other sections. Return the extracted text verbatim.",
    "Methodology": "Extract only the text under the 'Proposed Methodology' (or 'Methodology' / 'Methods') section. Return the section content exactly as shown in the paper.",
    "Challenges": "Extract only the text under the 'Challenges and Limitations' section. Do not include any unrelated content.",
    "Conclusion": "Extract only the text under the 'Conclusion' section. Return it exactly as it appears in the document."
}

SECTION_SUMMARY_PROMPTS = {
    "Abstract": "Summarize the extracted Abstract in 2–3 lines. Capture the main problem, proposed approach, and key findings. Ensure the summary is crisp yet impactful. Provide one thought-provoking insight about the direction or importance of the study.(70 words)",
    "Introduction": "Summarize the Introduction by capturing: the motivation behind the study, the context and significance of the problem, the research gap, and the objective of the work. Add one reflective insight.(70 words)",
    "Methodology": "Summarize the Proposed Methodology by capturing the core workflow, models/algorithms used, novel elements, and architecture. Explain the method in simple but technically accurate language.(70 words)",
    "Challenges": "Summarize the Challenges and Limitations section by identifying known weaknesses, practical barriers, failure cases, and assumptions. Add one reflective insight.(70 words)",
    "Conclusion": "Summarize the Conclusion by listing key outcomes, contributions, final remarks, and overall significance of the work. End with one thought-provoking idea about the broader impact.(70 words)"
}

# --- PDF & Text Processing Functions ---

def extract_text_from_pdf_stream(pdf_stream) -> List[Dict[str, Union[int, str]]]:
    """Extracts raw text from PDF bytes directly from memory."""
    try:
        # Open the PDF from the bytes stream
        with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
            pages_data = [{'page_number': i + 1, 'text': page.get_text()} for i, page in enumerate(doc)]
        return pages_data
    except Exception as e:
        print(f"Error opening PDF stream: {e}")
        return []

def preprocess_for_llm(text: str) -> str:
    """Cleans and structures raw text."""
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenated words
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Remove single newlines
    text = re.sub(r'\s{2,}', ' ', text)           # Remove extra spaces
    return text.strip()

def chunk_document(processed_pages: List[Dict[str, Union[int, str]]]) -> List:
    """Consolidates and chunks the document."""
    full_text = ""
    for page in processed_pages:
        page_separator = f"\n\n--- PAGE {page['page_number']} ---\n\n"
        full_text += page_separator + page['cleaned_text']

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=True,
        separators=[r"\n\n--- PAGE \d+ ---\n\n", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.create_documents([full_text])

    final_chunks = []
    last_seen_page_number = None
    for chunk in chunks:
        cleaned_content = re.sub(r"--- PAGE \d+ ---", "", chunk.page_content).strip()
        page_marker_match = re.search(r"--- PAGE (\d+) ---", chunk.page_content)
        current_page_number = last_seen_page_number
        if page_marker_match:
            current_page_number = int(page_marker_match.group(1))
            last_seen_page_number = current_page_number
        
        if cleaned_content:
            chunk.metadata = {"page_number": current_page_number if current_page_number else 1}
            chunk.page_content = cleaned_content
            final_chunks.append(chunk)
            
    return final_chunks

# --- RAG & AI Functions ---

def get_embedding_model():
    """Loads the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

def summarize_single_section(section_name: str, retriever, llm_model) -> str:
    """Retrieves context and generates summary for a specific section."""
    retrieval_prompt = SECTION_RETRIEVAL_PROMPTS.get(section_name)
    summary_instruction = SECTION_SUMMARY_PROMPTS.get(section_name)

    if not retrieval_prompt or not summary_instruction:
        return f"Configuration for section '{section_name}' not found."

    try:
        # Retrieve documents
        retrieved_docs = retriever.invoke(retrieval_prompt)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        if not context_text:
            return "No relevant information found in the document for this section."

        # Generate Summary
        response = llm_model.generate_content(
            f"Context from Research Paper:\n{context_text}\n\nInstruction: {summary_instruction}\n\nOutput format: Short, concise, engaging, with a 'Insight:' at the end."
        )
        return response.text
    except Exception as e:
        return f"Error processing {section_name}: {str(e)}"

# --- Translation & Audio Functions ---

def translate_text(text: str, target_lang_code: str) -> str:
    """Translates text using deep_translator."""
    if target_lang_code == 'en':
        return text
        
    try:
        translator = GoogleTranslator(source='auto', target=target_lang_code)
        # Split large text if necessary (Deep Translator handles some limits, but safety is good)
        if len(text) > 4500:
            text = text[:4500] + "... (truncated for translation safety)"
        return translator.translate(text)
    except Exception as e:
        return f"Translation Error: {e}"

def text_to_audio(text: str, lang_code: str):
    """Converts text to MP3 bytes in memory."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0) # Reset pointer to start of file
        return fp
    except Exception as e:
        print(f"Audio Error: {e}")
        return None