import os
import sys

# Add parent directory to path to allow importing config and functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import pickle
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import DATA_PATH, DB_PATH, EMBEDDING_MODEL_NAME, COLLECTION_NAME

from functions.gemini_utils import analyze_image_with_gemini
from PIL import Image
import io
import time

# --- Constants ---

SECTION_HEADERS = [
    "PROFILE", "SUMMARY", "OBJECTIVE",
    "SKILLS", "TECHNICAL SKILLS",
    "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT HISTORY",
    "PROJECTS", "ACADEMIC PROJECTS",
    "EDUCATION", "QUALIFICATIONS",
    "HOBBIES", "INTERESTS", "CERTIFICATIONS"
]

HEADERS_PATTERN = "|".join(map(re.escape, SECTION_HEADERS))
SECTION_REGEX = re.compile(f"(?:^|\\n)\\s*({HEADERS_PATTERN})\\s*(?:\\n|$)", re.IGNORECASE)


# --- Helper Functions ---

def ensure_directory_exists(path: str) -> bool:
    """Checks if a directory exists."""
    if not os.path.exists(path):
        return False
    return True

def load_documents(data_path: str) -> List[Document]:
    """Loads PDF documents from the specified directory."""
    print(f"Loading PDFs from '{data_path}'...")
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    if not documents:
        print("No documents found.")
        return []
    print(f"Loaded {len(documents)} document pages.")
    return documents

def load_documents_with_markitdown(data_path: str) -> List[Document]:
    """
    Loads documents (PDF, PPTX, DOCX, XLSX) from the directory using Microsoft MarkItDown.
    """
    try:
        from markitdown import MarkItDown
    except ImportError:
        raise ImportError("MarkItDown is not installed. Please install it using `pip install markitdown`.")

    print(f"Loading documents from '{data_path}' using MarkItDown...")
    documents = []
    
    # Initialize MarkItDown once
    md = MarkItDown()

    if not os.path.exists(data_path):
        print(f"Directory '{data_path}' does not exist.")
        return []

    # MarkItDown supports multiple formats, not just PDF
    supported_extensions = ('.pdf', '.pptx', '.docx', '.xlsx', '.html')

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        
        # Check if it is a file and has a supported extension
        if os.path.isfile(file_path) and filename.lower().endswith(supported_extensions):
            try:
                print(f"Converting '{filename}'...")
                
                # Convert the file
                # MarkItDown automatically detects format based on extension
                result = md.convert(file_path)
                
                # .text_content returns the converted Markdown string
                content = result.text_content
                
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "filename": filename
                        }
                    ))
            except Exception as e:
                print(f"Error converting {filename}: {e}")

    if not documents:
        print("No documents found.")
        return []
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def load_documents_with_docling(data_path: str,is_markdown: bool = True) -> List[Document]:
    """Loads PDF documents from the specified directory using Docling."""
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
    except ImportError:
        raise ImportError("Docling is not installed. Please install it using `pip install docling`.")

    print(f"Loading PDFs from '{data_path}' using Docling...")
    documents = []
    
    # Configure Pipeline Options to enforce OCR
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_picture_description=True
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if not os.path.exists(data_path):
        print(f"Directory '{data_path}' does not exist.")
        return []

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
            try:
                print(f"Converting '{filename}'...")
                result = converter.convert(file_path)
                # Export to markdown to preserve structure
                if is_markdown:
                    content = result.document.export_to_markdown()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path}
                    ))
                else:
                    content = result.document.export_to_dict()
                    documents.append({
                        "page_content":content,
                        "metadata":{"source": file_path}
                    })
            except Exception as e:
                print(f"Error converting {filename}: {e}")

    if not documents:
        print("No documents found.")
        return []
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def load_documents_with_docling_tesseract(data_path: str, is_markdown: bool = True) -> List[Document]:
    """Loads PDF documents from the specified directory using Docling with Tesseract OCR."""
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions,TesseractCliOcrOptions,TableStructureOptions
        from docling.datamodel.base_models import InputFormat
    except ImportError:
        raise ImportError("Docling is not installed. Please install it using `pip install docling`.")

    print(f"Loading PDFs from '{data_path}' using Docling with Tesseract...")
    documents = []
    
    # Configure Pipeline Options to enforce OCR using Tesseract
    ocr_options = TesseractCliOcrOptions(lang=["eng"])
    pipeline_options = PdfPipelineOptions(
        do_ocr=True, ocr_options=ocr_options
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if not os.path.exists(data_path):
        print(f"Directory '{data_path}' does not exist.")
        return []

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
            try:
                print(f"Converting '{filename}'...")
                result = converter.convert(file_path)
                # Export to markdown to preserve structure
                if is_markdown:
                    content = result.document.export_to_markdown()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path}
                    ))
                else:
                    content = result.document.export_to_dict()
                    documents.append({
                        "page_content":content,
                        "metadata":{"source": file_path}
                    })
            except Exception as e:
                print(f"Error converting {filename}: {e}")

    if not documents:
        print("No documents found.")
        return []
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def load_documents_with_docling_and_gemini(data_path: str) -> List[Document]:
    """
    Extracts images from PDF documents using Docling, sends them to Gemini for analysis,
    and creates a Markdown file with the content.
    Returns a list of paths to the created Markdown files.
    """
    documents = []

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
    except ImportError:
        raise ImportError("Docling is not installed. Please install it using `pip install docling`.")

    print(f"Extracting images from '{data_path}' using Docling...")
    
    # Configure Pipeline Options to generate page images
    # We enable both page images (for context if needed) and picture images (for extraction)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True


    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if not os.path.exists(data_path):
        print(f"Directory '{data_path}' does not exist.")
        return []


    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
            try:
                print(f"Processing '{filename}'...")
                result = converter.convert(file_path)
                
                # Accessing figures/pictures from the Docling document
                doc = result.document
                
                # Start with the text content of the document
                md_content = doc.export_to_markdown()

                # Note: Depending on docling version, pictures might be directly under document or pages.
                # We check the top-level 'pictures' attribute first.
                
                pictures_found = False
                if hasattr(doc, 'pictures') and doc.pictures:
                    print(f"Found {len(doc.pictures)} pictures in {filename}.")
                    pictures_found = True
                    for i, picture in enumerate(doc.pictures):
                        if hasattr(picture, 'image') and picture.image:
                            print(f"Analyzing picture {i+1}...")
                            
                            # Handle Docling ImageRef object
                            image_to_analyze = picture.image
                            if hasattr(image_to_analyze, 'pil_image') and image_to_analyze.pil_image:
                                image_to_analyze = image_to_analyze.pil_image
                            
                            # It's a PIL Image
                            description = analyze_image_with_gemini(
                                image_to_analyze, 
                                "Find only the text in the image and transcribe it."
                            )
                            print(f"--- Analysis for Picture {i+1} ---")
                            print(description)
                                
                            print("-" * 30)
                            pattern = r"no\s+text\s+in\s+(this|the)\s+image"
                            if re.search(pattern, description, re.IGNORECASE):
                                # Append the analysis to the markdown content
                                #  replace the first !-- image --> with the analysis
                                md_content = md_content.replace("<!-- image -->", "",1)
                            else:
                                md_content = md_content.replace("<!-- image -->", description,1)
                            # wait for 2 seconds
                            time.sleep(2)
                
                # Fallback: Check pages for images if no high-level pictures found
                # (Some versions/pdfs might not detect 'figures' but render 'page_images')
                if not pictures_found:
                    print("No distinct figures found. Checking page images...")
                    # This might be too much if we analyze every page, but good for testing.
                    # Let's just limit to the first page for now if we fall back to page images,
                    # or maybe just log it.
                    # Implementing a simple check on the first page.
                    if hasattr(doc, 'pages') and doc.pages:
                        first_page = next(iter(doc.pages.values())) # pages is usually a dict/list
                        if hasattr(first_page, 'image') and first_page.image:
                             print("Analyzing first page image as fallback...")
                             description = analyze_image_with_gemini(
                                 first_page.image.pil_image, # Docling page image wrapper
                                 "Describe this page layout and content."
                             )
                             print(f"--- Analysis for Page 1 ---")
                             print(description)
                             print("-" * 30)
                             
                             md_content += f"\n\n### Page 1 Analysis\n{description}\n"

                # Save the markdown file
                documents.append(Document(
                    page_content=md_content,
                    metadata={"source": file_path}
                ))
                
                print(f"Created Markdown file: {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return documents



def split_by_headers(documents: List[Document]) -> List[Document]:
    """
    Splits documents based on Resume headers.
    Returns new document chunks with 'section' metadata.
    """
    section_chunks = []

    for doc in documents:
        content = doc.page_content
        cv_data = extract_sections(content)
        for section_name, content in cv_data.items():
            if len(content) == 0:
                continue # Skip empty sections or pure image placeholders
            section_chunks.append(Document(
                page_content=content,
                metadata={
                    **doc.metadata, 
                    "section": section_name
                }
            ))
    return section_chunks

def generate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Generates unique IDs for each chunk based on doc_id and section."""
    for i, chunk in enumerate(chunks):
        doc_id = chunk.metadata.get("doc_id", "unknown")
        section = chunk.metadata.get("section", "unknown")
        chunk.metadata["chunk_id"] = f"{doc_id}_{section}_{i}"
    return chunks

def save_chunks_for_bm25(chunks: List[Document], db_path: str):
    """Saves the chunks to a pickle file for later use."""
    chunks_file = os.path.join(db_path, "chunks.pkl")
    print(f"Saving chunks for BM25 to '{chunks_file}'...")
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

def reset_vector_db(db_path: str):
    """Deletes the existing vector database directory if it exists."""
    if os.path.exists(db_path):
        print(f"Removing existing database at '{db_path}'...")
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

def create_and_persist_db(chunks: List[Document], db_path: str, collection_name: str, model_name: str,ids:List[str]):
    """Initializes the embedding model and creates the Chroma vector store."""
    if "gemini" in model_name:
        create_and_persist_db_gemini(chunks, db_path, collection_name, model_name, ids)
        return
    print(f"Initializing embedding model '{model_name}'...")
    embeddings = OllamaEmbeddings(model=model_name)

    print(f"Creating vector store in '{db_path}'...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_path,
        collection_name=collection_name,
        ids=ids
    )

    print("Vector store created successfully.")

def create_and_persist_db_gemini(chunks: List[Document], db_path: str, collection_name: str, model_name: str, ids: List[str]):
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError:
        GoogleGenerativeAIEmbeddings = None
    
    """Initializes the Gemini embedding model and creates the Chroma vector store."""
    print(f"Initializing Gemini embedding model '{model_name}'...")
    
    if GoogleGenerativeAIEmbeddings is None:
         raise ImportError("langchain_google_genai is not installed. Please install it with `pip install langchain-google-genai`.")

    # Ensure GEMINI_KEY is loaded
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise ValueError("GEMINI_KEY not found in environment variables.")

    # Filter out empty documents to avoid API errors
    valid_chunks = []
    valid_ids = []
    for i, chunk in enumerate(chunks):
        if chunk.page_content and chunk.page_content.strip():
            valid_chunks.append(chunk)
            if ids and i < len(ids):
                valid_ids.append(ids[i])
    
    if not valid_chunks:
        print("No valid content to embed. Skipping.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    print(f"Creating vector store in '{db_path}'...")
    Chroma.from_documents(
        documents=valid_chunks, 
        embedding=embeddings, 
        persist_directory=db_path,
        collection_name=collection_name,
        ids=valid_ids if valid_ids else None
    )

    print("Vector store created successfully.")

# --- Improved Section Extraction (Ported from JS) ---

CV_HEADING_PATTERNS = {
    "summary": ['summary', 'professional summary', 'profile', 'career summary', 'about me', 'objective'],
    "skills": ['skills', 'key skills', 'technical skills', 'professional skills', 'core skills', 'competencies', 'expertise'],
    "experience": ['experience', 'work experience', 'professional experience', 'employment history', 'career history', 'work history'],
    "education": ['education', 'academic background', 'educational qualifications', 'qualifications'],
    "projects": ['projects', 'personal projects', 'academic projects', 'professional projects'],
    "certifications": ['certifications', 'licenses', 'certified courses', 'certificates', 'certificate'],
    "achievements": ['achievements', 'accomplishments', 'awards', 'honors'],
    "interests": ['interests', 'hobbies', 'activities', 'extracurricular activities'],
    "languages": ['languages', 'language proficiency'],
    "publications": ['publications', 'research', 'papers'],
    "references": ['references', 'referees'],
    "personal": ['personal details', 'personal information', 'contact details']
}

def escape_regex(s):
    return re.escape(s)

def spaced_word(word):
    return r"\s*".join(list(word))

def spaced_phrase(phrase):
    words = phrase.strip().split()
    return r"\s+".join([spaced_word(w) for w in words])

def build_heading_regex(variants):
    patterns = []
    for v in variants:
        escaped = escape_regex(v)
        patterns.append(escaped)
        if re.match(r"^[a-zA-Z\s]+$", v):
            patterns.append(spaced_phrase(v))
            
    # pattern_str = f"^\\s*(?:#{{1,6}}\\s*)?(?:{'|'.join(patterns)})\\s*[:\\-]?\\s*$"
    pattern_str=f"^\\s*(?:#{1,6}\\s*)?[\\*_]*(?:${'|'.join(patterns)})[\\s\\*_\\.\\-:]*$"
    return re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)

def detect_cv_headings(cv_text):
    lines = cv_text.splitlines()
    headings = []
    
    for section, variants in CV_HEADING_PATTERNS.items():
        regex = build_heading_regex(variants)
        for i, line in enumerate(lines):
            if regex.search(line):
                 headings.append({
                     "section": section,
                     "line": line.strip(),
                     "lineNumber": i
                 })
    
    headings.sort(key=lambda x: x["lineNumber"])
    return headings

def extract_sections(cv_text):
    headings = detect_cv_headings(cv_text)
    lines = cv_text.splitlines()
    sections = {}
    
    if headings and headings[0]["lineNumber"] > 0:
        sections["general"] = "\n".join(lines[:headings[0]["lineNumber"]]).strip()
        
    for i in range(len(headings)):
        start = headings[i]["lineNumber"] + 1
        if i + 1 < len(headings):
            end = headings[i+1]["lineNumber"]
        else:
            end = len(lines)
            
        sections[headings[i]["section"]] = "\n".join(lines[start:end]).strip()
        
    return sections




if __name__ == "__main__":
    # print(extract_sections("data"))
    cv="""  #### **Professional Summary**
    A final year B.Tech...

    #### **Education**
    # **Cochin University...**
    #### **P R O J E C T S:---**
    new project
    # **Experience**
    # **Project Intern at N-OMS**
    - Contributed to the N-OMS...
    """;

    print(detect_cv_headings(cv))

