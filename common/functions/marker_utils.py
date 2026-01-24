from marker.converters.pdf import PdfConverter
import json
from pydantic import BaseModel
from marker.renderers.html import HTMLRenderer
from marker.models import create_model_dict
import base64
import io
import os
import pytesseract
import re  # <--- New Import
from typing import List
from langchain_core.documents import Document
from marker.config.parser import ConfigParser

# 1. Define configuration (Optional but recommended for control)
config = {
    "output_format": "markdown",  # Markdown handles tables best
    "paginate_output": True,      # Helps separate tables per page
    "disable_multiprocessing": False
}
config_parser = ConfigParser(config)

# Load models
model_dict = create_model_dict()

# Create converter
converter = PdfConverter(
    artifact_dict=model_dict,
    config=config_parser.generate_config_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer()
)

def get_md_data_from_marker(path, useOcr=True, keep_table_structure=False):
    # Convert PDF
    rendered = converter(path)

    # Process images with OCR and append text to markdown
    content=rendered.markdown
    print(f"Processing {len(rendered.images)} images for OCR...")
    if(useOcr and not keep_table_structure):
        for img_name, img in rendered.images.items():
            try:
                text = pytesseract.image_to_string(img)
                # Check if text contains at least 3 alphabetic characters to avoid noise
                if len(re.findall(r'[a-zA-Z]', text)) >= 3:
                    content=content.replace(f"![]({img_name})",f"\n{text.strip()}\n")
            except Exception as e:
                print(f"Error OCRing {img_name}: {str(e)}")

    return content




def load_documents_with_marker(data_path: str,is_markdown: bool = True,  skip_condition_func: callable = lambda x: False,keep_table_structure: bool = False,) -> List[Document]:


    if not os.path.exists(data_path):
        print(f"Directory '{data_path}' does not exist.")
        return []

    documents = []

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
            if skip_condition_func(filename):
                print(f"Skipping '{filename}'...")
                continue
            try:
                print(f"Converting '{filename}'...")
                content=get_md_data_from_marker(file_path, keep_table_structure=keep_table_structure)
                # Export to markdown to preserve structure
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path}
                ))
            except Exception as e:
                print(f"Error converting {filename}: {e}")

    if not documents:
        print("No documents found.")
        return []
    
    print(f"Loaded {len(documents)} documents.")
    return documents
