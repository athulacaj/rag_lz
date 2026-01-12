import json
import os
import random
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import DATA_PATH, MODEL_NAME

# Output file for training data
OUTPUT_FILE = "training_data.json"

def generate_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory '{DATA_PATH}' does not exist.")
        return

    print("Loading documents from", DATA_PATH)
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")

    # Initialize LLM
    llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

    # Prompt for generating questions
    # We ask for a question that can be answered by the text.
    prompt_template = """
    You are an expert at creating training data for embedding models.
    Your task is to generate a single relevant query or question based on the following text chunk.
    The query should be something a user might ask to which the text chunk provides the answer.
    
    Text chunk:
    {text}
    
    Generate ONLY the question. Do not add any explanation, quotes, or distinct formatting.
    Question:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm

    training_data = []

    print("Generating questions for chunks...")
    # Limit number of chunks if too many, or just do all for now.
    # We'll do a simple progress bar
    for i, chunk in enumerate(tqdm(chunks)):
        content = chunk.page_content.strip()
        if len(content) < 50:  # Skip very short chunks
            continue
            
        try:
            response = chain.invoke({"text": content})
            question = response.content.strip()
            
            if question:
                training_data.append({
                    "anchor": question,
                    "positive": content
                })
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    print(f"Generated {len(training_data)} pairs.")
    
    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Training data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_data()
