import argparse
import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config import DATA_PATH, DB_PATH, MODEL_NAME



PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Question: {question}
"""

def query_rag(query_text):
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)
    
    try:
        db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Did you run ingest.py first?")
        return

    # Search for relevant documents
    results = db.similarity_search_with_score(query_text, k=3)
    
    if not results:
        print("No relevant context found.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print(f"\nGeneratin answer using {MODEL_NAME}...\n")
    model = ChatOllama(model=MODEL_NAME)
    response_text = model.invoke(prompt)

    print("Response:")
    print(response_text.content)
    
    print("\nSources:")
    for doc, _score in results:
        print(f"- {doc.metadata.get('source', 'Unknown')}")

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python query.py \"Your question here\"")
    #     return
        
    # query_text = sys.argv[1]
    query_text="What is the best candidate for backend role?"
    query_rag(query_text)

if __name__ == "__main__":
    main()
