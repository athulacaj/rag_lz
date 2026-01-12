import argparse
import sys
from langchain_chroma import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from config import DATA_PATH, DB_PATH, MODEL_NAME, EMBEDDING_MODEL_NAME
import torch
# from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama




PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Question: {question}
"""

def query_rag(query_text):
    # embedding_function = OllamaEmbeddings(model=MODEL_NAME)
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
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
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # llm = HuggingFacePipeline.from_model_id(
    #     model_id=MODEL_NAME,
    #     task="text-generation",
    #     pipeline_kwargs={
    #         "max_new_tokens": 512,
    #         "do_sample": True,
    #         "temperature": 0.1,
    #     },
    #     device=device,
    # )
    
    # chat_model = ChatHuggingFace(llm=llm)
    # response_text = chat_model.invoke(prompt)
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
    query_text="give me the devolpers from kannur"
    query_rag(query_text)

if __name__ == "__main__":
    main()
