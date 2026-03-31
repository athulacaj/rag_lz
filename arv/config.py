
import os

DATA_PATH = "input"
DB_PATH = "chroma_db_arv"
# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Check if fine-tuned model exists, otherwise use base model
FINE_TUNED_MODEL_PATH = "fine_tuned_model"
if os.path.exists(FINE_TUNED_MODEL_PATH):
    EMBEDDING_MODEL_NAME = FINE_TUNED_MODEL_PATH
else:
    EMBEDDING_MODEL_NAME = "bge-m3"

MODEL_NAME = "llama3.2:3b"
COLLECTION_NAME = "resume_collection_arv"
