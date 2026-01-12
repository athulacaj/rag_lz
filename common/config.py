import os

DATA_PATH = "input"
DB_PATH = "vector_db_n"
# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Check if fine-tuned model exists, otherwise use base model
FINE_TUNED_MODEL_PATH = "fine_tuned_model"
if os.path.exists(FINE_TUNED_MODEL_PATH):
    EMBEDDING_MODEL_NAME = FINE_TUNED_MODEL_PATH
else:
    EMBEDDING_MODEL_NAME = "nomic-embed-text"

MODEL_NAME = "llama3.2:3b"