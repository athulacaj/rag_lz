from config import MODEL_NAME,PARSER,DB_NAME,DB_PATH,EMBEDDING_MODEL_NAME, COLLECTION_NAME
import os
import json
import functions.database_utils as db_utils
from langchain_core.documents import Document
from functions.ingestion_utils import (
    create_and_persist_db,
    reset_vector_db
)


def get_connection():
    return db_utils.get_db_connection(DB_NAME)


def create_tables():
    with get_connection() as conn:
        db_utils.create_resume_tables(conn)


def insert_data():
    path=os.path.join("processed/json/"+PARSER)
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks=[]
                chunk_ids=[]
                # key in data is "structured_data"
                for section in data:
                    if(section!="structured_data"):
                        email=data["structured_data"]["general"]["email"]
                        id=email+"_"+section
                        chunk=Document(
                            page_content=data[section],
                            metadata={
                                "source":filename,
                                "section":section,
                                "email":email
                            }
                        )
                        
                        chunk_ids.append(id)
                        chunks.append(chunk)
                with get_connection() as conn:
                    db_utils.insert_resume_data(conn,data["structured_data"])
                create_and_persist_db(
                    chunks=chunks,
                    db_path=DB_PATH,
                    collection_name=COLLECTION_NAME,
                    model_name=EMBEDDING_MODEL_NAME,
                    ids=chunk_ids
                )

    # with get_connection() as conn:
    #     db_utils.insert_resume(conn,resume)


def main():
    reset_vector_db(DB_PATH)
    create_tables()
    insert_data()


if __name__ == "__main__":
    main()