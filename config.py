import os

os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

MODEL_PATH = r'../model6'
USE_FP16 = True

DENSE_VECS = 'dense_vecs'

CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "qa_collection"
CHROMA_INCLUDE = ["documents", "metadatas", "distances"]
CHROMA_ADD_CONFIG = {
    "metadata_key": "question",
    "id_prefix": "id_"
}

SEARCH_THRESHOLD_HIGH = 0
SEARCH_THRESHOLD_LOW = 0.6

MYSQL_CONFIG = {
    "host": "1.1.1.1",
    "user": "root",
    "password": "1234",
    "database": "chat_bot"
}

COLUMN_CONFIG = {
    "question": "question",
    "answer": "answer",
    "embedding": "embeddingbmge"
}

QUERY_CONFIG = {
    "select_query": "SELECT question, answer, embeddingbmge FROM questions WHERE tag = %s",
    "tag": "t"
}
