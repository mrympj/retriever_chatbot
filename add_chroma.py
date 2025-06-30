from embeder import load_embedding_model, EmbeddingAdapter
import chromadb
import config
import uuid

model = load_embedding_model()
adapter = EmbeddingAdapter(model)

def add_qa_to_chroma(question: str, answer: str):
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    embedding = adapter.embed(question)

    unique_id = f"qa_{uuid.uuid4()}"

    collection.add(
        documents=[answer],
        metadatas=[{"question": question}],
        embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else embedding],
        ids=[unique_id]
    )

    print(f"Added QA with id: {unique_id}")
