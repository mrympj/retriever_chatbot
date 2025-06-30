from chromadb import PersistentClient
import config

def search_chroma(query, adapter, threshold, count):
    client = PersistentClient(path=config.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    query_vec = adapter.embed(query).tolist()
    count = count
    results = collection.query(
        query_embeddings=[query_vec],
        include=config.CHROMA_INCLUDE,
    )

    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    seen_answers = set()
    matches = []

    for i in range(len(distances)):
        similarity = distances[i]
        answer = documents[i]
        if similarity >= threshold and answer not in seen_answers:
            seen_answers.add(answer)
            matches.append({
                "answer": answer,
                "question": metadatas[i].get("question", ""),
                "similarity": similarity
            })
            if len(matches) >= count:
                break

    return matches
