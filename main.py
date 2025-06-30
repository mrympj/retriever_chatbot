from fastapi import FastAPI
from pydantic import BaseModel
from embeder import load_embedding_model, EmbeddingAdapter
from retriever.retriever_chatbot.chroma_manager import search_chroma
import config
import chromadb
import uuid
import ast

app = FastAPI()

model = load_embedding_model()
adapter = EmbeddingAdapter(model)

class SimpleQuery(BaseModel):
    text: str
    count: int = 5

class QAItem(BaseModel):
    question: str
    answer: str


@app.post("/search/")
def search(query: SimpleQuery):
    matches = search_chroma(
        query.text,
        adapter,
        threshold=config.SEARCH_THRESHOLD_HIGH,
        count=query.count
    )

    if matches:
        return {
            "message": f"{len(matches)} پاسخ پیدا شد.",
            "matches": matches
        }

    matches_low = search_chroma(
        query.text,
        adapter,
        threshold=config.SEARCH_THRESHOLD_LOW,
        count=query.count
    )

    if matches_low:
        return {
            "message": f"هیچ نتیجه‌ای با آستانه بالا نبود، اما {len(matches_low)} نتیجه با آستانه پایین‌تر یافت شد.",
            "matches": matches_low
        }

    return {
        "message": "هیچ نتیجه‌ای پیدا نشد.",
        "matches": []
    }

@app.post("/add_qa/")
def add_qa(item: QAItem):
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)

    embedding = adapter.embed(item.question)
    embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding

    unique_id = f"qa_{uuid.uuid4()}"

    collection.add(
        documents=[item.answer],
        metadatas=[{"question": item.question}],
        embeddings=[embedding_list],
        ids=[unique_id]
    )

    return {"message": "سوال و جواب با موفقیت اضافه شد.", "id": unique_id}

@app.post("/get_context/")
def get_context(query: SimpleQuery):
    matches = search_chroma(
        query.text,
        adapter,
        threshold=config.SEARCH_THRESHOLD_HIGH,
        count=query.count
    )

    if not matches:
        matches = search_chroma(
            query.text,
            adapter,
            threshold=config.SEARCH_THRESHOLD_LOW,
            count=query.count
        )
        if not matches:
            return {"message": "هیچ نتیجه‌ای پیدا نشد.", "context": ""}

    answers = []
    for match in matches:
        try:
            parsed_answers = ast.literal_eval(match['answer'])
            if isinstance(parsed_answers, list):
                answers.extend([a.strip() for a in parsed_answers])
            else:
                answers.append(str(parsed_answers).strip())
        except Exception:
            answers.append(str(match['answer']).strip())

    context = " ".join(answers)

    return {
        "message": f"{len(matches)} پاسخ پیدا شد.",
        "context": context
    }

@app.post("/get_prompt/")
def get_prompt(data: SimpleQuery):
    matches = search_chroma(
        data.text,
        adapter,
        threshold=config.SEARCH_THRESHOLD_HIGH,
        count=data.count
    )

    if not matches:
        matches = search_chroma(
            data.text,
            adapter,
            threshold=config.SEARCH_THRESHOLD_LOW,
            count=data.count
        )
        if not matches:
            context = []
        else:
            context = [match['answer'] for match in matches]
    else:
        context = [match['answer'] for match in matches]

    import ast
    context_clean = []
    for c in context:
        try:
            parsed = ast.literal_eval(c)
            if isinstance(parsed, list):
                context_clean.extend([x.strip() for x in parsed])
            else:
                context_clean.append(str(parsed).strip())
        except:
            context_clean.append(str(c).strip())

    full_context = context_clean if context_clean else [""]

    prompt = f"""<start_of_turn>user

Guidelines:

Question:
{data.text}
Context:
{full_context}
<end_of_turn>"""

    return {
        "message": "پرامپت ساخته شد.",
        "prompt": prompt
    }

