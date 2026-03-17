"""현재 classified_data_two_axis 기반으로 ChromaDB 재임베딩"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_PATH = Path("data/classified_data_two_axis/2026-02-09.json")
COLLECTION_NAME = "voc_demo_2026_02_09"
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
BATCH_SIZE = 100

def main():
    print(f"로딩: {DATA_PATH}")
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for item in data.get("letters", []):
        item["_type"] = "letter"
        items.append(item)
    for item in data.get("posts", []):
        item["_type"] = "post"
        items.append(item)

    # detail_tags 있는 것만
    items = [x for x in items if "detail_tags" in x]
    print(f"총 {len(items)}건 임베딩 대상")

    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path="chroma_db")

    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 '{COLLECTION_NAME}' 삭제")
    except Exception:
        pass

    col = client.create_collection(COLLECTION_NAME, embedding_function=ef,
                                   metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for i, item in enumerate(items):
        cls = item.get("classification", {})
        dt = item.get("detail_tags", {})
        master = re.sub(r"\d+$", "", item.get("masterName", "Unknown")).strip()
        content = (item.get("message") or item.get("textBody") or item.get("body", "")).strip()
        if not content:
            continue

        ids.append(str(item.get("_id", i)))
        docs.append(content[:512])
        metas.append({
            "type": item["_type"],
            "master": master,
            "topic": cls.get("topic", ""),
            "sentiment": cls.get("sentiment", ""),
            "category_tags": ", ".join(dt.get("category_tags", [])),
            "free_tags": ", ".join(dt.get("free_tags", [])),
            "summary": dt.get("summary", "")[:200],
        })

        if len(ids) >= BATCH_SIZE:
            col.add(ids=ids, documents=docs, metadatas=metas)
            print(f"  {i+1}/{len(items)} 완료", end="\r")
            ids, docs, metas = [], [], []

    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)

    print(f"\n완료: {col.count()}건 저장 → collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    main()
