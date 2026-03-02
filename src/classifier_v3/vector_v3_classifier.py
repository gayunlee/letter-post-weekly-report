"""Opus 라벨링 데이터 기반 KNN 벡터 분류기 (v3)

기존 VectorContentClassifier(v1)과 동일한 k-NN 가중 투표 방식.
벡터 인덱스는 data/vectorstore_v3/에 별도 persist (기존 chroma_db/와 격리).
5분류/4분류 모두 지원 (valid_topics 파라미터로 제어).

임베딩 모델: jhgan/ko-sroberta-multitask (ChromaVectorStore와 동일)
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

from ..vectorstore.chroma_store import ChromaVectorStore
from .taxonomy import TOPICS


def preprocess_text(text: str) -> str:
    """분류 전 텍스트 전처리: 노이즈 제거 및 정규화"""
    if not text:
        return text
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[■★●◆◇▶▷►▲△▽▼○◎□☆♥♡♣♠♦]', '', text)
    return text.strip()


VALID_TOPICS_5 = set(TOPICS)
VALID_TOPICS_4 = {"운영 피드백", "서비스 피드백", "콘텐츠·투자", "기타"}


class VectorV3Classifier:
    """v3 벡터 KNN 분류기 (5분류/4분류 겸용)

    Opus 라벨링 데이터(labeled_all.json)를 벡터화하여
    k-NN 유사도 가중 투표로 분류.
    """

    def __init__(
        self,
        persist_dir: str = "data/vectorstore_v3",
        collection_name: str = "v3_topic_guide",
        k: int = 7,
        valid_topics: set = None,
    ):
        self.k = k
        self.persist_dir = persist_dir
        self.valid_topics = valid_topics or VALID_TOPICS_5
        self.store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

    def build_index(self, labeled_data_path: str = "data/training_data/v3/labeled_all.json"):
        """Opus 라벨링 데이터에서 벡터 인덱스 구축

        필터: v3_confidence >= 0.7, 텍스트 >= 20자
        """
        path = Path(labeled_data_path)
        if not path.exists():
            raise FileNotFoundError(f"라벨링 데이터 없음: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # 기존 컬렉션 리셋
        self.store.reset()

        ids = []
        documents = []
        metadatas = []
        skipped = Counter()

        for item in data:
            text = item.get("text", "").strip()
            topic = item.get("v3_topic", "")
            confidence = item.get("v3_confidence", 0)

            if confidence < 0.7:
                skipped["low_confidence"] += 1
                continue
            if len(text) < 20:
                skipped["short_text"] += 1
                continue
            if topic not in self.valid_topics:
                skipped["invalid_topic"] += 1
                continue

            doc_id = item.get("_id", f"v3_{len(ids)}")
            processed = preprocess_text(text)[:500]

            ids.append(doc_id)
            documents.append(processed)
            metadatas.append({"category": topic})

        # 배치 추가
        BATCH = 500
        for start in range(0, len(ids), BATCH):
            end = min(start + BATCH, len(ids))
            self.store.collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        # 통계 출력
        topic_counts = Counter(m["category"] for m in metadatas)
        print(f"\n  벡터 인덱스 구축 완료: {len(ids)}건")
        print(f"  스킵: {dict(skipped)}")
        print(f"  카테고리별:")
        for topic in sorted(self.valid_topics):
            print(f"    {topic}: {topic_counts.get(topic, 0)}건")

        return {
            "total": len(ids),
            "skipped": dict(skipped),
            "by_topic": dict(topic_counts),
        }

    def classify_content(self, text: str) -> Dict[str, Any]:
        """단일 텍스트 분류 → {topic, confidence, method}"""
        if not text or len(text.strip()) == 0:
            return {"topic": "기타", "topic_confidence": 0.0, "method": "empty"}

        text = preprocess_text(text)
        similar = self.store.search_similar(
            query_text=text[:500],
            n_results=self.k,
        )

        return self._classify_from_neighbors(similar)

    def _classify_from_neighbors(self, similar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """k-NN 이웃 목록에서 가중 투표"""
        if not similar:
            return {"topic": "기타", "topic_confidence": 0.0, "method": "none"}

        vote_weights = Counter()
        for match in similar:
            category = match["metadata"].get("category", "기타")
            distance = match.get("distance", 1.0)
            similarity = max(0.0, 1.0 - distance)
            vote_weights[category] += similarity

        if not vote_weights:
            return {"topic": "기타", "topic_confidence": 0.0, "method": "none"}

        best = vote_weights.most_common(1)[0][0]
        total = sum(vote_weights.values())
        confidence = vote_weights[best] / total if total > 0 else 0.0

        return {
            "topic": best,
            "topic_confidence": round(confidence, 4),
            "method": "vector_v3",
        }

    def classify_batch(
        self,
        items: List[Dict[str, Any]],
        content_field: str = "message",
        batch_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """배치 분류"""
        results = [None] * len(items)

        valid_indices = []
        valid_texts = []
        for i, item in enumerate(items):
            text = item.get(content_field, "")
            if not text or len(text.strip()) == 0:
                result = item.copy()
                result["classification"] = {
                    "topic": "기타",
                    "topic_confidence": 0.0,
                    "method": "empty",
                }
                results[i] = result
            else:
                valid_indices.append(i)
                valid_texts.append(preprocess_text(text)[:500])

        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]

            batch_neighbors = self.store.search_similar_batch(
                query_texts=batch_texts,
                n_results=self.k,
            )

            for idx, neighbors in zip(batch_indices, batch_neighbors):
                classification = self._classify_from_neighbors(neighbors)
                result = items[idx].copy()
                result["classification"] = classification
                results[idx] = result

            processed = min(batch_end, len(valid_texts))
            print(f"  진행: {processed}/{len(valid_texts)} 완료")

        return results
