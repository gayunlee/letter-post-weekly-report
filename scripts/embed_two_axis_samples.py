"""2축 벡터 분류기용 샘플 임베딩 스크립트 (실험용)

vector_samples.json을 ChromaDB(chroma_db_two_axis)에 임베딩합니다.
기존 1축 chroma_db에는 영향을 주지 않습니다.

⚠️ 실제 파이프라인에는 사용하지 않음 (2026-02-19 결정)
- 벤치마크 결과: Topic 55.5% / Sentiment 45.3% (파인튜닝 대비 각각 14%p, 39%p 열세)
- 원인: 임베딩 유사도는 주제(semantic similarity)를 잡지, 감성(sentiment)을 구분하지 못함
  "투자 종목 불만"과 "투자 종목 감사"가 임베딩 공간에서 거의 동일한 위치
- 앙상블 효과도 기대 불가: 두 분류기의 정확도 격차가 너무 커서 보완이 아닌 정확도 하락 유발
- 결론: 2축 체계에서는 파인튜닝 모델만 프로덕션 사용, 벡터 분류기는 실험 기록으로 보존

사용법:
    python3 scripts/embed_two_axis_samples.py
"""
import json
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.chroma_store import ChromaVectorStore


def main():
    project_root = Path(__file__).parent.parent
    samples_file = project_root / "data" / "training_data" / "two_axis" / "vector_samples.json"
    persist_dir = str(project_root / "chroma_db_two_axis")

    print("=" * 60)
    print("2축 벡터 분류기 샘플 임베딩")
    print("=" * 60)

    # 샘플 로드
    with open(samples_file, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"\n샘플 수: {len(samples)}건")

    # 분포 확인
    topic_counts = Counter(s["topic"] for s in samples)
    sentiment_counts = Counter(s["sentiment"] for s in samples)
    combo_counts = Counter(f"{s['topic']}_{s['sentiment']}" for s in samples)

    print("\nTopic 분포:")
    for t, c in topic_counts.most_common():
        print(f"  {t}: {c}건")

    print("\nSentiment 분포:")
    for s, c in sentiment_counts.most_common():
        print(f"  {s}: {c}건")

    # Topic용 컬렉션
    print("\n\n[1/2] Topic 분류용 컬렉션 생성...")
    topic_store = ChromaVectorStore(
        collection_name="two_axis_topic",
        persist_directory=persist_dir,
    )

    # 기존 데이터 초기화
    if topic_store.collection.count() > 0:
        print(f"  기존 데이터 {topic_store.collection.count()}건 삭제 후 재생성")
        topic_store.reset()

    for i, sample in enumerate(samples):
        topic_store.add_content(
            content_id=f"topic_{i}",
            text=sample["text"][:500],
            metadata={"category": sample["topic"]},
        )
    print(f"  Topic 샘플 {len(samples)}건 임베딩 완료")

    # Sentiment용 컬렉션
    print("\n[2/2] Sentiment 분류용 컬렉션 생성...")
    sentiment_store = ChromaVectorStore(
        collection_name="two_axis_sentiment",
        persist_directory=persist_dir,
    )

    if sentiment_store.collection.count() > 0:
        print(f"  기존 데이터 {sentiment_store.collection.count()}건 삭제 후 재생성")
        sentiment_store.reset()

    for i, sample in enumerate(samples):
        sentiment_store.add_content(
            content_id=f"sentiment_{i}",
            text=sample["text"][:500],
            metadata={"category": sample["sentiment"]},
        )
    print(f"  Sentiment 샘플 {len(samples)}건 임베딩 완료")

    # 검증
    print(f"\n검증:")
    print(f"  Topic 컬렉션: {topic_store.collection.count()}건")
    print(f"  Sentiment 컬렉션: {sentiment_store.collection.count()}건")
    print(f"  저장 위치: {persist_dir}")

    # 간단한 테스트
    print("\n테스트 쿼리:")
    test_texts = [
        "감사합니다 덕분에 많이 배웠어요",
        "앱이 자꾸 튕겨요 불편합니다",
        "포트폴리오 비중 질문입니다",
    ]
    for text in test_texts:
        topic_results = topic_store.search_similar(text, n_results=3)
        sentiment_results = sentiment_store.search_similar(text, n_results=3)

        topic_votes = Counter()
        for r in topic_results:
            topic_votes[r["metadata"]["category"]] += 1

        sentiment_votes = Counter()
        for r in sentiment_results:
            sentiment_votes[r["metadata"]["category"]] += 1

        topic = topic_votes.most_common(1)[0][0]
        sentiment = sentiment_votes.most_common(1)[0][0]

        print(f"\n  \"{text[:30]}...\"")
        print(f"    → Topic: {topic}, Sentiment: {sentiment}")

    print(f"\n\n완료!")
    print(f"저장 위치: {persist_dir}")


if __name__ == "__main__":
    main()
