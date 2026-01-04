"""벡터 스토어 테스트 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier.content_classifier import ContentClassifier
from src.vectorstore.chroma_store import ChromaVectorStore


def main():
    """벡터 스토어 테스트"""

    print("=" * 60)
    print("벡터 스토어 테스트")
    print("=" * 60)
    print()

    # 벡터 스토어 초기화
    print("✓ 벡터 스토어 초기화 중...")
    store = ChromaVectorStore(
        collection_name="test_contents",
        persist_directory="./chroma_db_test"
    )
    print("✓ 초기화 완료\n")

    # 기존 데이터 초기화
    print("기존 데이터 초기화 중...")
    store.reset()
    print("✓ 초기화 완료\n")

    # BigQuery에서 샘플 데이터 가져오기
    print("BigQuery 연결 중...")
    client = BigQueryClient()
    query = WeeklyDataQuery(client)

    start_date, end_date = query.get_last_week_range()
    print(f"📅 조회 기간: {start_date} ~ {end_date}\n")

    print("📝 샘플 게시글 조회 중...")
    posts = query.get_weekly_posts(start_date, end_date)

    if not posts:
        print("❌ 게시글이 없습니다.")
        return

    # 처음 10건만 테스트
    sample_posts = posts[:10]
    print(f"✓ 샘플 {len(sample_posts)}건 선택\n")

    # 분류 수행
    print("=" * 60)
    print("콘텐츠 분류 중...")
    print("=" * 60)
    print()

    classifier = ContentClassifier()
    classified_posts = classifier.classify_batch(
        sample_posts,
        content_field="textBody"
    )
    print(f"\n✓ {len(classified_posts)}건 분류 완료\n")

    # 벡터 스토어에 저장
    print("=" * 60)
    print("벡터 스토어에 저장 중...")
    print("=" * 60)
    print()

    # 텍스트 필드를 맞춰서 저장
    for post in classified_posts:
        if "textBody" in post:
            post["message"] = post["textBody"]  # message 필드로 복사

    added_count = store.add_contents_batch(
        classified_posts,
        id_field="_id",
        text_field="message"
    )
    print(f"✓ {added_count}건 저장 완료\n")

    # 통계 조회
    print("=" * 60)
    print("저장소 통계")
    print("=" * 60)
    print()

    stats = store.get_stats()
    print(f"전체 콘텐츠 수: {stats['total_count']}")
    print(f"컬렉션 이름: {stats['collection_name']}\n")

    print("카테고리별 통계:")
    for category, count in stats['category_stats'].items():
        print(f"  {category}: {count}건")

    print()

    # 유사 콘텐츠 검색 테스트
    print("=" * 60)
    print("유사 콘텐츠 검색 테스트")
    print("=" * 60)
    print()

    query_text = "포트폴리오 구성 질문"
    print(f"검색 쿼리: '{query_text}'\n")

    similar = store.search_similar(query_text, n_results=3)

    if similar:
        print(f"✓ 유사 콘텐츠 {len(similar)}건 발견:\n")
        for i, content in enumerate(similar, 1):
            print(f"[{i}번째 결과]")
            print(f"  ID: {content['id']}")
            print(f"  카테고리: {content['metadata'].get('category', '미분류')}")
            text = content['text']
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"  내용: {text}")
            if content.get('distance'):
                print(f"  유사도: {1 - content['distance']:.2f}")
            print()
    else:
        print("검색 결과가 없습니다.")

    print("=" * 60)
    print("✓ 벡터 스토어 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
