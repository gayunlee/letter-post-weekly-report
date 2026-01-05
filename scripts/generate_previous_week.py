"""전주 데이터 생성 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier.vector_classifier import VectorContentClassifier
from src.storage.data_store import ClassifiedDataStore


def main():
    print("=" * 60)
    print("📊 전주 데이터 생성")
    print("=" * 60)
    print()

    data_store = ClassifiedDataStore()
    prev_start, prev_end = WeeklyDataQuery.get_previous_week_range()

    print(f"📅 전주 기간: {prev_start} ~ {prev_end}")
    print()

    if data_store.exists(prev_start):
        print(f"✓ 전주 데이터가 이미 존재합니다.")
        return

    print("BigQuery 조회 및 분류 중...")
    client = BigQueryClient()
    query = WeeklyDataQuery(client)
    weekly_data = query.get_weekly_data(prev_start, prev_end)

    letters = weekly_data['letters']
    posts = weekly_data['posts']
    print(f"✓ 편지 {len(letters)}건, 게시글 {len(posts)}건 조회")

    if not letters and not posts:
        print("❌ 데이터 없음")
        return

    classifier = VectorContentClassifier()
    classified_letters = classifier.classify_batch(letters, "message") if letters else []
    classified_posts = classifier.classify_batch(posts, "textBody") if posts else []

    data_store.save_weekly_data(prev_start, prev_end, classified_letters, classified_posts)
    print(f"✓ 저장 완료: {prev_start}.json")
    print()


if __name__ == "__main__":
    main()
