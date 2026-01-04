"""통계 분석 테스트 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier.content_classifier import ContentClassifier
from src.reporter.analytics import WeeklyAnalytics
import json


def main():
    """통계 분석 테스트"""

    print("=" * 60)
    print("통계 분석 시스템 테스트")
    print("=" * 60)
    print()

    # BigQuery에서 데이터 가져오기
    print("BigQuery 연결 중...")
    client = BigQueryClient()
    query = WeeklyDataQuery(client)

    start_date, end_date = query.get_last_week_range()
    print(f"📅 조회 기간: {start_date} ~ {end_date}\n")

    print("📝 데이터 조회 중...")
    weekly_data = query.get_weekly_data(start_date, end_date)
    letters = weekly_data['letters']
    posts = weekly_data['posts'][:50]  # 테스트를 위해 50건만

    print(f"✓ 편지글 {len(letters)}건, 게시글 {len(posts)}건 조회\n")

    if not posts and not letters:
        print("❌ 데이터가 없습니다.")
        return

    # 분류 수행 (게시글만)
    print("=" * 60)
    print("콘텐츠 분류 중...")
    print("=" * 60)
    print()

    classifier = ContentClassifier()

    if posts:
        print(f"게시글 {len(posts)}건 분류 중...")
        classified_posts = classifier.classify_batch(
            posts,
            content_field="textBody"
        )
        print(f"✓ 게시글 분류 완료\n")
    else:
        classified_posts = []

    if letters:
        print(f"편지글 {len(letters)}건 분류 중...")
        classified_letters = classifier.classify_batch(
            letters,
            content_field="message"
        )
        print(f"✓ 편지글 분류 완료\n")
    else:
        classified_letters = []

    # 통계 분석
    print("=" * 60)
    print("통계 분석 중...")
    print("=" * 60)
    print()

    analytics = WeeklyAnalytics()
    stats = analytics.analyze_weekly_data(
        classified_letters,
        classified_posts
    )

    # 전체 통계 출력
    print("📊 전체 통계")
    print("-" * 60)
    total = stats["total_stats"]["this_week"]
    print(f"편지글: {total['letters']}건")
    print(f"게시글: {total['posts']}건")
    print(f"총합: {total['total']}건")
    print()

    # 카테고리별 통계 출력
    print("📊 카테고리별 통계")
    print("-" * 60)
    for category, count in stats["category_stats"].items():
        print(f"{category}: {count}건")
    print()

    # 마스터별 통계 출력 (상위 5개)
    print("📊 마스터별 통계 (상위 5개)")
    print("-" * 60)
    master_stats = stats["master_stats"]

    # 총 건수로 정렬
    sorted_masters = sorted(
        master_stats.items(),
        key=lambda x: x[1]["this_week"]["total"],
        reverse=True
    )

    for i, (master_id, data) in enumerate(sorted_masters[:5], 1):
        this_week = data["this_week"]
        print(f"\n[{i}. Master ID: {master_id}]")
        print(f"  편지: {this_week['letters']}건")
        print(f"  게시글: {this_week['posts']}건")
        print(f"  총합: {this_week['total']}건")

        # 카테고리 분포
        if data["categories"]:
            print(f"  카테고리:")
            for cat, count in data["categories"].items():
                print(f"    - {cat}: {count}건")

    # 서비스 피드백 출력
    print("\n" + "=" * 60)
    print("📢 서비스 피드백")
    print("=" * 60)
    feedbacks = stats["service_feedbacks"]

    if feedbacks:
        print(f"\n총 {len(feedbacks)}건의 서비스 피드백 발견:\n")
        for i, feedback in enumerate(feedbacks[:5], 1):
            print(f"[{i}번째 피드백]")
            print(f"  유형: {feedback['type']}")
            if feedback.get('title'):
                print(f"  제목: {feedback['title']}")
            print(f"  내용: {feedback['content'][:100]}...")
            print(f"  이유: {feedback['reason']}")
            print()
    else:
        print("\n서비스 피드백이 없습니다.")

    print("\n" + "=" * 60)
    print("✓ 통계 분석 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
