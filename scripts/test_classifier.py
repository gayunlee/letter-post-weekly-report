"""콘텐츠 분류 시스템 테스트 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier.content_classifier import ContentClassifier


def main():
    """콘텐츠 분류 테스트"""

    print("=" * 60)
    print("콘텐츠 분류 시스템 테스트")
    print("=" * 60)
    print()

    # 분류기 초기화
    print("✓ 분류기 초기화 중...")
    classifier = ContentClassifier()
    print(f"✓ 분류 카테고리: {list(classifier.CATEGORIES.keys())}\n")

    # BigQuery에서 샘플 데이터 가져오기
    print("BigQuery 연결 중...")
    client = BigQueryClient()
    query = WeeklyDataQuery(client)

    # 지난 주 게시글 샘플 조회 (최대 5건)
    start_date, end_date = query.get_last_week_range()
    print(f"📅 조회 기간: {start_date} ~ {end_date}\n")

    print("📝 샘플 게시글 조회 중...")
    posts = query.get_weekly_posts(start_date, end_date)

    if not posts:
        print("❌ 게시글이 없습니다.")
        return

    # 처음 5건만 테스트
    sample_posts = posts[:5]
    print(f"✓ 샘플 {len(sample_posts)}건 선택\n")

    print("=" * 60)
    print("분류 시작")
    print("=" * 60)
    print()

    # 각 게시글 분류
    for i, post in enumerate(sample_posts, 1):
        print(f"[{i}번째 게시글]")
        print(f"제목: {post.get('title', '제목 없음')}")

        # 본문 내용
        content = post.get('textBody') or post.get('body', '')
        if len(content) > 200:
            display_content = content[:200] + '...'
        else:
            display_content = content

        print(f"내용: {display_content}")
        print()

        # 분류 수행
        print("분류 중...")
        classification = classifier.classify_content(content)

        print(f"✓ 카테고리: {classification.get('category')}")
        print(f"  확신도: {classification.get('confidence')}")
        print(f"  이유: {classification.get('reason')}")
        print()
        print("-" * 60)
        print()

    print("=" * 60)
    print("✓ 분류 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
