"""주간 리포트 생성 테스트 (샘플 데이터)"""
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier.content_classifier import ContentClassifier
from src.vectorstore.chroma_store import ChromaVectorStore
from src.reporter.analytics import WeeklyAnalytics
from src.reporter.report_generator import ReportGenerator

# 샘플 크기 설정
SAMPLE_SIZE = 30


def main():
    """주간 리포트 생성 (샘플 데이터)"""

    print("=" * 60)
    print(f"📊 주간 리포트 자동 생성 시스템 (샘플 {SAMPLE_SIZE}건)")
    print("=" * 60)
    print()

    # 1. BigQuery 데이터 조회
    print("1️⃣  BigQuery 데이터 조회")
    print("-" * 60)

    client = BigQueryClient()
    query = WeeklyDataQuery(client)

    start_date, end_date = query.get_last_week_range()
    print(f"📅 조회 기간: {start_date} ~ {end_date}")

    weekly_data = query.get_weekly_data(start_date, end_date)
    letters = weekly_data['letters'][:SAMPLE_SIZE]  # 샘플링
    posts = weekly_data['posts'][:SAMPLE_SIZE]  # 샘플링

    print(f"✓ 편지글 {len(letters)}건 조회 (샘플)")
    print(f"✓ 게시글 {len(posts)}건 조회 (샘플)")
    print()

    if not letters and not posts:
        print("❌ 데이터가 없어 리포트를 생성할 수 없습니다.")
        return

    # 2. 콘텐츠 분류
    print("2️⃣  콘텐츠 분류")
    print("-" * 60)

    classifier = ContentClassifier()

    if letters:
        print(f"편지글 {len(letters)}건 분류 중...")
        classified_letters = classifier.classify_batch(
            letters,
            content_field="message"
        )
        print(f"\n✓ 편지글 분류 완료")
    else:
        classified_letters = []

    if posts:
        print(f"게시글 {len(posts)}건 분류 중...")
        classified_posts = classifier.classify_batch(
            posts,
            content_field="textBody"
        )
        print(f"\n✓ 게시글 분류 완료")
    else:
        classified_posts = []

    print()

    # 3. 벡터 스토어에 저장
    print("3️⃣  벡터 스토어 저장")
    print("-" * 60)

    try:
        store = ChromaVectorStore(
            collection_name=f"week_{start_date}_sample",
            persist_directory="./chroma_db_test"
        )

        # 기존 데이터 초기화
        store.reset()

        # 데이터 저장
        total_added = 0
        if classified_letters:
            for letter in classified_letters:
                letter["message"] = letter.get("message", "")
            added = store.add_contents_batch(classified_letters, text_field="message")
            total_added += added

        if classified_posts:
            for post in classified_posts:
                post["message"] = post.get("textBody") or post.get("body", "")
            added = store.add_contents_batch(classified_posts, text_field="message")
            total_added += added

        print(f"✓ {total_added}건 벡터 스토어에 저장 완료")
    except Exception as e:
        print(f"⚠️  벡터 스토어 저장 실패: {str(e)}")

    print()

    # 4. 통계 분석
    print("4️⃣  통계 분석")
    print("-" * 60)

    analytics = WeeklyAnalytics()
    stats = analytics.analyze_weekly_data(
        classified_letters,
        classified_posts
    )

    total = stats["total_stats"]["this_week"]
    print(f"✓ 전체 통계: 편지 {total['letters']}건, 게시글 {total['posts']}건")

    category_stats = stats["category_stats"]
    print(f"✓ 카테고리별 통계:")
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {category}: {count}건")

    master_stats = stats["master_stats"]
    print(f"✓ 마스터별 통계: {len(master_stats)}개 마스터")

    feedbacks = stats.get("service_feedbacks", [])
    print(f"✓ 서비스 피드백: {len(feedbacks)}건")

    print()

    # 5. 리포트 생성
    print("5️⃣  리포트 생성")
    print("-" * 60)

    output_dir = "./reports"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"weekly_report_{start_date}_sample.md"
    output_path = os.path.join(output_dir, output_filename)

    generator = ReportGenerator()

    print("리포트 생성 중...")
    report = generator.generate_report(
        stats,
        start_date,
        end_date,
        output_path=output_path
    )

    print(f"✓ 리포트 생성 완료")
    print(f"✓ 저장 위치: {output_path}")
    print()

    # 6. 완료
    print("=" * 60)
    print("✅ 주간 리포트 생성 완료!")
    print("=" * 60)
    print()

    # 리포트 미리보기
    print("📄 리포트 미리보기:")
    print("-" * 60)
    lines = report.split('\n')
    for line in lines[:40]:
        print(line)

    if len(lines) > 40:
        print("\n... (전체 내용은 생성된 파일을 확인하세요)")


if __name__ == "__main__":
    main()
