"""주간 리포트 생성 (v5 4분류) — 원본 조회 → v5 분류 → 리포트

voc_labelled에 분류 데이터가 없는 기간도 처리 가능.
BigQuery 원본 → Haiku v5 분류 → 통계 분석 → 리포트 생성.

사용법:
    python scripts/generate_weekly_report_v5.py                    # 지난 주
    python scripts/generate_weekly_report_v5.py --start 2026-03-23 --end 2026-03-30
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier_v5.classifier import V5Classifier
from src.classifier_v5.bedrock_classifier import BedrockV5Classifier
from src.reporter.analytics import WeeklyAnalytics
from src.reporter.report_generator import ReportGenerator
from src.reporter.feedback_clusterer import cluster_feedbacks, enrich_master_stats_with_clusters
from src.reporter.tone_reviewer import review_report


def classify_items(items, content_field, classifier):
    """v5 분류 결과를 item에 직접 매핑 (analytics 호환)"""
    if not items:
        return items

    classifier.classify_batch(items, content_field=content_field)

    # classification 결과를 item 최상위 필드로 복사 (analytics 호환)
    for item in items:
        cls = item.get("classification", {})
        item["topic"] = cls.get("topic", "")
        item["subtag"] = cls.get("subtag", "")
        item["sentiment"] = cls.get("sentiment", "")
        item["summary"] = cls.get("summary", "")
        item["tags"] = cls.get("tags", [])
        item["confidence"] = cls.get("confidence", 0.0)

    return items


def main():
    parser = argparse.ArgumentParser(description="주간 리포트 생성 (v5)")
    parser.add_argument("--start", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--bedrock", action="store_true", help="Bedrock Haiku 사용 (Anthropic API 대신)")
    args = parser.parse_args()

    print("=" * 60)
    print("  주간 리포트 생성 (v5 4분류)")
    print("=" * 60)

    # BigQuery 클라이언트
    bq_client = BigQueryClient()
    query = WeeklyDataQuery(bq_client)

    # 날짜 범위
    if args.start and args.end:
        start_date, end_date = args.start, args.end
    else:
        start_date, end_date = WeeklyDataQuery.get_last_week_range()

    print(f"\n  대상 기간: {start_date} ~ {end_date}")

    # 1. 원본 데이터 조회
    print(f"\n  Phase 1: BigQuery 원본 조회")
    master_info = query.get_master_info()
    data = query.get_weekly_data(start_date, end_date)
    letters = data["letters"]
    posts = data["posts"]

    # 마스터 정보 매핑
    for item in letters:
        mid = item.get("masterId", "")
        if mid in master_info:
            item["masterName"] = master_info[mid].get("displayName") or master_info[mid].get("name", "Unknown")
    for item in posts:
        mid = item.get("postBoardId", "")
        if mid in master_info:
            item["masterName"] = master_info[mid].get("displayName") or master_info[mid].get("name", "Unknown")

    print(f"    편지 {len(letters)}건, 게시글 {len(posts)}건")

    # 2. v5 분류
    print(f"\n  Phase 2: v5 4분류 (Bedrock Haiku)")
    bedrock_workers = min(args.workers, 30)  # Bedrock Haiku는 30까지 OK
    classifier = BedrockV5Classifier(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0", max_workers=bedrock_workers)
    start_time = time.time()

    classify_items(letters, "message", classifier)
    classify_items(posts, "textBody", classifier)

    cost = classifier.get_cost_report()
    elapsed = time.time() - start_time
    print(f"    분류 완료: {cost['total_items']}건, {elapsed:.1f}초, ${cost['cost_usd']}")

    # 3. 전주 데이터 (건수 비교용 — 분류 불필요)
    print(f"\n  Phase 3: 전주 건수 조회")
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    prev_start = (start_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    prev_end = start_date
    print(f"    전주 기간: {prev_start} ~ {prev_end}")

    prev_data = query.get_weekly_data(prev_start, prev_end)
    prev_letters = prev_data["letters"]
    prev_posts = prev_data["posts"]

    # 전주 마스터 매핑 (count 비교용)
    for item in prev_letters:
        mid = item.get("masterId", "")
        if mid in master_info:
            item["masterName"] = master_info[mid].get("displayName") or master_info[mid].get("name", "Unknown")
        else:
            item["masterName"] = "Unknown"
    board_query_sql = f"SELECT _id as boardId, masterId FROM `{bq_client.project_id}.us_plus_new.postboards`"
    board_to_master = {b["boardId"]: b["masterId"] for b in bq_client.execute_query(board_query_sql)}
    for item in prev_posts:
        bid = item.get("postBoardId", "")
        mid = board_to_master.get(bid, bid)
        if mid in master_info:
            item["masterName"] = master_info[mid].get("displayName") or master_info[mid].get("name", "Unknown")
        else:
            item["masterName"] = "Unknown"

    if prev_letters or prev_posts:
        print(f"    전주: 편지 {len(prev_letters)}건, 게시글 {len(prev_posts)}건 (건수만 사용)")
    else:
        prev_letters = None
        prev_posts = None
        print(f"    전주 데이터 없음")

    # 4. 통계 분석
    print(f"\n  Phase 4: 통계 분석")
    analytics = WeeklyAnalytics()
    stats = analytics.analyze_weekly_data(
        letters, posts,
        previous_letters=prev_letters,
        previous_posts=prev_posts,
    )

    total = stats["total_stats"]["this_week"]
    print(f"    전체: 편지 {total['letters']}건, 게시글 {total['posts']}건")

    category_stats = stats["category_stats"]
    print(f"    토픽별:")
    for topic, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"      {topic}: {count}건")

    feedbacks = stats.get("service_feedbacks", [])
    print(f"    피드백: {len(feedbacks)}건")

    # 4-1. 피드백 클러스터링
    print(f"\n  Phase 4-1: 피드백 클러스터링")
    if feedbacks:
        enriched, cluster_info = cluster_feedbacks(feedbacks)
        stats["service_feedbacks"] = enriched
        stats["feedback_clusters"] = cluster_info
        enrich_master_stats_with_clusters(stats)
        multi = [c for c in cluster_info.values() if c["size"] >= 2]
        print(f"    {len(cluster_info)}개 클러스터 (2건+: {len(multi)}개)")
    else:
        print(f"    피드백 없음")

    # 5. 리포트 생성
    print(f"\n  Phase 5: 리포트 생성")
    output_dir = os.getenv("REPORT_OUTPUT_DIR", "./reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"weekly_report_v5_{start_date}.md")

    generator = ReportGenerator()
    report = generator.generate_report(stats, start_date, end_date, output_path=output_path)

    # 5-1. 톤 검수
    print(f"\n  Phase 5-1: 톤 검수")
    fixed_report, review_stats = review_report(report)
    print(f"    {review_stats['fixed_sections']}/{review_stats['total_sections']} 섹션 수정, {review_stats['total_issues']}건 교정")
    if review_stats["total_issues"] > 0:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(fixed_report)
        report = fixed_report

    total_cost = classifier.get_cost_report()
    print(f"\n  저장: {output_path}")
    print(f"  총 비용: ${total_cost['cost_usd']}")
    print(f"\n{'='*60}")
    print(f"  완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
