"""v3 분류 체계 주간 분류 스크립트

BigQuery → v3 Topic(5분류) + Sentiment(3분류) 분류 → JSON 캐시 저장
리포트 생성 없이 분류만 수행합니다 (리포트는 추후 별도 파이프라인).

사용법:
    python3 scripts/classify_v3.py
    python3 scripts/classify_v3.py --start 2026-02-09 --end 2026-02-16
    python3 scripts/classify_v3.py --skip-detail-tags   # 태그 추출 생략
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier_v3.v3_topic_classifier import V3TopicClassifier

V3_DATA_DIR = "./data/classified_data_v3"


def classify_week(start_date, end_date, classifier, master_info=None):
    """특정 주간의 v3 분류 데이터 생성"""
    print(f"\n{'='*60}")
    print(f"  {start_date} ~ {end_date} 데이터 생성 (v3)")
    print("=" * 60)

    # 캐시 확인
    cache_path = os.path.join(V3_DATA_DIR, f"{start_date}.json")
    if os.path.exists(cache_path):
        print("  이미 존재하는 데이터 — 캐시에서 로드")
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("letters", []), data.get("posts", [])

    print("  BigQuery 조회 중...")
    client = BigQueryClient()
    query = WeeklyDataQuery(client)

    if master_info is None:
        print("  마스터 정보 조회 중...")
        master_info = query.get_master_info()
        print(f"  {len(master_info)}개 마스터 정보 로드")

    weekly_data = query.get_weekly_data(start_date, end_date)
    letters = weekly_data["letters"]
    posts = weekly_data["posts"]
    print(f"  편지 {len(letters)}건, 게시글 {len(posts)}건 조회")

    if not letters and not posts:
        print("  데이터 없음")
        return [], []

    # 마스터 이름 매핑
    for item in letters:
        master_id = item.get("masterId")
        if master_id and master_id in master_info:
            item["masterName"] = master_info[master_id]["displayName"]
            item["masterClubName"] = master_info[master_id]["clubName"]
            item["actualMasterId"] = master_id
        else:
            item["masterName"] = "Unknown"
            item["masterClubName"] = "Unknown"
            item["actualMasterId"] = master_id or "unknown"

    # 게시글: postBoardId → 실제 masterId 변환
    client_for_boards = BigQueryClient()
    board_query = f"""
    SELECT _id as boardId, masterId
    FROM `{client_for_boards.project_id}.us_plus.postboards`
    """
    board_to_master = {
        b["boardId"]: b["masterId"]
        for b in client_for_boards.execute_query(board_query)
    }

    for item in posts:
        board_id = item.get("postBoardId")
        actual_master_id = board_to_master.get(board_id, board_id)
        if actual_master_id and actual_master_id in master_info:
            item["masterName"] = master_info[actual_master_id]["displayName"]
            item["masterClubName"] = master_info[actual_master_id]["clubName"]
            item["actualMasterId"] = actual_master_id
        else:
            item["masterName"] = "Unknown"
            item["masterClubName"] = "Unknown"
            item["actualMasterId"] = actual_master_id or "unknown"

    # v3 분류
    print("  v3 분류 중...")
    classified_letters = classifier.classify_batch(letters, "message") if letters else []
    classified_posts = classifier.classify_batch(posts, "textBody") if posts else []

    # 캐시 저장
    os.makedirs(V3_DATA_DIR, exist_ok=True)
    cache_data = {
        "start_date": start_date,
        "end_date": end_date,
        "letters": classified_letters,
        "posts": classified_posts,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  캐시 저장: {cache_path}")

    return classified_letters, classified_posts


def main():
    parser = argparse.ArgumentParser(description="v3 분류 체계 주간 분류")
    parser.add_argument("--start", default="2026-02-09", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-16", help="종료일 (exclusive)")
    parser.add_argument("--skip-detail-tags", action="store_true", help="detail_tags 추출 생략")
    parser.add_argument("--detail-tag-model", default="claude-haiku-4-5-20251001",
                        help="detail_tags 모델 (기본: claude-haiku-4-5-20251001)")
    args = parser.parse_args()

    print("=" * 60)
    print("  v3 분류 체계 주간 분류")
    print(f"  대상: {args.start} ~ {args.end}")
    print("=" * 60)

    # 분류기 초기화
    print("\n[0단계] v3 분류기 초기화")
    classifier = V3TopicClassifier()

    # 분류 실행
    print("\n[1단계] 주간 데이터 분류")
    classified_letters, classified_posts = classify_week(args.start, args.end, classifier)

    if not classified_letters and not classified_posts:
        print("\n  데이터가 없어 종료합니다.")
        return

    total = len(classified_letters) + len(classified_posts)
    print(f"\n  분류 완료: 편지 {len(classified_letters)}건, 게시글 {len(classified_posts)}건 (총 {total}건)")

    # Topic 분포
    from collections import Counter
    topic_dist = Counter()
    sent_dist = Counter()
    for item in classified_letters + classified_posts:
        cls = item.get("classification", {})
        topic_dist[cls.get("topic", "미분류")] += 1
        sent_dist[cls.get("sentiment", "미분류")] += 1

    print("\n  Topic 분포:")
    for topic, count in topic_dist.most_common():
        pct = count / total * 100
        print(f"    {topic}: {count}건 ({pct:.1f}%)")

    print("\n  Sentiment 분포:")
    for sent, count in sent_dist.most_common():
        pct = count / total * 100
        print(f"    {sent}: {count}건 ({pct:.1f}%)")

    # detail_tags 추출 (선택)
    if not args.skip_detail_tags:
        has_tags = any(
            item.get("detail_tags") for item in (classified_letters + classified_posts)
        )
        if has_tags:
            print("\n[2단계] detail_tags — 이미 존재, 건너뜀")
        else:
            print(f"\n[2단계] detail_tags 추출 (모델: {args.detail_tag_model})")
            from src.classifier_v3.detail_tag_extractor import DetailTagExtractorV3, aggregate_category_tags

            tag_extractor = DetailTagExtractorV3(model=args.detail_tag_model)

            if classified_letters:
                print(f"  편지 {len(classified_letters)}건 태그 추출 중...")
                classified_letters = tag_extractor.extract_tags_batch(
                    classified_letters, content_field="message"
                )
            if classified_posts:
                print(f"  게시글 {len(classified_posts)}건 태그 추출 중...")
                classified_posts = tag_extractor.extract_tags_batch(
                    classified_posts, content_field="textBody"
                )

            cost = tag_extractor.get_cost_report()
            print(f"  비용: ${cost['estimated_cost_usd']:.4f} "
                  f"(건당 ${cost['cost_per_item_usd']:.6f}, "
                  f"파싱 성공률 {cost['parse_success_rate']}%)")

            agg = aggregate_category_tags(classified_letters + classified_posts)
            print(f"  태그 커버리지: {agg['tag_coverage']}% ({agg['tagged_items']}/{agg['total_items']}건)")

            # 캐시 업데이트 (detail_tags 포함)
            cache_path = os.path.join(V3_DATA_DIR, f"{args.start}.json")
            cache_data = {
                "start_date": args.start,
                "end_date": args.end,
                "letters": classified_letters,
                "posts": classified_posts,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"  캐시 업데이트: {cache_path}")
    else:
        print("\n[2단계] detail_tags — 건너뜀")

    print("\n" + "=" * 60)
    print("  v3 분류 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
