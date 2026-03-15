"""채널톡 CS 데이터 2축 분류 파이프라인

BigQuery → dedup → route 판정 → topic 분류(LLM) → 저장 + 통계

사용법:
    python scripts/classify_channel_talk.py --weeks 1
    python scripts/classify_channel_talk.py --weeks 4
    python scripts/classify_channel_talk.py --start 2026-02-01 --end 2026-03-01

출력:
    data/channel_io/classified_{start}_{end}.json
"""
import sys
import os
import json
import argparse
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bigquery.client import BigQueryClient
from src.bigquery.channel_queries import ChannelQueryService
from src.bigquery.channel_preprocessor import (
    dedup_messages,
    build_chat_items,
)
from src.classifier_v4.channel_classifier import ChannelClassifier

OUTPUT_DIR = "./data/channel_io"


def print_crosstab(items):
    """route × topic 교차표 출력"""
    cross = Counter()
    for item in items:
        route = item.get("route", "unknown")
        topic = (item.get("classification") or {}).get("topic", "미분류")
        cross[(route, topic)] += 1

    # 축 값 수집
    routes = sorted({r for r, _ in cross.keys()})
    topics = sorted({t for _, t in cross.keys()})

    # 헤더
    header = f"{'route':<20}" + "".join(f"{t:>12}" for t in topics) + f"{'합계':>10}"
    print(header)
    print("-" * len(header))

    for route in routes:
        row = f"{route:<20}"
        row_total = 0
        for topic in topics:
            count = cross.get((route, topic), 0)
            row += f"{count:>12}"
            row_total += count
        row += f"{row_total:>10}"
        print(row)

    # 합계 행
    total_row = f"{'합계':<20}"
    grand_total = 0
    for topic in topics:
        col_total = sum(cross.get((r, topic), 0) for r in routes)
        total_row += f"{col_total:>12}"
        grand_total += col_total
    total_row += f"{grand_total:>10}"
    print("-" * len(header))
    print(total_row)


def main():
    parser = argparse.ArgumentParser(description="채널톡 CS 2축 분류 파이프라인")
    parser.add_argument("--start", default=None, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--weeks", type=int, default=1, help="최근 N주 (start/end 미지정시)")
    parser.add_argument("--min-chars", type=int, default=10, help="최소 사용자 텍스트 길이")
    parser.add_argument("--output", default=None, help="출력 파일 경로")
    parser.add_argument("--dry-run", action="store_true", help="LLM 호출 없이 route만 판정")
    args = parser.parse_args()

    # 날짜 계산
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        today = datetime.now()
        this_monday = today - timedelta(days=today.weekday())
        end_date = this_monday.strftime("%Y-%m-%d")
        start_date = (this_monday - timedelta(weeks=args.weeks)).strftime("%Y-%m-%d")

    output_path = args.output or os.path.join(
        OUTPUT_DIR, f"classified_{start_date}_{end_date}.json"
    )

    print("=" * 60)
    print("  채널톡 CS 2축 분류 파이프라인")
    print(f"  기간: {start_date} ~ {end_date}")
    if args.dry_run:
        print("  ** DRY RUN — LLM 호출 없이 route만 판정 **")
    print("=" * 60)

    # [1] BigQuery 조회
    print("\n[1] BigQuery에서 messages + chats 조회 중...")
    client = BigQueryClient()
    cq = ChannelQueryService(client)
    raw_messages, chat_states = cq.get_weekly_conversations(start_date, end_date)
    print(f"  {len(raw_messages):,}건 메시지 조회 완료")
    print(f"  {len(chat_states):,}건 chat state 조회 완료")

    if not raw_messages:
        print("  메시지가 없습니다.")
        return

    # [2] Dedup
    print("\n[2] 메시지 중복 제거...")
    messages = dedup_messages(raw_messages)
    removed = len(raw_messages) - len(messages)
    print(f"  {removed:,}건 중복 제거 → {len(messages):,}건")

    # [3] 전처리 + route 판정
    print("\n[3] chatId별 그룹핑 + route 판정...")
    items = build_chat_items(
        messages,
        min_user_chars=args.min_chars,
        chat_states=chat_states,
    )
    print(f"  {len(items):,}건 대화 (유효 텍스트 {args.min_chars}자 이상)")

    if not items:
        print("  유효한 대화가 없습니다.")
        return

    # route 분포
    route_counts = Counter(item["route"] for item in items)
    print(f"\n  Route 분포:")
    for route, count in route_counts.most_common():
        pct = count / len(items) * 100
        print(f"    {route:<20} {count:>6}건 ({pct:.1f}%)")

    # 워크플로우 버튼 통계
    button_counts = Counter()
    for item in items:
        for btn in item.get("workflow_buttons", []):
            button_counts[btn] += 1
    if button_counts:
        print(f"\n  워크플로우 버튼 분포:")
        for btn, count in button_counts.most_common(10):
            print(f"    {btn:<25} {count:>6}건")

    # [4] Topic 분류 (LLM)
    if args.dry_run:
        print("\n[4] DRY RUN — topic 분류 생략")
    else:
        # manager_resolved + bot_resolved만 분류
        classify_targets = [
            item for item in items if item["route"] in ("manager_resolved", "bot_resolved")
        ]
        skip_count = len(items) - len(classify_targets)
        print(f"\n[4] Topic 분류 (Haiku API)...")
        print(f"  분류 대상: {len(classify_targets):,}건 (abandoned {skip_count}건 제외)")

        if classify_targets:
            classifier = ChannelClassifier()
            classifier.classify_batch(classify_targets, content_field="text")

            # 비용 리포트
            cost = classifier.get_cost_report()
            print(f"\n  API 비용: ${cost['cost_usd']:.4f}")
            print(f"  토큰: input {cost['input_tokens']:,} / output {cost['output_tokens']:,}")
            if cost["errors"]:
                print(f"  에러: {cost['errors']}건")

    # [5] 통계 출력
    print(f"\n[5] 분류 결과 통계")
    print("-" * 60)

    # topic 분포 (분류된 것만)
    topic_counts = Counter()
    for item in items:
        cls = item.get("classification")
        if cls:
            topic_counts[cls["topic"]] += 1

    if topic_counts:
        print(f"\n  Topic 분포:")
        for topic, count in topic_counts.most_common():
            pct = count / sum(topic_counts.values()) * 100
            print(f"    {topic:<15} {count:>6}건 ({pct:.1f}%)")

        # 교차표
        print(f"\n  Route × Topic 교차표:")
        print_crosstab(items)

    # [6] 저장
    print(f"\n[6] 결과 저장...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    export_data = {
        "metadata": {
            "source": "channel_io",
            "pipeline": "2axis_classification",
            "start_date": start_date,
            "end_date": end_date,
            "total_raw_messages": len(raw_messages),
            "total_deduped_messages": len(messages),
            "total_chats": len(items),
            "route_distribution": dict(route_counts),
            "topic_distribution": dict(topic_counts) if topic_counts else {},
            "dry_run": args.dry_run,
            "created_at": datetime.now().isoformat(),
        },
        "items": items,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

    file_size = os.path.getsize(output_path) / 1024
    print(f"  저장 완료: {output_path} ({file_size:.1f}KB)")

    # 샘플 확인
    classified = [i for i in items if i.get("classification")]
    if classified:
        print(f"\n[7] 분류 샘플 (상위 5건)")
        print("-" * 60)
        for item in classified[:5]:
            cls = item["classification"]
            preview = item["text"][:120].replace("\n", " ")
            buttons = ", ".join(item.get("workflow_buttons", [])) or "-"
            print(f"  chatId: {item['chatId']}")
            print(f"  route: {item['route']} | topic: {cls['topic']} | conf: {cls['confidence']:.2f}")
            print(f"  buttons: {buttons}")
            print(f"  summary: {cls.get('summary', '-')}")
            print(f"  text: {preview}{'...' if len(item['text']) > 120 else ''}")
            print()

    print("=" * 60)
    print("  분류 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
