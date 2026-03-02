"""채널톡 데이터 라벨링용 추출 스크립트

BigQuery → chatId 그룹핑 → 사용자 텍스트 연결 → JSON 저장
생성된 JSON은 Colab에서 EXAONE 라벨링에 사용됩니다.

사용법:
    python3 scripts/export_channel_for_labeling.py
    python3 scripts/export_channel_for_labeling.py --start 2026-02-09 --end 2026-02-16
    python3 scripts/export_channel_for_labeling.py --weeks 4  # 최근 4주

출력:
    data/channel_io/channel_items_for_labeling.json
"""
import sys
import os
import json
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bigquery.client import BigQueryClient
from src.bigquery.channel_queries import ChannelQueryService
from src.bigquery.channel_preprocessor import build_chat_items

OUTPUT_DIR = "./data/channel_io"


def main():
    parser = argparse.ArgumentParser(description="채널톡 라벨링용 데이터 추출")
    parser.add_argument("--start", default=None, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--weeks", type=int, default=4, help="최근 N주 (start/end 미지정시)")
    parser.add_argument("--min-chars", type=int, default=10, help="최소 사용자 텍스트 길이")
    parser.add_argument("--output", default=None, help="출력 파일 경로")
    args = parser.parse_args()

    # 날짜 계산
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        today = datetime.now()
        # 이번 주 월요일
        this_monday = today - timedelta(days=today.weekday())
        end_date = this_monday.strftime("%Y-%m-%d")
        start_date = (this_monday - timedelta(weeks=args.weeks)).strftime("%Y-%m-%d")

    output_path = args.output or os.path.join(OUTPUT_DIR, "channel_items_for_labeling.json")

    print("=" * 60)
    print("  채널톡 라벨링용 데이터 추출")
    print(f"  기간: {start_date} ~ {end_date}")
    print("=" * 60)

    # BigQuery 조회
    print("\n[1] BigQuery에서 메시지 조회 중...")
    client = BigQueryClient()
    cq = ChannelQueryService(client)
    messages = cq.get_weekly_messages(start_date, end_date)
    print(f"  {len(messages):,}건 메시지 조회 완료")

    if not messages:
        print("  메시지가 없습니다.")
        return

    # chatId 그룹핑 + 텍스트 연결
    print("\n[2] chatId별 그룹핑 + 사용자 텍스트 연결...")
    items = build_chat_items(messages, min_user_chars=args.min_chars)
    print(f"  {len(items):,}건 대화 (유효 텍스트 {args.min_chars}자 이상)")

    if not items:
        print("  유효한 대화가 없습니다.")
        return

    # 통계
    text_lengths = [len(item["text"]) for item in items]
    avg_len = sum(text_lengths) / len(text_lengths)
    median_len = sorted(text_lengths)[len(text_lengths) // 2]

    print(f"\n  텍스트 길이: 평균 {avg_len:.0f}자, 중앙값 {median_len}자, "
          f"최소 {min(text_lengths)}자, 최대 {max(text_lengths)}자")

    msg_counts = [item["message_count"] for item in items]
    print(f"  대화당 메시지: 평균 {sum(msg_counts)/len(msg_counts):.1f}건")

    # 저장
    print(f"\n[3] JSON 저장...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    export_data = {
        "metadata": {
            "source": "channel_io",
            "start_date": start_date,
            "end_date": end_date,
            "total_messages": len(messages),
            "total_chats": len(items),
            "min_chars": args.min_chars,
            "exported_at": datetime.now().isoformat(),
        },
        "items": items,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

    file_size = os.path.getsize(output_path) / 1024
    print(f"  저장 완료: {output_path} ({file_size:.1f}KB)")
    print(f"  대화 {len(items):,}건, 메시지 {len(messages):,}건")

    # 샘플 확인
    print(f"\n[4] 샘플 확인 (상위 3건)")
    print("-" * 40)
    for item in items[:3]:
        preview = item["text"][:150].replace("\n", " ")
        print(f"  chatId: {item['chatId']}")
        print(f"  메시지: {item['message_count']}건 (user:{item['user_message_count']}, "
              f"bot:{item['bot_message_count']}, mgr:{item['manager_message_count']})")
        print(f"  텍스트: {preview}{'...' if len(item['text']) > 150 else ''}")
        print()

    print("=" * 60)
    print("  추출 완료!")
    print(f"  다음 단계: make push → Colab에서 라벨링")
    print("=" * 60)


if __name__ == "__main__":
    main()
