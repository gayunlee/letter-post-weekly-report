"""채널톡 데이터 탐색 스크립트

BigQuery에서 채널톡(channel.io) 데이터를 조회하여 분류 체계 설계에 필요한
기초 통계를 확인합니다.

확인 항목:
- messages 테이블 스키마
- personType 분포 (user/bot/manager)
- 워크플로우 유형 (봇 메시지 패턴)
- 대화 길이 분포 (chatId별)
- 주요 키워드

사용법:
    python3 scripts/explore_channel_io.py
    python3 scripts/explore_channel_io.py --start 2026-02-09 --end 2026-02-16
"""
import sys
import os
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bigquery.client import BigQueryClient
from src.bigquery.channel_queries import ChannelQueryService
from src.bigquery.channel_preprocessor import group_by_chat, extract_user_text


def main():
    parser = argparse.ArgumentParser(description="채널톡 데이터 탐색")
    parser.add_argument("--start", default="2026-02-09", help="시작일")
    parser.add_argument("--end", default="2026-02-16", help="종료일")
    parser.add_argument("--sample", type=int, default=0, help="샘플 수 (0=주간 전체)")
    args = parser.parse_args()

    print("=" * 60)
    print("  채널톡 데이터 탐색")
    print("=" * 60)

    client = BigQueryClient()
    cq = ChannelQueryService(client)

    # 1. 스키마 확인
    print("\n[1] messages 테이블 스키마")
    print("-" * 40)
    try:
        schema = cq.get_schema_info()
        for field in schema:
            print(f"  {field['name']:30s} {field['type']:10s} ({field['mode']})")
    except Exception as e:
        print(f"  스키마 조회 실패: {e}")
        print("  데이터셋/테이블이 존재하는지 확인하세요.")
        return

    # 2. personType 분포
    print(f"\n[2] personType 분포 ({args.start} ~ {args.end})")
    print("-" * 40)
    try:
        dist = cq.get_person_type_distribution(args.start, args.end)
        total_msgs = sum(d["count"] for d in dist)
        total_chats = 0
        for d in dist:
            pct = d["count"] / total_msgs * 100 if total_msgs > 0 else 0
            print(f"  {d['personType']:15s}  메시지 {d['count']:>6,}건 ({pct:5.1f}%)  대화 {d['chat_count']:>5,}건")
            total_chats = max(total_chats, d["chat_count"])
        print(f"  {'합계':15s}  메시지 {total_msgs:>6,}건           대화 ~{total_chats:>5,}건")
    except Exception as e:
        print(f"  분포 조회 실패: {e}")

    # 3. chatId별 통계
    print(f"\n[3] 대화별 메시지 수 분포")
    print("-" * 40)
    try:
        chat_stats = cq.get_chat_stats(args.start, args.end)
        msg_counts = [s["message_count"] for s in chat_stats]
        user_counts = [s["user_messages"] for s in chat_stats]

        if msg_counts:
            print(f"  대화 수: {len(msg_counts):,}건")
            print(f"  메시지 수/대화: 평균 {sum(msg_counts)/len(msg_counts):.1f}, "
                  f"중앙값 {sorted(msg_counts)[len(msg_counts)//2]}, "
                  f"최대 {max(msg_counts)}")
            print(f"  사용자 메시지/대화: 평균 {sum(user_counts)/len(user_counts):.1f}, "
                  f"최대 {max(user_counts)}")

            # 길이 분포 히스토그램
            buckets = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, None)]
            print("\n  메시지 수 분포:")
            for lo, hi in buckets:
                if hi is None:
                    cnt = sum(1 for m in msg_counts if m >= lo)
                    label = f"  {lo}+"
                else:
                    cnt = sum(1 for m in msg_counts if lo <= m <= hi)
                    label = f"  {lo}-{hi}"
                bar = "#" * min(cnt * 50 // len(msg_counts), 50)
                print(f"  {label:>8s}: {cnt:>5,}건 {bar}")
    except Exception as e:
        print(f"  통계 조회 실패: {e}")

    # 4. 샘플 대화 확인
    print(f"\n[4] 샘플 대화 (최근 5건)")
    print("-" * 40)
    try:
        if args.sample > 0:
            messages = cq.get_sample_messages(limit=args.sample)
        else:
            messages = cq.get_weekly_messages(args.start, args.end, limit=500)

        groups = group_by_chat(messages)

        shown = 0
        for chat_id, chat_msgs in list(groups.items())[:5]:
            user_text = extract_user_text(chat_msgs)
            user_count = sum(1 for m in chat_msgs if m.get("personType") == "user")
            bot_count = sum(1 for m in chat_msgs if m.get("personType") == "bot")

            print(f"\n  chatId: {chat_id}")
            print(f"  메시지: {len(chat_msgs)}건 (user:{user_count}, bot:{bot_count})")
            print(f"  사용자 텍스트 ({len(user_text)}자):")
            preview = user_text[:200].replace("\n", " ")
            print(f"    {preview}{'...' if len(user_text) > 200 else ''}")
            shown += 1

    except Exception as e:
        print(f"  샘플 조회 실패: {e}")

    # 5. 사용자 메시지 키워드 분석
    print(f"\n[5] 사용자 메시지 주요 키워드")
    print("-" * 40)
    try:
        if 'messages' not in dir() or not messages:
            messages = cq.get_weekly_messages(args.start, args.end, limit=2000)

        word_counter = Counter()
        for msg in messages:
            if msg.get("personType") == "user":
                text = (msg.get("plainText") or "").strip()
                # 간단한 형태소 분리 (공백 기준)
                words = [w for w in text.split() if len(w) >= 2]
                word_counter.update(words)

        if word_counter:
            print("  상위 30개:")
            for word, count in word_counter.most_common(30):
                print(f"    {word}: {count}회")
    except Exception as e:
        print(f"  키워드 분석 실패: {e}")

    print("\n" + "=" * 60)
    print("  탐색 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
