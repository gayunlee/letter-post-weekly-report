#!/usr/bin/env python3
"""라벨링된 데이터로 주간 리포트 생성

GPT-4o-mini로 라벨링된 데이터를 사용하여 리포트 생성

사용법:
    python scripts/generate_report_from_labeled.py
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.reporter.analytics import WeeklyAnalytics
from src.reporter.report_generator import ReportGenerator
from src.integrations.notion_client import NotionReportClient
from src.integrations.slack_client import SlackNotifier
from src.utils.excel_exporter import export_to_excel

# 새 카테고리 → 기존 리포트용 카테고리 매핑
CATEGORY_MAPPING = {
    "서비스 이슈": "불편사항",  # 서비스 관련 불만/문의
    "서비스 칭찬": "감사·후기",  # 서비스/콘텐츠 칭찬
    "투자 질문": "질문·토론",  # 투자 관련 질문
    "정보/의견": "정보성 글",  # 뉴스, 분석, 투자 심리
    "일상 소통": "일상·공감",  # 인사, 안부
}


def load_labeled_data(labeled_path: str, start_date: str, end_date: str):
    """라벨링된 데이터 로드 및 날짜 필터링"""
    with open(labeled_path, encoding="utf-8") as f:
        data = json.load(f)

    # 날짜 필터링
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    filtered = []
    for item in data:
        created_at = item.get("createdAt", "")
        if created_at:
            try:
                item_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if start_dt <= item_dt.replace(tzinfo=None) < end_dt:
                    filtered.append(item)
            except Exception:
                pass

    return filtered


def get_master_id_mapping(client, items: list):
    """id를 기준으로 masterId 매핑 조회"""
    letter_ids = [item["id"] for item in items if item.get("source") == "letter"]
    post_ids = [item["id"] for item in items if item.get("source") == "post"]

    id_to_master = {}

    # 편지 masterId 조회
    if letter_ids:
        ids_str = ", ".join([f"'{id}'" for id in letter_ids])
        query = f"""
        SELECT _id as id, masterId
        FROM `{client.project_id}.us_plus.usermastermessages`
        WHERE _id IN ({ids_str})
        """
        results = client.execute_query(query)
        for r in results:
            id_to_master[r["id"]] = {"masterId": r["masterId"], "postBoardId": None}

    # 게시글 postBoardId 조회
    if post_ids:
        ids_str = ", ".join([f"'{id}'" for id in post_ids])
        query = f"""
        SELECT _id as id, postBoardId
        FROM `{client.project_id}.us_plus.posts`
        WHERE _id IN ({ids_str})
        """
        results = client.execute_query(query)
        for r in results:
            id_to_master[r["id"]] = {"masterId": None, "postBoardId": r["postBoardId"]}

    return id_to_master


def enrich_with_master_info(items: list, master_info: dict, board_to_master: dict, id_to_master: dict):
    """마스터 정보 추가"""
    for item in items:
        source = item.get("source", "letter")
        item_id = item.get("id")

        mapping = id_to_master.get(item_id, {})

        if source == "letter":
            master_id = mapping.get("masterId")
            item["masterId"] = master_id
        else:
            # post의 경우 postBoardId를 masterId로 변환
            board_id = mapping.get("postBoardId")
            item["postBoardId"] = board_id
            master_id = board_to_master.get(board_id, board_id)

        if master_id and master_id in master_info:
            item["masterName"] = master_info[master_id]["displayName"]
            item["masterClubName"] = master_info[master_id]["clubName"]
            item["actualMasterId"] = master_id
        else:
            item["masterName"] = "Unknown"
            item["masterClubName"] = "Unknown"
            item["actualMasterId"] = master_id or "unknown"

    return items


def convert_to_classified_format(items: list):
    """라벨링 데이터를 기존 분류 형식으로 변환"""
    letters = []
    posts = []

    for item in items:
        # 새 카테고리를 기존 카테고리로 매핑
        new_category = item.get("category", "일상 소통")
        mapped_category = CATEGORY_MAPPING.get(new_category, "일상·공감")

        # classification 형식으로 변환
        classified_item = {
            **item,
            "classification": {
                "category": mapped_category,
                "reason": f"GPT-4o-mini 분류: {new_category}",
                "confidence": 0.9,
            },
            # 원본 카테고리 보존
            "original_category": new_category,
        }

        if item.get("source") == "letter":
            classified_item["message"] = item.get("text", "")
            letters.append(classified_item)
        else:
            classified_item["textBody"] = item.get("text", "")
            posts.append(classified_item)

    return letters, posts


def main():
    project_root = Path(__file__).parent.parent

    # 대상 기간 (1월 12일 ~ 1월 18일)
    target_start = "2026-01-12"
    target_end = "2026-01-19"  # exclusive

    # 전주 기간 (1월 5일 ~ 1월 11일)
    prev_start = "2026-01-05"
    prev_end = "2026-01-12"

    # 라벨링 데이터 경로
    labeled_path = project_root / "data" / "labeling" / "gpt4o_labeled.json"

    print("=" * 60)
    print("📊 라벨링 데이터로 주간 리포트 생성")
    print("=" * 60)
    print(f"대상 기간: {target_start} ~ {target_end}")
    print(f"전주 기간: {prev_start} ~ {prev_end}")

    # 1. 마스터 정보 조회
    print("\n[1단계] 마스터 정보 조회")
    client = BigQueryClient()
    query = WeeklyDataQuery(client)
    master_info = query.get_master_info()
    print(f"✓ {len(master_info)}개 마스터 정보 로드")

    # postBoardId → masterId 매핑
    board_to_master_query = f"""
    SELECT _id as boardId, masterId
    FROM `{client.project_id}.us_plus.postboards`
    """
    board_to_master = {
        b["boardId"]: b["masterId"] for b in client.execute_query(board_to_master_query)
    }

    # 2. 라벨링 데이터 로드
    print("\n[2단계] 라벨링 데이터 로드")
    target_items = load_labeled_data(str(labeled_path), target_start, target_end)
    prev_items = load_labeled_data(str(labeled_path), prev_start, prev_end)
    print(f"✓ 대상 주간: {len(target_items)}건")
    print(f"✓ 전주: {len(prev_items)}건")

    if not target_items:
        print("\n❌ 대상 기간 데이터가 없습니다.")
        return

    # 3. 마스터 정보 추가
    print("\n[3단계] 마스터 정보 추가")
    # id로 masterId 매핑 조회
    all_items = target_items + prev_items
    id_to_master = get_master_id_mapping(client, all_items)
    print(f"✓ {len(id_to_master)}개 항목 masterId 매핑 완료")

    target_items = enrich_with_master_info(target_items, master_info, board_to_master, id_to_master)
    prev_items = enrich_with_master_info(prev_items, master_info, board_to_master, id_to_master)

    # 4. 분류 형식 변환
    print("\n[4단계] 분류 형식 변환")
    classified_letters, classified_posts = convert_to_classified_format(target_items)
    prev_letters, prev_posts = convert_to_classified_format(prev_items)
    print(f"✓ 편지: {len(classified_letters)}건, 게시글: {len(classified_posts)}건")

    # 5. 통계 분석
    print("\n[5단계] 통계 분석")
    analytics = WeeklyAnalytics()
    stats = analytics.analyze_weekly_data(
        classified_letters,
        classified_posts,
        previous_letters=prev_letters,
        previous_posts=prev_posts,
    )

    total = stats["total_stats"]["this_week"]
    print(f"✓ 전체 통계: 편지 {total['letters']}건, 게시글 {total['posts']}건")

    # 카테고리 분포 출력
    print("\n카테고리 분포:")
    for cat, count in sorted(
        stats["category_stats"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {cat}: {count}건")

    # 6. 리포트 생성
    print("\n[6단계] 리포트 생성")
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)

    output_filename = f"weekly_report_{target_start}.md"
    output_path = output_dir / output_filename

    generator = ReportGenerator()
    report = generator.generate_report(
        stats, target_start, target_end, output_path=str(output_path)
    )

    print(f"✓ 리포트 생성 완료: {output_path}")

    # 7. 엑셀 파일 생성
    print("\n[7단계] 엑셀 파일 생성")
    excel_filename = f"weekly_data_{target_start}.xlsx"
    excel_path = output_dir / excel_filename
    export_to_excel(classified_letters, classified_posts, str(excel_path))
    print(f"✓ 엑셀 파일 생성: {excel_path}")

    # 8. Notion 업로드
    print("\n[8단계] Notion 업로드")
    try:
        notion_client = NotionReportClient()
        week_label = SlackNotifier.get_week_label(target_start)

        start_formatted = datetime.strptime(target_start, "%Y-%m-%d").strftime(
            "%Y.%m.%d"
        )
        end_dt = datetime.strptime(target_end, "%Y-%m-%d")
        end_formatted = (end_dt - timedelta(days=1)).strftime("%m.%d")
        page_title = f"이용자 반응 리포트 ({start_formatted} ~ {end_formatted})"

        page_info = notion_client.create_report_page(
            title=page_title,
            markdown_content=report,
            start_date=target_start,
            end_date=target_end,
        )

        notion_url = page_info["url"]
        print(f"✓ Notion 페이지 생성 완료")
        print(f"✓ URL: {notion_url}")
    except Exception as e:
        print(f"⚠️  Notion 업로드 실패: {str(e)}")
        notion_url = None

    # 9. Slack 알림
    print("\n[9단계] Slack 알림 전송")
    slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
    slack_channel_id = os.getenv("SLACK_CHANNEL_ID")

    if not slack_bot_token or not slack_channel_id:
        print("⚠️  SLACK_BOT_TOKEN 또는 SLACK_CHANNEL_ID가 설정되지 않았습니다.")
        print("   .env 파일에 다음 설정을 추가하세요:")
        print("   SLACK_BOT_TOKEN=xoxb-...")
        print("   SLACK_CHANNEL_ID=C...")
    elif notion_url:
        try:
            slack_client = SlackNotifier()
            result = slack_client.send_report_notification(
                week_label=week_label,
                start_date=target_start,
                end_date=target_end,
                notion_url=notion_url,
            )

            if result.get("ok"):
                print(f"✓ Slack 알림 전송 완료")

                # 엑셀 파일 스레드 업로드
                message_ts = result.get("message_ts")
                if message_ts and excel_path.exists():
                    file_result = slack_client.upload_file_to_thread(
                        file_path=str(excel_path),
                        thread_ts=message_ts,
                        title=f"원본 데이터 ({target_start})",
                        comment="📎 라벨링된 원본 데이터 파일입니다.",
                    )
                    if file_result.get("ok"):
                        print(f"✓ 엑셀 파일 업로드 완료")
                    else:
                        print(f"⚠️  엑셀 파일 업로드 실패: {file_result.get('error')}")
            else:
                print(f"⚠️  Slack 알림 전송 실패: {result.get('error')}")
        except Exception as e:
            print(f"⚠️  Slack 알림 전송 실패: {str(e)}")
    else:
        print("⚠️  Notion URL이 없어 Slack 알림을 건너뜁니다.")

    print()
    print("=" * 60)
    print("✅ 주간 리포트 생성 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
