"""기존 생성된 마크다운 리포트를 Notion/Slack에 업로드 (재분류 없이)

사용법:
    python scripts/upload_existing_report.py --report reports/weekly_report_v5_2026-04-06.md \
        --start 2026-04-06 --end 2026-04-13
"""
import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from src.integrations.notion_client import NotionReportClient
from src.integrations.slack_client import SlackNotifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="마크다운 리포트 파일 경로")
    parser.add_argument("--start", required=True, help="시작일 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="종료일 YYYY-MM-DD (exclusive)")
    parser.add_argument("--excel", help="(선택) 엑셀 파일 경로 — Slack 스레드 첨부용")
    parser.add_argument("--skip-notion", action="store_true", help="Notion 업로드 건너뛰기")
    parser.add_argument("--notion-url", help="기존 Notion URL (--skip-notion 시 Slack 메시지에 사용)")
    args = parser.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        report_md = f.read()

    print(f"리포트 로드: {args.report} ({len(report_md)} bytes)")

    # Notion 업로드
    if args.skip_notion:
        print("\n[1] Notion 업로드 건너뜀")
        notion_url = args.notion_url
        if not notion_url:
            raise SystemExit("--skip-notion 사용 시 --notion-url 필수")
        print(f"  기존 Notion URL 사용: {notion_url}")
    else:
        print("\n[1] Notion 업로드")
        start_formatted = datetime.strptime(args.start, '%Y-%m-%d').strftime('%Y.%m.%d')
        end_dt = datetime.strptime(args.end, '%Y-%m-%d')
        end_formatted = (end_dt - timedelta(days=1)).strftime('%m.%d')
        page_title = f"이용자 반응 리포트 ({start_formatted} ~ {end_formatted})"

        notion_client = NotionReportClient()
        page_info = notion_client.create_report_page(
            title=page_title,
            markdown_content=report_md,
            start_date=args.start,
            end_date=args.end,
        )
        notion_url = page_info["url"]
        print(f"  Notion URL: {notion_url}")

    # Slack 알림
    print("\n[2] Slack 알림")
    slack_client = SlackNotifier()
    week_label = SlackNotifier.get_week_label(args.start)
    result = slack_client.send_report_notification(
        week_label=week_label,
        start_date=args.start,
        end_date=args.end,
        notion_url=notion_url,
    )

    if result.get("ok"):
        print(f"  Slack 알림 전송 완료")
        message_ts = result.get("message_ts")
        if args.excel and message_ts and os.path.exists(args.excel):
            file_result = slack_client.upload_file_to_thread(
                file_path=args.excel,
                thread_ts=message_ts,
                title=f"원본 데이터 ({args.start})",
                comment="📎 라벨링된 원본 데이터 파일입니다.",
            )
            if file_result.get("ok"):
                print(f"  엑셀 업로드 완료")
            else:
                print(f"  엑셀 업로드 실패: {file_result.get('error')}")
    else:
        print(f"  Slack 알림 실패: {result.get('error')}")

    print("\n완료!")


if __name__ == "__main__":
    main()
