"""기존 리포트를 Notion에 업로드하는 스크립트"""
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.integrations.notion_client import NotionReportClient


def upload_report_to_notion(report_path: str, start_date: str, end_date: str):
    """
    기존 마크다운 리포트를 Notion에 업로드

    Args:
        report_path: 리포트 파일 경로
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD, exclusive)
    """
    print("="*60)
    print("📤 Notion 리포트 업로드")
    print("="*60)

    # 리포트 파일 읽기
    if not os.path.exists(report_path):
        print(f"❌ 리포트 파일을 찾을 수 없습니다: {report_path}")
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()

    print(f"✓ 리포트 파일 로드: {report_path}")

    # Notion 클라이언트 초기화
    try:
        notion_client = NotionReportClient()
        print("✓ Notion 클라이언트 초기화 완료")
    except Exception as e:
        print(f"❌ Notion 클라이언트 초기화 실패: {e}")
        return

    # 페이지 제목 생성
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    start_formatted = start_dt.strftime('%Y.%m.%d')
    end_formatted = (end_dt - timedelta(days=1)).strftime('%m.%d')
    page_title = f"이용자 반응 리포트 ({start_formatted} ~ {end_formatted})"

    # Notion에 업로드
    try:
        page_info = notion_client.create_report_page(
            title=page_title,
            markdown_content=report_content,
            start_date=start_date,
            end_date=end_date
        )

        notion_url = page_info["url"]
        print(f"✓ Notion 페이지 생성 완료")
        print(f"✓ 제목: {page_title}")
        print(f"✓ URL: {notion_url}")

        return notion_url

    except Exception as e:
        print(f"❌ Notion 업로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 리포트 설정
    start_date = "2026-01-05"
    end_date = "2026-01-12"  # exclusive
    report_path = f"./reports/weekly_report_{start_date}.md"

    # Notion 업로드
    notion_url = upload_report_to_notion(report_path, start_date, end_date)

    if notion_url:
        print("\n" + "="*60)
        print("✅ Notion 업로드 완료!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Notion 업로드 실패")
        print("="*60)
