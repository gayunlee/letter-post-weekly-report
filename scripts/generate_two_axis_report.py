"""2축 분류 체계 주간 리포트 생성 스크립트

기존 1축 generate_custom_week_report.py와 독립적으로 동작합니다.
파이프라인: BigQuery → 2축 파인튜닝 분류 → 통계 분석 → 리포트 생성 → 엑셀 → Notion → Slack

사용법:
    python3 scripts/generate_two_axis_report.py
    python3 scripts/generate_two_axis_report.py --start 2026-02-09 --end 2026-02-16
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier_v2.two_axis_classifier import TwoAxisClassifier
from src.storage.data_store import ClassifiedDataStore
from src.reporter.two_axis_analytics import TwoAxisAnalytics
from src.reporter.two_axis_report_generator import TwoAxisReportGenerator
from src.integrations.notion_client import NotionReportClient
from src.integrations.slack_client import SlackNotifier
from src.utils.two_axis_excel_exporter import export_two_axis_to_excel
from src.classifier_v2.sub_theme_analyzer import SubThemeAnalyzer, save_sub_themes


# 2축 데이터는 별도 디렉토리에 저장 (1축과 분리)
TWO_AXIS_DATA_DIR = "./data/classified_data_two_axis"


def generate_week_data(start_date, end_date, classifier, master_info=None):
    """특정 주간의 2축 분류 데이터 생성"""
    print(f"\n{'='*60}")
    print(f"  {start_date} ~ {end_date} 데이터 생성 (2축)")
    print('='*60)

    # 캐시 확인
    cache_path = os.path.join(TWO_AXIS_DATA_DIR, f"{start_date}.json")
    if os.path.exists(cache_path):
        import json
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

    # 2축 분류
    print("  2축 분류 중...")
    classified_letters = classifier.classify_batch(letters, "message") if letters else []
    classified_posts = classifier.classify_batch(posts, "textBody") if posts else []

    # 캐시 저장
    import json
    os.makedirs(TWO_AXIS_DATA_DIR, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="2축 분류 주간 리포트 생성")
    parser.add_argument("--start", default="2026-02-09", help="대상 주간 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-16", help="대상 주간 종료일 (exclusive)")
    parser.add_argument("--prev-start", default=None, help="전주 시작일 (미지정시 자동 계산)")
    parser.add_argument("--prev-end", default=None, help="전주 종료일 (미지정시 자동 계산)")
    parser.add_argument("--skip-notion", action="store_true", help="Notion 업로드 건너뛰기")
    parser.add_argument("--skip-slack", action="store_true", help="Slack 알림 건너뛰기")
    args = parser.parse_args()

    target_start = args.start
    target_end = args.end

    # 전주 날짜 자동 계산
    if args.prev_start and args.prev_end:
        prev_start = args.prev_start
        prev_end = args.prev_end
    else:
        from datetime import datetime, timedelta
        start_dt = datetime.strptime(target_start, "%Y-%m-%d")
        prev_start = (start_dt - timedelta(days=7)).strftime("%Y-%m-%d")
        prev_end = target_start

    print("="*60)
    print("  2축 분류 체계 주간 리포트 생성")
    print(f"  대상: {target_start} ~ {target_end}")
    print(f"  전주: {prev_start} ~ {prev_end}")
    print("="*60)

    # 0. 분류기 초기화 (한 번만)
    print("\n[0단계] 2축 분류기 초기화")
    classifier = TwoAxisClassifier()

    # 1. 전주 데이터
    print("\n[1단계] 전주 데이터 생성")
    prev_letters, prev_posts = generate_week_data(prev_start, prev_end, classifier)

    # 2. 대상 주간 데이터
    print("\n[2단계] 대상 주간 데이터 생성")
    classified_letters, classified_posts = generate_week_data(
        target_start, target_end, classifier
    )

    if not classified_letters and not classified_posts:
        print("\n  대상 주간 데이터가 없어 리포트를 생성할 수 없습니다.")
        return

    # 3. 통계 분석
    print(f"\n[3단계] 2축 통계 분석")
    analytics = TwoAxisAnalytics()
    stats = analytics.analyze_weekly_data(
        classified_letters,
        classified_posts,
        previous_letters=prev_letters,
        previous_posts=prev_posts,
    )

    total = stats["total_stats"]["this_week"]
    print(f"  전체: 편지 {total['letters']}건, 게시글 {total['posts']}건")

    spikes = stats.get("negative_spike_masters", [])
    drops = stats.get("negative_drop_masters", [])
    if spikes:
        print(f"  부정 급증 마스터: {', '.join(s['master'] for s in spikes)}")
    if drops:
        print(f"  부정 개선 마스터: {', '.join(d['master'] for d in drops)}")

    # 3.5. 서브 테마 분석 (서비스 이슈 클러스터링 + 특이 패턴 요약)
    print(f"\n[3.5단계] 서브 테마 분석")
    sub_analyzer = SubThemeAnalyzer()
    sub_themes = sub_analyzer.analyze(classified_letters, classified_posts, stats)

    service_clusters = sub_themes.get("service_clusters", {})
    notable_patterns = sub_themes.get("notable_patterns", [])

    if service_clusters:
        print(f"  서비스 이슈 클러스터: {len(service_clusters)}개")
        for cid, info in service_clusters.items():
            print(f"    [{info['label']}] {info['count']}건")
    if notable_patterns:
        for p in notable_patterns:
            print(f"  [{p['topic']}] 부정 {p['negative_count']}건 — 패턴 요약 완료")

    # 서브 테마 저장 (주간 증감 비교용)
    sub_theme_path = save_sub_themes(sub_themes, target_start)
    print(f"  서브 테마 저장: {sub_theme_path}")

    # stats에 서브 테마 추가 (리포트 생성에 사용)
    stats["sub_themes"] = sub_themes

    # 4. 리포트 생성
    print(f"\n[4단계] 2축 리포트 생성")
    output_dir = "./reports"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"two_axis_report_{target_start}.md"
    output_path = os.path.join(output_dir, output_filename)

    generator = TwoAxisReportGenerator()
    report = generator.generate_report(
        stats, target_start, target_end, output_path=output_path
    )
    print(f"  리포트 생성 완료: {output_path}")

    # 5. 엑셀 파일 생성
    print(f"\n[5단계] 엑셀 파일 생성")
    excel_filename = f"two_axis_data_{target_start}.xlsx"
    excel_path = os.path.join(output_dir, excel_filename)
    export_two_axis_to_excel(classified_letters, classified_posts, excel_path)
    print(f"  엑셀 생성: {excel_path}")

    # 6. Notion 업로드
    notion_url = None
    if not args.skip_notion:
        print(f"\n[6단계] Notion 업로드")
        try:
            notion_client = NotionReportClient()
            from datetime import datetime, timedelta
            start_formatted = datetime.strptime(target_start, "%Y-%m-%d").strftime("%Y.%m.%d")
            end_dt = datetime.strptime(target_end, "%Y-%m-%d")
            end_formatted = (end_dt - timedelta(days=1)).strftime("%m.%d")
            page_title = f"이용자 반응 리포트 [2축] ({start_formatted} ~ {end_formatted})"

            page_info = notion_client.create_report_page(
                title=page_title,
                markdown_content=report,
                start_date=target_start,
                end_date=target_end,
            )
            notion_url = page_info["url"]
            print(f"  Notion 페이지: {notion_url}")
        except Exception as e:
            print(f"  Notion 업로드 실패: {e}")
    else:
        print("\n[6단계] Notion 업로드 — 건너뜀")

    # 7. Slack 알림
    if not args.skip_slack and notion_url:
        print(f"\n[7단계] Slack 알림 전송")
        try:
            slack_client = SlackNotifier()
            week_label = SlackNotifier.get_week_label(target_start)

            result = slack_client.send_report_notification(
                week_label=f"{week_label} [2축]",
                start_date=target_start,
                end_date=target_end,
                notion_url=notion_url,
            )

            if result.get("ok"):
                print("  Slack 전송 완료")
                message_ts = result.get("message_ts")
                if message_ts and os.path.exists(excel_path):
                    file_result = slack_client.upload_file_to_thread(
                        file_path=excel_path,
                        thread_ts=message_ts,
                        title=f"2축 원본 데이터 ({target_start})",
                        comment="2축(Topic x Sentiment) 분류된 원본 데이터입니다.",
                    )
                    if file_result.get("ok"):
                        print("  엑셀 업로드 완료")
            else:
                print(f"  Slack 전송 실패: {result.get('error')}")
        except Exception as e:
            print(f"  Slack 전송 실패: {e}")
    else:
        print("\n[7단계] Slack 알림 — 건너뜀")

    print()
    print("="*60)
    print("  2축 주간 리포트 생성 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
