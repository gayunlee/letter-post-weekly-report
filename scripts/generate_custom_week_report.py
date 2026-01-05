"""특정 주간 리포트 생성 스크립트 (날짜 지정)"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier.vector_classifier import VectorContentClassifier
from src.storage.data_store import ClassifiedDataStore
from src.vectorstore.chroma_store import ChromaVectorStore
from src.reporter.analytics import WeeklyAnalytics
from src.reporter.report_generator import ReportGenerator


def generate_week_data(start_date, end_date, data_store, master_info=None):
    """특정 주간의 데이터 생성"""
    print(f"\n{'='*60}")
    print(f"📅 {start_date} ~ {end_date} 데이터 생성")
    print('='*60)

    if data_store.exists(start_date):
        print("✓ 이미 존재하는 데이터 - 건너뜀")
        data = data_store.load_weekly_data(start_date)
        return data['letters'], data['posts']

    print("BigQuery 조회 중...")
    client = BigQueryClient()
    query = WeeklyDataQuery(client)

    # 마스터 정보 조회 (한 번만)
    if master_info is None:
        print("마스터 정보 조회 중...")
        master_info = query.get_master_info()
        print(f"✓ {len(master_info)}개 마스터 정보 로드")

    weekly_data = query.get_weekly_data(start_date, end_date)

    letters = weekly_data['letters']
    posts = weekly_data['posts']
    print(f"✓ 편지 {len(letters)}건, 게시글 {len(posts)}건 조회")

    if not letters and not posts:
        print("❌ 데이터 없음")
        return [], []

    # 마스터 이름 추가 및 실제 masterId 설정
    for item in letters:
        master_id = item.get('masterId')
        if master_id and master_id in master_info:
            item['masterName'] = master_info[master_id]['displayName']
            item['masterClubName'] = master_info[master_id]['clubName']
            item['actualMasterId'] = master_id
        else:
            item['masterName'] = 'Unknown'
            item['masterClubName'] = 'Unknown'
            item['actualMasterId'] = master_id or 'unknown'

    # 게시글: postBoardId를 실제 masterId로 변환
    client_for_boards = BigQueryClient()
    board_to_master_query = f"""
    SELECT _id as boardId, masterId
    FROM `{client_for_boards.project_id}.us_plus.postboards`
    """
    board_to_master = {b['boardId']: b['masterId']
                       for b in client_for_boards.execute_query(board_to_master_query)}

    for item in posts:
        board_id = item.get('postBoardId')
        # postBoardId를 실제 masterId로 변환
        actual_master_id = board_to_master.get(board_id, board_id)

        if actual_master_id and actual_master_id in master_info:
            item['masterName'] = master_info[actual_master_id]['displayName']
            item['masterClubName'] = master_info[actual_master_id]['clubName']
            item['actualMasterId'] = actual_master_id
        else:
            item['masterName'] = 'Unknown'
            item['masterClubName'] = 'Unknown'
            item['actualMasterId'] = actual_master_id or 'unknown'

    print("분류 중...")
    classifier = VectorContentClassifier()
    classified_letters = classifier.classify_batch(letters, "message") if letters else []
    classified_posts = classifier.classify_batch(posts, "textBody") if posts else []

    print("저장 중...")
    data_store.save_weekly_data(start_date, end_date, classified_letters, classified_posts)
    print(f"✓ 저장 완료: {start_date}.json")

    return classified_letters, classified_posts


def main():
    # 대상 주간
    target_start = "2025-12-15"
    target_end = "2025-12-22"  # 12-21 다음날까지 (exclusive)

    # 전주
    prev_start = "2025-12-08"
    prev_end = "2025-12-15"

    print("="*60)
    print("📊 특정 주간 리포트 생성")
    print("="*60)

    data_store = ClassifiedDataStore()

    # 1. 전주 데이터 생성
    print("\n[1단계] 전주 데이터 생성")
    prev_letters, prev_posts = generate_week_data(prev_start, prev_end, data_store)

    # 2. 대상 주간 데이터 생성
    print("\n[2단계] 대상 주간 데이터 생성")
    classified_letters, classified_posts = generate_week_data(target_start, target_end, data_store)

    if not classified_letters and not classified_posts:
        print("\n❌ 대상 주간 데이터가 없어 리포트를 생성할 수 없습니다.")
        return

    # 3. 벡터 스토어 저장
    print(f"\n[3단계] 벡터 스토어 저장")
    try:
        store = ChromaVectorStore(
            collection_name=f"week_{target_start}",
            persist_directory="./chroma_db"
        )
        store.reset()

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

        print(f"✓ {total_added}건 벡터 스토어 저장 완료")
    except Exception as e:
        print(f"⚠️  벡터 스토어 저장 실패: {str(e)}")

    # 4. 통계 분석 (전주 비교)
    print(f"\n[4단계] 통계 분석 (전주 비교)")
    analytics = WeeklyAnalytics()
    stats = analytics.analyze_weekly_data(
        classified_letters,
        classified_posts,
        previous_letters=prev_letters,
        previous_posts=prev_posts
    )

    total = stats["total_stats"]["this_week"]
    print(f"✓ 전체 통계: 편지 {total['letters']}건, 게시글 {total['posts']}건")

    # 5. 리포트 생성
    print(f"\n[5단계] 리포트 생성")
    output_dir = "./reports"
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"weekly_report_{target_start}.md"
    output_path = os.path.join(output_dir, output_filename)

    generator = ReportGenerator()
    report = generator.generate_report(
        stats,
        target_start,
        target_end,
        output_path=output_path
    )

    print(f"✓ 리포트 생성 완료")
    print(f"✓ 저장 위치: {output_path}")
    print()

    print("="*60)
    print("✅ 주간 리포트 생성 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
