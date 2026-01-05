"""빠른 재분류 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(line_buffering=True)

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier.vector_classifier import VectorContentClassifier
from src.storage.data_store import ClassifiedDataStore

def classify_week(start_date, end_date, data_store, classifier, master_info):
    """주간 데이터 분류"""
    print(f"\n📅 {start_date} ~ {end_date}", flush=True)

    if data_store.exists(start_date):
        print("  ✓ 이미 존재", flush=True)
        data = data_store.load_weekly_data(start_date)
        return data['letters'], data['posts']

    print("  BigQuery 조회...", flush=True)
    client = BigQueryClient()
    query = WeeklyDataQuery(client)
    weekly_data = query.get_weekly_data(start_date, end_date)

    letters = weekly_data['letters']
    posts = weekly_data['posts']
    print(f"  편지 {len(letters)}건, 게시글 {len(posts)}건", flush=True)

    if not letters and not posts:
        return [], []

    # 마스터 정보 추가
    board_query = f"""
    SELECT _id as boardId, masterId
    FROM `{client.project_id}.us_plus.postboards`
    """
    board_to_master = {b['boardId']: b['masterId'] for b in client.execute_query(board_query)}

    for item in letters:
        master_id = item.get('masterId')
        if master_id and master_id in master_info:
            item['masterName'] = master_info[master_id]['displayName']
            item['masterClubName'] = master_info[master_id]['clubName']

    for item in posts:
        board_id = item.get('postBoardId')
        actual_master_id = board_to_master.get(board_id, board_id)
        if actual_master_id and actual_master_id in master_info:
            item['masterName'] = master_info[actual_master_id]['displayName']
            item['masterClubName'] = master_info[actual_master_id]['clubName']

    print("  분류 중...", flush=True)
    classified_letters = classifier.classify_batch(letters, "message") if letters else []
    classified_posts = classifier.classify_batch(posts, "textBody") if posts else []

    print("  저장 중...", flush=True)
    data_store.save_weekly_data(start_date, end_date, classified_letters, classified_posts)
    print(f"  ✓ 완료", flush=True)

    return classified_letters, classified_posts


def main():
    print("=" * 50, flush=True)
    print("빠른 재분류", flush=True)
    print("=" * 50, flush=True)

    data_store = ClassifiedDataStore()
    classifier = VectorContentClassifier()

    # 마스터 정보 조회
    print("\n마스터 정보 조회...", flush=True)
    client = BigQueryClient()
    query = WeeklyDataQuery(client)
    master_info = query.get_master_info()
    print(f"✓ {len(master_info)}개 마스터", flush=True)

    # 전주
    classify_week("2025-12-22", "2025-12-28", data_store, classifier, master_info)

    # 대상 주
    classify_week("2025-12-29", "2026-01-04", data_store, classifier, master_info)

    # 결과 확인
    print("\n" + "=" * 50, flush=True)
    print("분류 결과:", flush=True)

    import json
    with open('data/classified_data/2025-12-29.json') as f:  # 2025-12-29 ~ 2026-01-04
        data = json.load(f)

    categories = {}
    for item in data.get('letters', []) + data.get('posts', []):
        cat = item.get('classification', {}).get('category', '미분류')
        categories[cat] = categories.get(cat, 0) + 1

    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}건", flush=True)

    print("\n✅ 완료!", flush=True)


if __name__ == "__main__":
    main()
