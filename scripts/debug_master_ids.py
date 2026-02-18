"""마스터 ID 매칭 확인"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery


def main():
    client = BigQueryClient()
    query = WeeklyDataQuery(client)

    # 게시글 샘플 조회
    print("게시글 샘플:")
    posts_query = f"""
    SELECT postBoardId, title
    FROM `{client.project_id}.us_plus.posts`
    WHERE isBlock = 'false' AND deleted = 'false'
    LIMIT 5
    """
    posts = client.execute_query(posts_query)
    for post in posts:
        print(f"  postBoardId: {post.get('postBoardId')}")

    print("\n마스터 ID 샘플:")
    masters = query.get_master_info()
    for master_id in list(masters.keys())[:5]:
        print(f"  masterId: {master_id} -> {masters[master_id]['displayName']}")

    # postBoardId와 매칭 확인
    print("\n매칭 확인:")
    for post in posts:
        board_id = post.get('postBoardId')
        if board_id in masters:
            print(f"  ✓ {board_id} -> {masters[board_id]['displayName']}")
        else:
            print(f"  ✗ {board_id} -> NOT FOUND")

    # postboards 테이블 확인
    print("\n\npostboards 테이블 확인:")
    boards_query = f"""
    SELECT _id, name, masterId
    FROM `{client.project_id}.us_plus.postboards`
    LIMIT 10
    """
    try:
        boards = client.execute_query(boards_query)
        print("postboards 구조:")
        for board in boards:
            print(f"  boardId: {board.get('_id')} -> masterId: {board.get('masterId')}, name: {board.get('name')}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
