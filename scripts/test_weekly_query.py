"""주간 데이터 조회 테스트 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery


def main():
    """주간 데이터 조회 테스트"""

    print("BigQuery 연결 중...")
    client = BigQueryClient()
    query = WeeklyDataQuery(client)

    print(f"✓ 프로젝트 ID: {client.project_id}\n")

    # 지난 주 날짜 범위 계산
    start_date, end_date = query.get_last_week_range()
    print("=" * 60)
    print(f"📅 조회 기간: {start_date} ~ {end_date}")
    print("=" * 60)
    print()

    # 편지글 조회
    print("📧 편지글 조회 중...")
    letters = query.get_weekly_letters(start_date, end_date)
    print(f"✓ 총 {len(letters)}건의 편지글 발견\n")

    if letters:
        print("샘플 편지글 (최대 3건):")
        for i, letter in enumerate(letters[:3], 1):
            print(f"\n[{i}번째 편지글]")
            print(f"  ID: {letter.get('_id')}")
            print(f"  마스터 ID: {letter.get('masterId')}")
            print(f"  생성일: {letter.get('createdAt')}")
            # 메시지 내용이 길면 잘라서 표시
            message = letter.get('message', '')
            if len(message) > 100:
                message = message[:100] + '...'
            print(f"  내용: {message}")

    print("\n" + "=" * 60)

    # 게시글 조회
    print("📝 게시글 조회 중...")
    posts = query.get_weekly_posts(start_date, end_date)
    print(f"✓ 총 {len(posts)}건의 게시글 발견\n")

    if posts:
        print("샘플 게시글 (최대 3건):")
        for i, post in enumerate(posts[:3], 1):
            print(f"\n[{i}번째 게시글]")
            print(f"  ID: {post.get('_id')}")
            print(f"  제목: {post.get('title')}")
            print(f"  생성일: {post.get('createdAt')}")
            print(f"  좋아요: {post.get('likeCount', 0)}, 댓글: {post.get('replyCount', 0)}")
            # 본문 내용이 길면 잘라서 표시
            body = post.get('textBody') or post.get('body', '')
            if len(body) > 100:
                body = body[:100] + '...'
            print(f"  내용: {body}")

    print("\n" + "=" * 60)
    print("✓ 주간 데이터 조회 테스트 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
