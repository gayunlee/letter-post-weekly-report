"""주간 리포트 생성 메인 스크립트"""
import sys
import os
from datetime import datetime
from typing import List, Dict, Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.classifier.vector_classifier import VectorContentClassifier
from src.vectorstore.chroma_store import ChromaVectorStore
from src.reporter.analytics import WeeklyAnalytics
from src.reporter.report_generator import ReportGenerator
from src.storage.data_store import ClassifiedDataStore


# 서비스 공지글 필터링 키워드
FILTER_KEYWORDS = [
    "channel.io",
    "어떤 채팅방도 운영하지 않습니다",
    "어떤 채팅방 · 밴드도 운영하지 않습니다",
    "문의하기:",
]


def filter_service_notices(items: List[Dict[str, Any]], content_field: str = "message") -> List[Dict[str, Any]]:
    """
    서비스 공지글 필터링

    Args:
        items: 필터링할 아이템 리스트
        content_field: 콘텐츠 필드명

    Returns:
        필터링된 아이템 리스트
    """
    filtered = []
    removed_count = 0

    for item in items:
        content = item.get(content_field, "") or ""
        is_notice = any(keyword in content for keyword in FILTER_KEYWORDS)

        if is_notice:
            removed_count += 1
        else:
            filtered.append(item)

    if removed_count > 0:
        print(f"  ⚠️  서비스 공지글 {removed_count}건 필터링됨")

    return filtered


def main():
    """주간 리포트 생성 메인 프로세스"""

    print("=" * 60)
    print("📊 주간 리포트 자동 생성 시스템 (증분 처리)")
    print("=" * 60)
    print()

    # 0. 데이터 저장소 초기화
    data_store = ClassifiedDataStore(
        classified_data_dir=os.getenv("CLASSIFIED_DATA_DIR", "./data/classified_data"),
        stats_dir=os.getenv("STATS_DIR", "./data/stats")
    )

    # 날짜 범위 계산
    start_date, end_date = WeeklyDataQuery.get_last_week_range()
    print(f"📅 대상 기간: {start_date} ~ {end_date}")
    print()

    # 1. 저장된 분류 결과 확인
    print("1️⃣  분류 데이터 확인")
    print("-" * 60)

    if data_store.exists(start_date):
        print(f"✓ 저장된 분류 결과 발견!")
        print(f"  로드 중...")

        classified_data = data_store.load_weekly_data(start_date)
        classified_letters = classified_data['letters']
        classified_posts = classified_data['posts']

        print(f"✓ 편지글 {len(classified_letters)}건 로드")
        print(f"✓ 게시글 {len(classified_posts)}건 로드")
        print(f"⚡ 재분류 생략 (기존 데이터 재사용)")
    else:
        print(f"❌ 저장된 분류 결과 없음")
        print(f"  BigQuery 조회 및 분류 시작...")
        print()

        # BigQuery 데이터 조회
        print("  📊 BigQuery 데이터 조회")
        print("  " + "-" * 58)

        client = BigQueryClient()
        query_with_client = WeeklyDataQuery(client)

        # 마스터 정보 조회
        print("  마스터 정보 조회 중...")
        master_info = query_with_client.get_master_info()
        print(f"  ✓ {len(master_info)}개 마스터/게시판 정보 로드")

        weekly_data = query_with_client.get_weekly_data(start_date, end_date)
        letters = weekly_data['letters']
        posts = weekly_data['posts']

        print(f"  ✓ 편지글 {len(letters)}건 조회")
        print(f"  ✓ 게시글 {len(posts)}건 조회")
        print()

        if not letters and not posts:
            print("  ❌ 데이터가 없어 리포트를 생성할 수 없습니다.")
            return

        # 게시판 -> 마스터 매핑 조회
        board_to_master_query = f"""
        SELECT _id as boardId, masterId
        FROM `{client.project_id}.us_plus.postboards`
        """
        board_to_master = {b['boardId']: b['masterId']
                           for b in client.execute_query(board_to_master_query)}

        # 편지글: 마스터 이름 추가
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
        for item in posts:
            board_id = item.get('postBoardId')
            actual_master_id = board_to_master.get(board_id, board_id)

            if actual_master_id and actual_master_id in master_info:
                item['masterName'] = master_info[actual_master_id]['displayName']
                item['masterClubName'] = master_info[actual_master_id]['clubName']
                item['actualMasterId'] = actual_master_id
            else:
                item['masterName'] = 'Unknown'
                item['masterClubName'] = 'Unknown'
                item['actualMasterId'] = actual_master_id or 'unknown'

        # 서비스 공지글 필터링
        print("  🔍 서비스 공지글 필터링")
        letters = filter_service_notices(letters, content_field="message")
        posts = filter_service_notices(posts, content_field="textBody")
        print()

        # 콘텐츠 분류 (벡터 기반)
        print("  📝 콘텐츠 분류 (벡터 유사도 기반)")
        print("  " + "-" * 58)

        classifier = VectorContentClassifier()

        if letters:
            print(f"  편지글 {len(letters)}건 분류 중...")
            classified_letters = classifier.classify_batch(
                letters,
                content_field="message"
            )
            print(f"  ✓ 편지글 분류 완료")
        else:
            classified_letters = []

        if posts:
            print(f"  게시글 {len(posts)}건 분류 중...")
            classified_posts = classifier.classify_batch(
                posts,
                content_field="textBody"
            )
            print(f"  ✓ 게시글 분류 완료")
        else:
            classified_posts = []

        print()

        # 분류 결과 저장 (2-Tier)
        print("  💾 분류 결과 저장 (2-Tier)")
        print("  " + "-" * 58)

        data_store.save_weekly_data(
            start_date,
            end_date,
            classified_letters,
            classified_posts
        )

        print(f"  ✓ 전체 데이터 저장: data/classified_data/{start_date}.json")
        print(f"  ✓ 통계 요약 저장: data/stats/{start_date}.json")

    print()

    # 3. 벡터 스토어에 저장 (선택)
    print("3️⃣  벡터 스토어 저장")
    print("-" * 60)

    try:
        store = ChromaVectorStore(
            collection_name=f"week_{start_date}",
            persist_directory="./chroma_db"
        )

        # 기존 데이터 초기화
        store.reset()

        # 데이터 저장
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

        print(f"✓ {total_added}건 벡터 스토어에 저장 완료")
    except Exception as e:
        print(f"⚠️  벡터 스토어 저장 실패: {str(e)}")
        print("   (리포트 생성은 계속 진행됩니다)")

    print()

    # 2. 전주 데이터 로드 (전주 비교)
    print("2️⃣  전주 데이터 로드")
    print("-" * 60)

    prev_start, prev_end = WeeklyDataQuery.get_previous_week_range()
    print(f"📅 전주 기간: {prev_start} ~ {prev_end}")

    previous_letters = None
    previous_posts = None

    if data_store.exists(prev_start):
        try:
            previous_data = data_store.load_weekly_data(prev_start)
            previous_letters = previous_data['letters']
            previous_posts = previous_data['posts']
            print(f"✓ 전주 데이터 로드: 편지 {len(previous_letters)}건, 게시글 {len(previous_posts)}건")
        except Exception as e:
            print(f"⚠️  전주 데이터 로드 실패: {str(e)}")
            print("   (전주 비교 없이 진행됩니다)")
    else:
        print(f"❌ 전주 데이터 없음 (첫 실행 또는 전주 데이터 미생성)")

    print()

    # 4. 통계 분석
    print("4️⃣  통계 분석 (전주 비교)")
    print("-" * 60)

    analytics = WeeklyAnalytics()
    stats = analytics.analyze_weekly_data(
        classified_letters,
        classified_posts,
        previous_letters=previous_letters,
        previous_posts=previous_posts
    )

    total = stats["total_stats"]["this_week"]
    print(f"✓ 전체 통계: 편지 {total['letters']}건, 게시글 {total['posts']}건")

    category_stats = stats["category_stats"]
    print(f"✓ 카테고리별 통계:")
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {category}: {count}건")

    master_stats = stats["master_stats"]
    print(f"✓ 마스터별 통계: {len(master_stats)}개 마스터")

    feedbacks = stats.get("service_feedbacks", [])
    print(f"✓ 서비스 피드백: {len(feedbacks)}건")

    print()

    # 5. 리포트 생성
    print("5️⃣  리포트 생성")
    print("-" * 60)

    # 출력 디렉토리 설정
    output_dir = os.getenv("REPORT_OUTPUT_DIR", "./reports")
    os.makedirs(output_dir, exist_ok=True)

    # 파일명 생성 (YYYY-MM-DD 형식)
    output_filename = f"weekly_report_{start_date}.md"
    output_path = os.path.join(output_dir, output_filename)

    generator = ReportGenerator()

    print("리포트 생성 중...")
    report, slack_summary = generator.generate_report(
        stats,
        start_date,
        end_date,
        output_path=output_path
    )

    print(f"✓ 리포트 생성 완료")
    print(f"✓ 저장 위치: {output_path}")
    print(f"✓ 슬랙 요약:\n{slack_summary}")
    print()

    # 6. 완료
    print("=" * 60)
    print("✅ 주간 리포트 생성 완료!")
    print("=" * 60)
    print()

    # 리포트 미리보기 (처음 30줄)
    print("📄 리포트 미리보기:")
    print("-" * 60)
    lines = report.split('\n')
    for line in lines[:30]:
        print(line)

    if len(lines) > 30:
        print("\n... (전체 내용은 생성된 파일을 확인하세요)")


if __name__ == "__main__":
    main()
