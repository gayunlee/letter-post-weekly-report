"""일간 VOC 분류 파이프라인

매일 전날 데이터를 BigQuery에서 조회 → 분류 → voc_labelled에 저장 → Slack 알림

사용법:
    python scripts/run_daily_pipeline.py                # 어제 데이터
    python scripts/run_daily_pipeline.py --date 2026-03-22  # 특정 날짜
    python scripts/run_daily_pipeline.py --dry-run       # 테스트 (5건만)
"""
import sys
import os
import re
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.bigquery.client import BigQueryClient
from src.bigquery.queries import WeeklyDataQuery
from src.bigquery.writer import BigQueryWriter
from src.classifier_v5.bedrock_classifier import BedrockV5Classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 서비스 공지 필터링 키워드
FILTER_KEYWORDS = [
    "운영 안내", "공지사항", "서비스 점검", "시스템 점검",
    "업데이트 안내", "이벤트 안내", "서비스 안내",
]


def filter_service_notices(items):
    """서비스 공지 필터링"""
    filtered = []
    for item in items:
        text = item.get("message") or item.get("textBody") or item.get("body", "")
        if any(kw in text[:100] for kw in FILTER_KEYWORDS):
            continue
        filtered.append(item)
    return filtered


def enrich_master_info(items, master_info, field="masterId"):
    """마스터 이름 매핑"""
    for item in items:
        mid = item.get(field, "")
        if field == "postBoardId":
            mid = item.get("postBoardId", "")
        if mid in master_info:
            info = master_info[mid]
            item["masterName"] = info.get("displayName") or info.get("name", "Unknown")
            item["masterClubName"] = info.get("clubName", "")
    return items


def send_slack_alert(items, pipeline_date):
    """부정 급증 시 Slack 알림"""
    try:
        from src.integrations.slack_client import SlackNotifier
        slack = SlackNotifier()
    except Exception:
        logger.warning("Slack 연동 실패 — 알림 건너뜀")
        return

    # 마스터별 부정률 계산
    master_stats = {}
    for item in items:
        master = re.sub(r"\d+$", "", item.get("masterName", "Unknown")).strip()
        cls = item.get("classification", {})
        if master not in master_stats:
            master_stats[master] = {"total": 0, "neg": 0}
        master_stats[master]["total"] += 1
        if cls.get("sentiment") == "부정":
            master_stats[master]["neg"] += 1

    # 부정률 30% 이상 + 5건 이상인 마스터 감지
    alerts = []
    for master, stats in master_stats.items():
        if stats["total"] >= 5:
            neg_r = stats["neg"] / stats["total"] * 100
            if neg_r >= 30:
                alerts.append(f"• *{master}*: 부정 {stats['neg']}건/{stats['total']}건 ({neg_r:.0f}%)")

    if alerts:
        msg = f"🚨 *[VOC 부정 감지] {pipeline_date}*\n\n" + "\n".join(alerts)
        try:
            slack._send_message(msg)
            logger.info(f"Slack 알림 발송: {len(alerts)}건")
        except Exception as e:
            logger.warning(f"Slack 발송 실패: {e}")
    else:
        logger.info("부정 급증 없음 — 알림 없음")


def main():
    parser = argparse.ArgumentParser(description="일간 VOC 분류 파이프라인")
    parser.add_argument("--date", help="대상 날짜 (YYYY-MM-DD, 기본: 어제)")
    parser.add_argument("--dry-run", action="store_true", help="테스트 모드 (5건만)")
    parser.add_argument("--skip-channel", action="store_true", help="채널톡 건너뛰기")
    parser.add_argument("--skip-slack", action="store_true", help="Slack 알림 건너뛰기")
    parser.add_argument("--workers", type=int, default=5, help="병렬 워커 수")
    args = parser.parse_args()

    # 날짜 설정
    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    next_date = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"=== VOC 일간 파이프라인 시작 ===")
    logger.info(f"대상 날짜: {target_date}")
    pipeline_start = time.time()

    # ── Phase 1: BigQuery 데이터 조회 ──
    logger.info("Phase 1: BigQuery 조회")
    bq_client = BigQueryClient()
    query = WeeklyDataQuery(bq_client)
    master_info = query.get_master_info()

    data = query.get_weekly_data(target_date, next_date)
    letters = data["letters"]
    posts = data["posts"]

    # 마스터 정보 매핑
    letters = enrich_master_info(letters, master_info, "masterId")
    posts = enrich_master_info(posts, master_info, "postBoardId")

    # 서비스 공지 필터링
    letters = filter_service_notices(letters)
    posts = filter_service_notices(posts)

    logger.info(f"  편지 {len(letters)}건, 게시글 {len(posts)}건")

    if args.dry_run:
        letters = letters[:3]
        posts = posts[:2]
        logger.info(f"  [DRY RUN] 편지 {len(letters)}건, 게시글 {len(posts)}건으로 제한")

    # ── Phase 2: 편지글/게시글 분류 (Bedrock Sonnet) ──
    logger.info("Phase 2: 편지글/게시글 분류 (Bedrock Sonnet v5 4분류)")
    classifier = BedrockV5Classifier(max_workers=args.workers)

    all_items = []
    if letters:
        logger.info(f"  편지 {len(letters)}건 분류 중...")
        classifier.classify_batch(letters, content_field="message")
        all_items.extend(letters)

    if posts:
        logger.info(f"  게시글 {len(posts)}건 분류 중...")
        classifier.classify_batch(posts, content_field="textBody")
        all_items.extend(posts)

    cost = classifier.get_cost_report()
    logger.info(f"  분류 완료: {cost['total_items']}건, 에러 {cost['errors']}건, ${cost['cost_usd']}")

    # ── Phase 3: 채널톡 분류 (KcELECTRA + Bedrock) ──
    channel_items = []
    if not args.skip_channel:
        logger.info("Phase 3: 채널톡 분류 (KcELECTRA + Bedrock)")
        try:
            from src.classifier_v4.bedrock_two_depth import BedrockTwoDepthClassifier
            ch_classifier = BedrockTwoDepthClassifier()

            # 채널톡 데이터 조회
            try:
                from src.bigquery.channel_queries import ChannelQueryService
                ch_query = ChannelQueryService(bq_client)
                ch_data = ch_query.get_daily_chats(target_date, next_date)

                from src.bigquery.channel_preprocessor import build_chat_items
                channel_items = build_chat_items(ch_data)
                logger.info(f"  채널톡 {len(channel_items)}건 조회")

                if args.dry_run:
                    channel_items = channel_items[:3]

                channel_items = ch_classifier.classify_batch(channel_items)
                logger.info(f"  채널톡 분류 완료: {len(channel_items)}건")
            except ImportError:
                logger.warning("  채널톡 쿼리 모듈 없음 — 건너뜀")
            except Exception as e:
                logger.warning(f"  채널톡 처리 실패: {e}")
        except Exception as e:
            logger.warning(f"  KcELECTRA 로드 실패: {e}")

    # ── Phase 4: BigQuery 저장 ──
    logger.info("Phase 4: BigQuery voc_labelled 저장")
    writer = BigQueryWriter(bq_client.client)

    lp_count = writer.write_letters_posts(all_items, target_date)
    ch_count = 0
    if channel_items:
        ch_count = writer.write_channel_talk(channel_items, target_date)

    # ── Phase 5: Slack 알림 ──
    if not args.skip_slack and all_items:
        logger.info("Phase 5: Slack 알림 체크")
        send_slack_alert(all_items, target_date)

    # ── 완료 ──
    elapsed = time.time() - pipeline_start
    logger.info(f"\n=== 파이프라인 완료 ===")
    logger.info(f"대상 날짜: {target_date}")
    logger.info(f"편지/게시글: {lp_count}건 저장")
    logger.info(f"채널톡: {ch_count}건 저장")
    logger.info(f"소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    logger.info(f"분류 비용: ${cost['cost_usd']}")


if __name__ == "__main__":
    main()
