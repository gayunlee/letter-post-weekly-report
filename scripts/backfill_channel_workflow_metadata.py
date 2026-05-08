"""채널톡 voc_labelled rows에 워크플로우 메타데이터만 보강.

기존 topic/subtag는 유지하고, 원본 messages를 재조회하여
workflow_buttons / has_free_text / interaction_type만 MERGE.

일자별로 처리해서 daily pipeline 의미(KST 일자 단위)를 보존한다.

사용 예:
    python3 scripts/backfill_channel_workflow_metadata.py --start 2026-04-27 --end 2026-05-04 --dry-run
    python3 scripts/backfill_channel_workflow_metadata.py --start 2026-04-27 --end 2026-05-04 --write
"""
import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from google.cloud import bigquery

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bigquery.channel_preprocessor import build_chat_items
from src.bigquery.channel_queries import ChannelQueryService
from src.bigquery.client import BigQueryClient
from src.bigquery.writer import BigQueryWriter, CHANNEL_TALK_SCHEMA


def daterange(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    cur = s
    while cur < e:
        out.append(cur.isoformat())
        cur = cur + timedelta(days=1)
    return out


def build_metadata_rows(bq: BigQueryClient, day: str) -> List[Dict[str, Any]]:
    """하루치 messages → chat_id별 metadata."""
    next_day = (datetime.strptime(day, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()
    cq = ChannelQueryService(bq)
    messages, chat_states = cq.get_weekly_conversations(day, next_day)
    items = build_chat_items(messages, chat_states=chat_states)
    rows = []
    for item in items:
        rows.append({
            "chat_id": item["chatId"],
            "pipeline_date": day,
            "workflow_buttons": item.get("workflow_buttons", []),
            "has_free_text": bool(item.get("has_free_text", False)),
            "interaction_type": item.get("interaction_type", ""),
        })
    return rows


def merge_rows(bq: BigQueryClient, rows: List[Dict[str, Any]], dry_run: bool) -> int:
    if not rows:
        return 0

    if dry_run:
        return len(rows)

    writer = BigQueryWriter(bq.client)
    table_id = writer._ensure_table("channel_talk", CHANNEL_TALK_SCHEMA)

    temp_table = f"{bq.project_id}.voc_labelled._channel_metadata_updates"
    bq.client.delete_table(temp_table, not_found_ok=True)
    schema = [
        field for field in CHANNEL_TALK_SCHEMA
        if field.name in {"chat_id", "pipeline_date", "workflow_buttons",
                          "has_free_text", "interaction_type"}
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    load_job = bq.client.load_table_from_json(rows, temp_table, job_config=job_config)
    load_job.result()

    merge_sql = f"""
    MERGE `{table_id}` T
    USING `{temp_table}` S
    ON T.chat_id = S.chat_id AND T.pipeline_date = S.pipeline_date
    WHEN MATCHED THEN UPDATE SET
      workflow_buttons = S.workflow_buttons,
      has_free_text = S.has_free_text,
      interaction_type = S.interaction_type
    """
    bq.client.query(merge_sql).result()
    bq.client.delete_table(temp_table, not_found_ok=True)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="채널톡 workflow metadata 보강")
    parser.add_argument("--start", required=True, help="포함 시작일 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="제외 종료일 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.dry_run and args.write:
        raise SystemExit("--dry-run and --write cannot be used together")
    if not args.dry_run and not args.write:
        raise SystemExit("choose --dry-run or --write")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bq = BigQueryClient()

    total = 0
    for day in daterange(args.start, args.end):
        rows = build_metadata_rows(bq, day)
        wf_only = sum(1 for r in rows if r["interaction_type"] == "workflow_only")
        mixed = sum(1 for r in rows if r["interaction_type"] == "mixed")
        free_text = sum(1 for r in rows if r["interaction_type"] == "free_text")
        logging.info(
            "%s: chats=%s (workflow_only=%s, mixed=%s, free_text=%s)",
            day, len(rows), wf_only, mixed, free_text,
        )
        merged = merge_rows(bq, rows, dry_run=args.dry_run)
        total += merged

    if args.write:
        logging.info("BigQuery metadata MERGE 완료: 총 %s rows", total)
    else:
        logging.info("DRY RUN 완료: 총 %s rows (실제 쓰기 없음)", total)


if __name__ == "__main__":
    main()
