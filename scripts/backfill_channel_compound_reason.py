"""채널톡 기존 분류 결과에 복합 맥락 필드를 보강.

기존 topic/subtag를 재분류하지 않고, 저장된 text와 topic을 바탕으로
is_compound, compound_reason, summary, tags만 추출한다.

사용 예:
    python3 scripts/backfill_channel_compound_reason.py --start 2026-04-27 --end 2026-05-04 --limit 20 --dry-run
    python3 scripts/backfill_channel_compound_reason.py --start 2026-04-27 --end 2026-05-04 --limit 100 --write
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import boto3
from google.cloud import bigquery

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bigquery.client import BigQueryClient
from src.bigquery.writer import BigQueryWriter, CHANNEL_TALK_SCHEMA
from src.classifier_v4.subtag_detail import empty_subtag_detail, normalize_subtag_detail
from src.classifier_v4.subtag_prompt import SUBTAG_SYSTEM_PROMPT
from src.classifier_v4.two_depth_classifier import WORKFLOW_NOISE, strip_workflow_buttons


MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
WORKFLOW_NOISE_SQL_PATTERN = "|".join(
    ["👆🏻.*"] + [re.escape(text) for text in WORKFLOW_NOISE]
)


def parse_json_object(raw: str) -> Dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(raw[start:end])
    return json.loads(raw)


class ChannelCompoundEnricher:
    def __init__(self, region: str = "us-west-2", model_id: str = MODEL_ID):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def enrich(self, text: str, topic: str) -> Dict[str, Any]:
        if topic not in SUBTAG_SYSTEM_PROMPT:
            return empty_subtag_detail()

        prompt_text = strip_workflow_buttons(text or "")[:500]
        if not prompt_text:
            return empty_subtag_detail()

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 250,
                "system": SUBTAG_SYSTEM_PROMPT[topic],
                "messages": [{"role": "user", "content": prompt_text}],
            }),
        )
        result = json.loads(response["body"].read())
        raw = result["content"][0]["text"].strip()
        try:
            parsed = parse_json_object(raw)
        except (ValueError, json.JSONDecodeError):
            logging.warning("LLM 응답 JSON 파싱 실패, empty detail로 대체: %s", raw[:120])
            return empty_subtag_detail()
        return normalize_subtag_detail(parsed, topic)


def fetch_rows(bq: BigQueryClient, start: str, end: str, limit: int) -> List[Dict[str, Any]]:
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT chat_id, text, topic, subtag, pipeline_date
    FROM `{bq.project_id}.voc_labelled.channel_talk`
    WHERE pipeline_date >= DATE "{start}"
      AND pipeline_date < DATE "{end}"
      AND text IS NOT NULL
      AND has_free_text = TRUE
      AND (summary IS NULL OR summary = "")
      AND topic IN ("결제·구독", "콘텐츠·수강", "기술·오류")
      AND TRIM(REGEXP_REPLACE(
        TRIM(text),
        r'(?m)^\\s*({WORKFLOW_NOISE_SQL_PATTERN})\\s*$',
        ''
      )) != ''
    ORDER BY pipeline_date DESC, chat_id
    {limit_clause}
    """
    return bq.execute_query(query)


def update_rows(bq: BigQueryClient, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    writer = BigQueryWriter(bq.client)
    table_id = writer._ensure_table("channel_talk", CHANNEL_TALK_SCHEMA)

    temp_table = f"{bq.project_id}.voc_labelled._channel_compound_updates"
    bq.client.delete_table(temp_table, not_found_ok=True)
    schema = [
        field for field in CHANNEL_TALK_SCHEMA
        if field.name in {"chat_id", "pipeline_date", "is_compound", "compound_reason", "summary", "tags"}
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
      is_compound = S.is_compound,
      compound_reason = S.compound_reason,
      summary = S.summary,
      tags = S.tags
    """
    bq.client.query(merge_sql).result()
    bq.client.delete_table(temp_table, not_found_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="채널톡 compound_reason 보강")
    parser.add_argument("--start", required=True, help="포함 시작일 YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="제외 종료일 YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.dry_run and args.write:
        raise SystemExit("--dry-run and --write cannot be used together")
    if not args.dry_run and not args.write:
        raise SystemExit("choose --dry-run or --write")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bq = BigQueryClient()
    source_rows = fetch_rows(bq, args.start, args.end, args.limit)
    logging.info("대상 rows: %s", len(source_rows))

    enricher = ChannelCompoundEnricher(region=args.region)
    updates = []
    for idx, row in enumerate(source_rows, start=1):
        detail = enricher.enrich(row.get("text", ""), row.get("topic", ""))
        update = {
            "chat_id": row["chat_id"],
            "pipeline_date": row["pipeline_date"].isoformat()
            if hasattr(row["pipeline_date"], "isoformat")
            else str(row["pipeline_date"]),
            "is_compound": detail["is_compound"],
            "compound_reason": detail["compound_reason"],
            "summary": detail["summary"],
            "tags": detail["tags"],
        }
        updates.append(update)
        logging.info(
            "%s/%s %s | %s | compound=%s | %s",
            idx,
            len(source_rows),
            row.get("topic", ""),
            detail.get("subtag", ""),
            detail["is_compound"],
            detail["compound_reason"] or detail["summary"],
        )

    if args.write:
        update_rows(bq, updates)
        logging.info("BigQuery 업데이트 완료: %s rows", len(updates))
    else:
        print(json.dumps(updates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
