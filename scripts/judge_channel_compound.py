"""채널톡 compound 분류 결과 LLM-as-judge 검수.

backfill 완료 후 voc_labelled.channel_talk에서 분류 결과를 읽어
별도 LLM(judge)에게 "이 분류 결과가 원문에 충실한가?"를 물어 채점한다.

사용 예:
    python3 scripts/judge_channel_compound.py --start 2026-04-27 --end 2026-05-04 \
        --output reports/judge_compound_2026-04-27.json
    python3 scripts/judge_channel_compound.py --start 2026-04-27 --end 2026-05-04 \
        --limit 50 --output reports/judge_compound_sample.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import boto3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bigquery.client import BigQueryClient


JUDGE_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

JUDGE_SYSTEM_PROMPT = """당신은 채널톡 CS 분류 결과를 검수하는 평가자입니다.

원문 텍스트와 분류 결과(topic, subtag, summary, tags, is_compound, compound_reason)를 받습니다.

다음을 평가하여 JSON으로만 답하세요. 다른 말은 절대 추가하지 마세요.

{
  "summary_ok": true|false,        // summary가 원문 핵심을 정확히 반영하는가
  "tags_ok": true|false,           // tags 4개 이내가 원문과 관련 있는가
  "compound_ok": true|false,       // is_compound=true이면 compound_reason이 실제로 두 가지 이상의 별개 이슈를 지칭하는가. is_compound=false이면 단일 이슈가 맞는가
  "topic_ok": true|false,          // topic이 원문 의도와 부합하는가
  "subtag_ok": true|false,         // subtag가 원문 핵심 요청과 부합하는가
  "issues": ["...", "..."]         // 위 항목 중 false인 이유 (한국어, 항목당 한 줄)
}

엄격한 기준:
- summary가 원문에 없는 정보를 추가하면 false (환각)
- summary가 핵심을 빠뜨리면 false
- is_compound=true인데 compound_reason이 단일 이슈를 부풀려 표현했으면 false
- tags가 원문과 무관한 일반 단어면 false
"""


def parse_json(raw: str) -> Dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(raw[start:end])
    return json.loads(raw)


class Judge:
    def __init__(self, region: str = "us-west-2", model_id: str = JUDGE_MODEL_ID):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def judge(self, row: Dict[str, Any]) -> Dict[str, Any]:
        user_msg = json.dumps({
            "원문": (row.get("text") or "")[:1000],
            "topic": row.get("topic"),
            "subtag": row.get("subtag"),
            "summary": row.get("summary"),
            "tags": row.get("tags") or [],
            "is_compound": bool(row.get("is_compound")),
            "compound_reason": row.get("compound_reason"),
        }, ensure_ascii=False)

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 400,
                "system": JUDGE_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            }),
        )
        result = json.loads(response["body"].read())
        raw = result["content"][0]["text"].strip()
        try:
            return parse_json(raw)
        except (ValueError, json.JSONDecodeError):
            return {
                "summary_ok": None, "tags_ok": None, "compound_ok": None,
                "topic_ok": None, "subtag_ok": None,
                "issues": [f"JUDGE_PARSE_ERROR: {raw[:200]}"],
            }


def fetch_rows(bq: BigQueryClient, start: str, end: str, limit: int) -> List[Dict[str, Any]]:
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT chat_id, text, topic, subtag, summary, tags,
           is_compound, compound_reason, pipeline_date
    FROM `{bq.project_id}.voc_labelled.channel_talk`
    WHERE pipeline_date >= DATE "{start}"
      AND pipeline_date < DATE "{end}"
      AND has_free_text = TRUE
      AND summary IS NOT NULL
      AND summary != ""
    ORDER BY pipeline_date DESC, chat_id
    {limit_clause}
    """
    return bq.execute_query(query)


def main() -> None:
    parser = argparse.ArgumentParser(description="채널톡 compound 결과 LLM judge 검수")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--output", required=True, help="결과 JSON 경로")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bq = BigQueryClient()
    rows = fetch_rows(bq, args.start, args.end, args.limit)
    logging.info("판정 대상: %s rows", len(rows))

    judge = Judge(region=args.region)
    judgments = []
    flagged = 0
    for idx, row in enumerate(rows, start=1):
        result = judge.judge(row)
        any_false = any(result.get(k) is False for k in ("summary_ok", "tags_ok", "compound_ok", "topic_ok", "subtag_ok"))
        if any_false:
            flagged += 1
        judgments.append({
            "chat_id": row["chat_id"],
            "pipeline_date": row["pipeline_date"].isoformat() if hasattr(row["pipeline_date"], "isoformat") else str(row["pipeline_date"]),
            "row": {
                "text": (row.get("text") or "")[:300],
                "topic": row.get("topic"),
                "subtag": row.get("subtag"),
                "summary": row.get("summary"),
                "tags": list(row.get("tags") or []),
                "is_compound": bool(row.get("is_compound")),
                "compound_reason": row.get("compound_reason"),
            },
            "judgment": result,
        })
        if idx % 50 == 0:
            logging.info("%s/%s 판정, flagged=%s", idx, len(rows), flagged)

    summary = {
        "total": len(judgments),
        "flagged": flagged,
        "summary_ok_pct": sum(1 for j in judgments if j["judgment"].get("summary_ok") is True) / max(len(judgments), 1) * 100,
        "tags_ok_pct": sum(1 for j in judgments if j["judgment"].get("tags_ok") is True) / max(len(judgments), 1) * 100,
        "compound_ok_pct": sum(1 for j in judgments if j["judgment"].get("compound_ok") is True) / max(len(judgments), 1) * 100,
        "topic_ok_pct": sum(1 for j in judgments if j["judgment"].get("topic_ok") is True) / max(len(judgments), 1) * 100,
        "subtag_ok_pct": sum(1 for j in judgments if j["judgment"].get("subtag_ok") is True) / max(len(judgments), 1) * 100,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "judgments": judgments}, f, ensure_ascii=False, indent=2)

    logging.info("저장: %s", out_path)
    logging.info("요약: %s", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
