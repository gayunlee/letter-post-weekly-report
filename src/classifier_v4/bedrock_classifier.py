"""BedrockClassifier — Bedrock Sonnet 3.7로 편지글/게시글 3분류

unified_classifier.py의 Bedrock 버전.
Sonnet 전용 프롬프트(sonnet_prompt.py) 사용.
"""
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

from .sonnet_prompt import SONNET_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
V4_TOPICS = ["대응 필요", "콘텐츠·투자", "노이즈"]
V4_SENTIMENTS = ["긍정", "부정", "중립"]


def _parse_json_response(raw: str) -> dict:
    """LLM 응답에서 JSON 추출."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(raw[start:end])
    return json.loads(raw)


class BedrockClassifier:
    """Bedrock Sonnet 3.7로 편지글/게시글 3분류 + tags + summary"""

    def __init__(self, region="us-west-2", model_id=None, max_workers=5):
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id or MODEL_ID
        self.max_workers = max_workers
        self._input_tokens = 0
        self._output_tokens = 0
        self._errors = 0
        self._total = 0

    def classify_single(self, text: str) -> dict:
        """단일 건 분류 → {topic, sentiment, tags, summary, confidence}"""
        if not text or len(text.strip()) < 2:
            return {
                "topic": "노이즈", "sentiment": "중립",
                "tags": [], "summary": "", "confidence": 1.0,
            }

        for attempt in range(3):
            try:
                resp = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 350,
                        "system": SONNET_SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": text[:500]}],
                    }),
                )
                result = json.loads(resp["body"].read())
                self._input_tokens += result.get("usage", {}).get("input_tokens", 0)
                self._output_tokens += result.get("usage", {}).get("output_tokens", 0)

                parsed = _parse_json_response(result["content"][0]["text"])

                topic = parsed.get("topic", "콘텐츠·투자")
                if topic not in V4_TOPICS:
                    topic = "콘텐츠·투자"

                sentiment = parsed.get("sentiment", "중립")
                if sentiment not in V4_SENTIMENTS:
                    sentiment = "중립"

                tags = parsed.get("tags", [])
                if not isinstance(tags, list):
                    tags = []

                return {
                    "topic": topic,
                    "sentiment": sentiment,
                    "tags": tags[:4],
                    "summary": str(parsed.get("summary", ""))[:200],
                    "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
                }

            except Exception as e:
                if "Throttl" in str(e) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                self._errors += 1
                logger.warning(f"분류 실패 (attempt {attempt+1}): {e}")
                return {
                    "topic": "콘텐츠·투자", "sentiment": "중립",
                    "tags": [], "summary": "", "confidence": 0.0,
                }

    def classify_batch(self, items, content_field="message"):
        """ThreadPoolExecutor 병렬 분류."""
        self._total = len(items)
        start_time = time.time()
        done_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {}
            for i, item in enumerate(items):
                text = item.get(content_field, "") if isinstance(item, dict) else str(item)
                future = executor.submit(self.classify_single, text)
                future_map[future] = i

            for future in as_completed(future_map):
                idx = future_map[future]
                classification = future.result()
                if isinstance(items[idx], dict):
                    items[idx]["classification"] = classification
                done_count += 1
                if done_count % 100 == 0 or done_count == len(items):
                    elapsed = time.time() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    logger.info(f"  {done_count}/{len(items)} ({rate:.1f}건/초)")

        return items

    def get_cost_report(self) -> dict:
        """비용 리포트"""
        input_cost = self._input_tokens * 3.0 / 1_000_000
        output_cost = self._output_tokens * 15.0 / 1_000_000
        return {
            "model": self.model_id,
            "total_items": self._total,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cost_usd": round(input_cost + output_cost, 4),
            "errors": self._errors,
        }
