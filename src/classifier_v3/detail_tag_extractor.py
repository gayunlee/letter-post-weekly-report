"""Detail Tag + Intent 추출 모듈 (v3 — 4분류 체계)

v2와 동일 구조이되, Topic 체계가 4분류로 변경됨.
- 운영 피드백 / 서비스 피드백 / 콘텐츠·투자 / 기타
- "콘텐츠·투자"는 콘텐츠 반응 + 투자 담론 태그를 합산하여 사용
- 카테고리 태그 28종이 4분류 Topic에 맞춰 배치
"""
import json
import time
from typing import List, Dict, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic
from dotenv import load_dotenv

from src.classifier_v3.taxonomy import CATEGORY_TAGS, ALL_VALID_TAGS

load_dotenv()


SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC(Voice of Customer) 데이터 분석가입니다.
사용자가 보낸 편지글이나 게시글을 읽고, 태그 추출과 의도 분류를 수행합니다.

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

## 태그 추출 규칙

### 1. 카테고리 태그 (category_tags)
아래 목록에서 1~2개를 선택하세요. 반드시 목록에 있는 것만 선택하세요.
{category_list}

### 2. 자유 태그 (free_tags)
2~3개의 구체적 명사구를 추출하세요 (2~5단어).
다른 팀이 검색할 때 유용한 구체적 내용이어야 합니다.
예: "게시판 폐쇄 사전 공지 부족", "ARKK 포트폴리오 비중 논란"

### 3. 한 줄 요약 (summary)
15자~40자 내외로 핵심 내용을 요약하세요.

### 4. 의도 분류 (intent)
아래 4가지 중 하나를 선택하세요:
- "질문/요청": 답변이나 조치를 기대하는 글 (질문, 도움 요청, 문의)
- "피드백/의견": 경험·감정을 공유하는 글, 답변 불필요 (후기, 감상, 칭찬, 불만 토로)
- "제보/건의": 문제 신고, 기능 요청, 정책 제안 (버그 제보, 개선 요청)
- "정보공유": 뉴스, 분석, 투자 경험 등 정보 전달 목적의 글

intent_confidence: 0.0~1.0 사이의 확신도를 함께 제시하세요.

## 응답 형식
반드시 아래 JSON만 출력하세요. 다른 텍스트 없이:
{"category_tags": ["태그1"], "free_tags": ["태그1", "태그2"], "summary": "한 줄 요약", "intent": "피드백/의견", "intent_confidence": 0.85}"""

MODEL_COSTS = {
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
}

VALID_INTENTS = {"질문/요청", "피드백/의견", "제보/건의", "정보공유"}


class DetailTagExtractorV3:
    """v3 분류 체계용 세부 태그 추출기"""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_workers: int = 5,
        max_tokens: int = 300,
    ):
        self.client = Anthropic()
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        self._parse_failures = 0
        self._invalid_tag_count = 0

    def _build_system_prompt(self, topic: str) -> str:
        if topic == "콘텐츠·투자":
            tags = CATEGORY_TAGS.get("콘텐츠 반응", []) + CATEGORY_TAGS.get("투자 담론", [])
        else:
            tags = CATEGORY_TAGS.get(topic, [])
        tag_list = "\n".join(f"- {t}" for t in tags)
        return SYSTEM_PROMPT.replace("{category_list}", tag_list)

    def extract_tags(self, text: str, topic: str, sentiment: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 10:
            return {
                "category_tags": [], "free_tags": [], "summary": "",
                "intent": "피드백/의견", "intent_confidence": 0.0, "parse_ok": True,
            }

        sys_prompt = self._build_system_prompt(topic)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=sys_prompt,
                messages=[{
                    "role": "user",
                    "content": f"[감성: {sentiment}]\n\n{text[:500]}"
                }],
            )

            usage = response.usage
            self._total_input_tokens += usage.input_tokens
            self._total_output_tokens += usage.output_tokens
            self._total_calls += 1

            raw = response.content[0].text
            return self._parse_response(raw, topic)

        except Exception as e:
            self._total_calls += 1
            self._parse_failures += 1
            return {
                "category_tags": [], "free_tags": [],
                "summary": f"API 오류: {str(e)[:50]}",
                "intent": "피드백/의견", "intent_confidence": 0.0, "parse_ok": False,
            }

    def _parse_response(self, raw: str, topic: str) -> Dict[str, Any]:
        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            if text.startswith("{{") and text.endswith("}}"):
                text = text[1:-1]
            result = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            self._parse_failures += 1
            return {
                "category_tags": [], "free_tags": [], "summary": "파싱 실패",
                "intent": "피드백/의견", "intent_confidence": 0.0,
                "parse_ok": False, "raw": raw,
            }

        cat_tags = result.get("category_tags", [])
        free_tags = result.get("free_tags", [])
        summary = result.get("summary", "")

        intent = result.get("intent", "피드백/의견")
        if intent not in VALID_INTENTS:
            intent = "피드백/의견"
        intent_confidence = result.get("intent_confidence", 0.5)
        if not isinstance(intent_confidence, (int, float)):
            intent_confidence = 0.5
        intent_confidence = max(0.0, min(1.0, float(intent_confidence)))

        if topic == "콘텐츠·투자":
            valid_for_topic = set(
                CATEGORY_TAGS.get("콘텐츠 반응", []) + CATEGORY_TAGS.get("투자 담론", [])
            )
        else:
            valid_for_topic = set(CATEGORY_TAGS.get(topic, []))
        validated_tags = []
        for tag in cat_tags:
            if tag in valid_for_topic:
                validated_tags.append(tag)
            else:
                self._invalid_tag_count += 1

        return {
            "category_tags": validated_tags,
            "free_tags": free_tags[:3],
            "summary": summary[:60],
            "intent": intent,
            "intent_confidence": intent_confidence,
            "parse_ok": True,
        }

    def extract_tags_batch(
        self,
        items: List[Dict[str, Any]],
        content_field: str = "message",
        batch_size: int = 50,
    ) -> List[Dict[str, Any]]:
        results = []
        total = len(items)
        start_time = time.time()

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = items[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {}
                for i, item in enumerate(batch):
                    cls = item.get("classification", {})
                    topic = cls.get("topic", "기타")
                    sentiment = cls.get("sentiment", "중립")
                    text = item.get(content_field, "")
                    if not text:
                        text = item.get("textBody") or item.get("body", "")

                    future = executor.submit(self.extract_tags, text, topic, sentiment)
                    future_map[future] = i

                batch_results = [None] * len(batch)
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        tag_result = future.result()
                    except Exception:
                        tag_result = {
                            "category_tags": [], "free_tags": [],
                            "summary": "", "intent": "피드백/의견",
                            "intent_confidence": 0.0, "parse_ok": False,
                        }
                    batch_results[idx] = tag_result

            for i, item in enumerate(batch):
                enriched = item.copy()
                detail = batch_results[i]
                enriched["detail_tags"] = {
                    "category_tags": detail["category_tags"],
                    "free_tags": detail["free_tags"],
                    "summary": detail["summary"],
                }
                cls = enriched.get("classification", {})
                if "intent" not in cls or not cls.get("intent"):
                    cls["intent"] = detail.get("intent", "피드백/의견")
                    cls["intent_confidence"] = detail.get("intent_confidence", 0.0)
                    enriched["classification"] = cls
                results.append(enriched)

            elapsed = time.time() - start_time
            processed = batch_end
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  detail_tags(v3): {processed}/{total} 완료 ({rate:.1f}건/초)")

        return results

    def get_cost_report(self) -> Dict[str, Any]:
        cost_info = MODEL_COSTS.get(self.model, {"input": 0, "output": 0})
        input_cost = self._total_input_tokens / 1000 * cost_info["input"]
        output_cost = self._total_output_tokens / 1000 * cost_info["output"]
        total_cost = input_cost + output_cost

        return {
            "model": self.model,
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "parse_failures": self._parse_failures,
            "invalid_tags": self._invalid_tag_count,
            "parse_success_rate": (
                round((1 - self._parse_failures / self._total_calls) * 100, 1)
                if self._total_calls > 0 else 0
            ),
            "estimated_cost_usd": round(total_cost, 4),
            "cost_per_item_usd": (
                round(total_cost / self._total_calls, 6)
                if self._total_calls > 0 else 0
            ),
        }

    def reset_counters(self):
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        self._parse_failures = 0
        self._invalid_tag_count = 0


def aggregate_category_tags(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """카테고리 태그 집계 분석"""
    overall = Counter()
    by_topic = {}
    by_sentiment = {}
    tagged_count = 0

    for item in items:
        detail = item.get("detail_tags", {})
        cls = item.get("classification", {})
        topic = cls.get("topic", "미분류")
        sentiment = cls.get("sentiment", "미분류")

        cat_tags = detail.get("category_tags", [])
        if cat_tags:
            tagged_count += 1

        for tag in cat_tags:
            overall[tag] += 1
            by_topic.setdefault(topic, Counter())[tag] += 1
            by_sentiment.setdefault(sentiment, Counter())[tag] += 1

    total = len(items)
    return {
        "total_items": total,
        "tagged_items": tagged_count,
        "tag_coverage": round(tagged_count / total * 100, 1) if total > 0 else 0,
        "overall": overall,
        "by_topic": by_topic,
        "by_sentiment": by_sentiment,
    }
