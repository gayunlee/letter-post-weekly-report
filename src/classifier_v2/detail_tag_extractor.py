"""Detail Tag 추출 모듈 (카테고리 태그 + 자유 태그 + 요약)

2축 분류(Topic × Sentiment) 이후, 각 건에 대해 세부 태그를 추출합니다.
- category_tags: 사전 정의된 목록에서 1~2개 선택
- free_tags: 구체적 명사구 2~3개 (검색용)
- summary: 15~40자 한 줄 요약

비용 최적화:
- 기본 모델: Claude Haiku (저비용, 충분한 품질)
- 배치 처리 + 병렬 호출로 처리 시간 최소화
"""
import json
import os
import time
from typing import List, Dict, Any, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Topic별 카테고리 태그 목록 ──────────────────────────────────────
CATEGORY_TAGS: Dict[str, List[str]] = {
    "서비스 이슈": [
        "결제/환불/구독",
        "앱/기능 오류",
        "게시판/커뮤니티 운영",
        "온보딩/접근성",
        "배송/일정",
        "가격/프로모션 정책",
        "콘텐츠 접근 문제",
        "기타 서비스",
    ],
    "투자 이야기": [
        "포트폴리오 전략/비중",
        "개별 종목 분석",
        "시장 전망/매크로",
        "수익/손실 공유",
        "매매 타이밍",
        "섹터/테마 분석",
        "투자 학습 질문",
        "기타 투자",
    ],
    "콘텐츠 반응": [
        "콘텐츠 품질/깊이",
        "마스터 소통/태도",
        "강의/수업 피드백",
        "리포트/브리핑 피드백",
        "콘텐츠 주제 요청",
        "기타 콘텐츠",
    ],
    "커뮤니티 소통": [
        "인사/안부/감사",
        "투자 경험 공유",
        "마스터 응원/격려",
        "커뮤니티 분위기",
        "일상 공유",
        "기타 소통",
    ],
}

# 전체 유효 태그 집합 (검증용)
ALL_VALID_TAGS = set()
for tags in CATEGORY_TAGS.values():
    ALL_VALID_TAGS.update(tags)

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC(Voice of Customer) 데이터 분석가입니다.
사용자가 보낸 편지글이나 게시글을 읽고, 2종류의 태그를 추출합니다.

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

## 응답 형식
반드시 아래 JSON만 출력하세요. 다른 텍스트 없이:
{"category_tags": ["태그1"], "free_tags": ["태그1", "태그2"], "summary": "한 줄 요약"}"""

# 모델별 비용 (1K tokens 기준, USD)
MODEL_COSTS = {
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
}


class DetailTagExtractor:
    """2축 분류 결과에 세부 태그를 부착하는 추출기"""

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

        # 비용 추적
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        self._parse_failures = 0
        self._invalid_tag_count = 0

    def _build_system_prompt(self, topic: str) -> str:
        """Topic에 맞는 카테고리 목록 포함 시스템 프롬프트"""
        tags = CATEGORY_TAGS.get(topic, [])
        tag_list = "\n".join(f"- {t}" for t in tags)
        return SYSTEM_PROMPT.replace("{category_list}", tag_list)

    def extract_tags(self, text: str, topic: str, sentiment: str) -> Dict[str, Any]:
        """단일 건 태그 추출"""
        if not text or len(text.strip()) < 10:
            return {
                "category_tags": [],
                "free_tags": [],
                "summary": "",
                "parse_ok": True,
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

            # 비용 추적
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
                "category_tags": [],
                "free_tags": [],
                "summary": f"API 오류: {str(e)[:50]}",
                "parse_ok": False,
            }

    def _parse_response(self, raw: str, topic: str) -> Dict[str, Any]:
        """LLM 응답 JSON 파싱 + 태그 검증"""
        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            # 이중 중괄호 처리 (프롬프트의 {{ }} 복사 방지)
            if text.startswith("{{") and text.endswith("}}"):
                text = text[1:-1]
            result = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            self._parse_failures += 1
            return {
                "category_tags": [],
                "free_tags": [],
                "summary": "파싱 실패",
                "parse_ok": False,
                "raw": raw,
            }

        cat_tags = result.get("category_tags", [])
        free_tags = result.get("free_tags", [])
        summary = result.get("summary", "")

        # 카테고리 태그 검증: 해당 topic의 유효 태그만 허용
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
            "parse_ok": True,
        }

    def extract_tags_batch(
        self,
        items: List[Dict[str, Any]],
        content_field: str = "message",
        batch_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """배치 태그 추출 (병렬 처리)

        Args:
            items: 2축 분류가 완료된 데이터 리스트
            content_field: 텍스트 필드명
            batch_size: 한 번에 병렬 처리할 건수

        Returns:
            각 item에 detail_tags 필드가 추가된 리스트
        """
        results = []
        total = len(items)
        start_time = time.time()

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = items[batch_start:batch_end]

            # 병렬 호출
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {}
                for i, item in enumerate(batch):
                    cls = item.get("classification", {})
                    topic = cls.get("topic", "커뮤니티 소통")
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
                            "summary": "", "parse_ok": False,
                        }
                    batch_results[idx] = tag_result

            # 결과 병합
            for i, item in enumerate(batch):
                enriched = item.copy()
                detail = batch_results[i]
                enriched["detail_tags"] = {
                    "category_tags": detail["category_tags"],
                    "free_tags": detail["free_tags"],
                    "summary": detail["summary"],
                }
                results.append(enriched)

            elapsed = time.time() - start_time
            processed = batch_end
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  detail_tags: {processed}/{total} 완료 ({rate:.1f}건/초)")

        return results

    def get_cost_report(self) -> Dict[str, Any]:
        """비용 추적 리포트"""
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
            "cost_per_item_usd": round(total_cost / self._total_calls, 6) if self._total_calls > 0 else 0,
        }

    def reset_counters(self):
        """비용 카운터 초기화"""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        self._parse_failures = 0
        self._invalid_tag_count = 0


def aggregate_category_tags(
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """카테고리 태그 집계 분석

    Returns:
        {
            "total_items": int,
            "tagged_items": int,  # category_tags가 1개 이상인 건
            "tag_coverage": float,  # tagged / total (%)
            "overall": Counter,  # 전체 태그 빈도
            "by_topic": {topic: Counter},  # Topic별 태그 빈도
            "by_sentiment": {sentiment: Counter},  # Sentiment별 태그 빈도
            "invalid_rate": float,  # 목록 외 태그 비율 (%)
        }
    """
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
