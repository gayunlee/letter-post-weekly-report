"""소수 카테고리 오분류 보정 모듈

KcBERT 4분류에서 소수 클래스(운영/서비스 피드백)가 다수 클래스(콘텐츠·투자)로
빨려가는 데이터 불균형 문제를 LLM 재검증으로 보정.

전략: KcBERT가 콘텐츠·투자로 분류했지만 top2가 소수 카테고리인 건을
Haiku에게 binary 질문으로 최종 판단을 맡김.

NOTE: KcBERT 4분류 모델은 극도로 overconfident (margin 0.93+)하여
margin 기반 선별이 무의미. top2 기반으로만 후보를 선별.
"""
import json
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


LLM_SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC 분류 검증기입니다.
사용자가 보낸 글을 읽고, 이 글의 주제가 무엇인지 판단합니다.

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

## 분류 기준
아래 두 가지 중 하나를 선택하세요:
1. "콘텐츠·투자" — 마스터/콘텐츠에 대한 반응(칭찬, 불만, 후기), 투자 담론(종목, 시장, 포트폴리오), 투자 질문
2. "{alternative}" — {alternative_desc}

## 판단 원칙
- 글의 **주된 목적**이 무엇인지 판단하세요
- 투자/콘텐츠 이야기 중에 부수적으로 서비스/운영 언급이 있으면 "콘텐츠·투자"
- 서비스/운영 이슈가 글의 핵심이면 "{alternative}"

반드시 아래 JSON만 출력하세요:
{{"topic": "선택한 주제", "confidence": 0.0~1.0, "reason": "한 줄 근거"}}"""

TOPIC_DESCRIPTIONS = {
    "운영 피드백": "운영팀이 사람 대 사람으로 처리할 요청·이슈 (세미나, 환불, 멤버십, 배송, 구독, 가격, 프로모션)",
    "서비스 피드백": "개발팀이 시스템을 수정해야 하는 기술적 이슈·요청 (앱 버그, 결제 오류, 로그인, 기능 요청, UX 개선)",
}

MODEL_COSTS = {
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
}


class TopicCorrector:
    """LLM 기반 topic 재검증기"""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_workers: int = 5):
        self.client = Anthropic()
        self.model = model
        self.max_workers = max_workers
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

    def _verify_single(self, text: str, alternative: str) -> Dict[str, Any]:
        """단일 항목 binary 검증"""
        desc = TOPIC_DESCRIPTIONS.get(alternative, "")
        system = LLM_SYSTEM_PROMPT.replace("{alternative}", alternative).replace(
            "{alternative_desc}", desc
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                system=system,
                messages=[{"role": "user", "content": text[:500]}],
            )

            usage = response.usage
            self._total_input_tokens += usage.input_tokens
            self._total_output_tokens += usage.output_tokens
            self._total_calls += 1

            raw = response.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
            return {
                "topic": result.get("topic", "콘텐츠·투자"),
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", ""),
            }
        except Exception:
            self._total_calls += 1
            return {"topic": "콘텐츠·투자", "confidence": 0.0, "reason": "API 오류"}

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
            "estimated_cost_usd": round(total_cost, 4),
        }


def correct_topics(
    items: List[Dict],
    content_field: str = "message",
    corrector: TopicCorrector = None,
) -> Dict[str, Any]:
    """LLM 재검증으로 소수 카테고리 오분류 보정

    대상: topic == "콘텐츠·투자" AND top2 ∈ {운영/서비스 피드백}
    → Haiku binary 질문으로 최종 판단

    Args:
        items: classification 포함된 항목 리스트
        content_field: 텍스트 필드명
        corrector: TopicCorrector 인스턴스 (없으면 생성)

    Returns:
        {"items": 보정된 리스트, "stats": 보정 통계}
    """
    total = len(items)
    start_time = time.time()

    if corrector is None:
        corrector = TopicCorrector()

    minority_topics = {"운영 피드백", "서비스 피드백"}

    # 검토 대상 선별: top2가 소수 카테고리인 콘텐츠·투자 건
    candidates = []
    for i, item in enumerate(items):
        cls = item.get("classification", {})
        if cls.get("topic") != "콘텐츠·투자":
            continue

        top2 = cls.get("topic_top2", "")

        if top2 in minority_topics:
            text = item.get(content_field, "")
            if not text:
                text = item.get("textBody") or item.get("body", "")
            if text:
                candidates.append((i, text, top2))

    if not candidates:
        elapsed = time.time() - start_time
        print(f"  보정 대상 없음 (top2 ∈ 소수 카테고리: 0건)")
        return {
            "items": items,
            "stats": {
                "total": total,
                "candidates": 0,
                "corrected": 0,
                "elapsed_sec": round(elapsed, 1),
                "llm_cost": corrector.get_cost_report(),
            },
        }

    print(f"  LLM 재검증 대상: {len(candidates)}건 (top2 ∈ 소수 카테고리)")

    # 병렬 LLM 검증
    with ThreadPoolExecutor(max_workers=corrector.max_workers) as executor:
        future_map = {}
        for idx, text, top2 in candidates:
            future = executor.submit(corrector._verify_single, text, top2)
            future_map[future] = (idx, top2)

        corrected = 0
        for future in as_completed(future_map):
            idx, top2 = future_map[future]
            try:
                result = future.result()
            except Exception:
                continue

            if result["topic"] != "콘텐츠·투자" and result["confidence"] >= 0.6:
                cls = items[idx]["classification"]
                cls["topic_before_correction"] = cls["topic"]
                cls["topic"] = result["topic"]
                cls["correction_method"] = "llm_binary"
                cls["correction_reason"] = result["reason"]
                cls["correction_confidence"] = result["confidence"]
                items[idx]["classification"] = cls
                corrected += 1

    elapsed = time.time() - start_time
    llm_cost = corrector.get_cost_report()

    print(f"  LLM 보정: {corrected}/{len(candidates)}건, "
          f"${llm_cost['estimated_cost_usd']:.4f}, {elapsed:.1f}초")

    return {
        "items": items,
        "stats": {
            "total": total,
            "candidates": len(candidates),
            "corrected": corrected,
            "elapsed_sec": round(elapsed, 1),
            "llm_cost": llm_cost,
        },
    }
