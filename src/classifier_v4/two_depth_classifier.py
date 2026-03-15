"""2-depth 채널톡 분류기

1차: KcELECTRA 3분류 (로컬, ~10ms/건)
2차: LLM 서브태그 (conf < threshold 또는 전건)

사용법:
    classifier = TwoDepthClassifier(
        model_dir="models/channel_3class/kcelectra-base-v2022-v9-boost/final_model",
        llm_model="claude-haiku-4-5-20251001",
        confidence_threshold=0.9,
    )
    result = classifier.classify("환불해주세요")
    # {"topic": "결제·구독", "confidence": 0.98, "subtag": "환불", "source": "kcelectra"}
"""
import json
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .subtag_prompt import SUBTAG_SYSTEM_PROMPT, SUBTAGS

WORKFLOW_NOISE = [
    "그 외 기타 문의(오류/구독해지/환불)", "그 외 기타 문의",
    "💬1:1 상담 문의하기", "💬 1:1 고객센터 문의하기",
    "💬상담매니저에게 직접 문의", "어스 구독신청/결제하기",
    "어스 이용방법", "사이트 및 동영상 오류", "수강 및 상품문의",
    "라이브 콘텐츠 참여 방법", "수강방법", "↩ 이전으로",
    "✅ 1:1 문의하기", "구독 상품변경/결제정보 확인",
    "결제실패 후 카드변경 방법",
    "💬 구독 결제/변경/정보 확인 직접 문의하기",
    "구독상품 변경", "구독 결제/변경/정보 직접 문의하기",
    "🏠처음으로", "🏠 처음으로",
]

ID_TO_CATEGORY = {0: "결제·구독", 1: "콘텐츠·수강", 2: "기술·오류"}


def strip_workflow_buttons(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in WORKFLOW_NOISE:
            continue
        if stripped.startswith("👆🏻"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


class TwoDepthClassifier:
    """2-depth 채널톡 CS 분류기"""

    def __init__(
        self,
        model_dir: str = "models/channel_3class/kcelectra-base-v2022-v9-boost/final_model",
        llm_model: str = "claude-haiku-4-5-20251001",
        confidence_threshold: float = 0.9,
        subtag_all: bool = True,
    ):
        """
        Args:
            model_dir: KcELECTRA 모델 경로
            llm_model: LLM fallback 모델 (서브태그용)
            confidence_threshold: 이 값 미만이면 LLM이 1차 분류도 재판단
            subtag_all: True면 모든 건에 서브태그 부여, False면 fallback 건만
        """
        self.confidence_threshold = confidence_threshold
        self.subtag_all = subtag_all
        self.llm_model = llm_model

        # KcELECTRA 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.model = self.model.to(self.device)

        # LLM 클라이언트 (lazy init)
        self._llm_client = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            from anthropic import Anthropic
            from dotenv import load_dotenv
            load_dotenv()
            self._llm_client = Anthropic()
        return self._llm_client

    def classify_kcelectra(self, text: str) -> Dict[str, Any]:
        """1차: KcELECTRA 분류"""
        cleaned = strip_workflow_buttons(text)
        if len(cleaned.strip()) < 5:
            cleaned = text

        inputs = self.tokenizer(
            cleaned[:512], truncation=True, padding="max_length",
            max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
            pred_id = probs.argmax().item()
            confidence = probs[pred_id].item()

        return {
            "topic": ID_TO_CATEGORY[pred_id],
            "confidence": round(confidence, 4),
        }

    def classify_llm_topic(self, text: str) -> Dict[str, Any]:
        """LLM fallback: 1차 분류 재판단"""
        prompt = f"""채널톡 CS 문의를 3개 카테고리 중 하나로 분류하세요.

카테고리:
- 결제·구독: 결제, 환불, 구독, 해지, 카드 변경 등 돈 관련
- 콘텐츠·수강: 강의 접근, 수강 방법, 오프라인 참석, 자료 요청
- 기술·오류: 로그인 안됨, 앱 오류, 콘텐츠 접근 차단, 기능 건의

분류 기준: "발생한 현상 / 고객이 요청한 사항"

JSON만: {{"topic": "결제·구독"}}

문의: {text[:500]}"""

        try:
            resp = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
                timeout=15.0,
            )
            raw = resp.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            topic = result.get("topic", "콘텐츠·수강")
            if topic not in ID_TO_CATEGORY.values():
                topic = "콘텐츠·수강"
            return {"topic": topic}
        except Exception:
            return {"topic": "콘텐츠·수강"}

    def classify_subtag(self, text: str, topic: str) -> str:
        """2차: 서브태그 부여"""
        if topic not in SUBTAG_SYSTEM_PROMPT:
            return "기타"

        prompt_text = strip_workflow_buttons(text)[:500]

        try:
            resp = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=50,
                system=SUBTAG_SYSTEM_PROMPT[topic],
                messages=[{"role": "user", "content": prompt_text}],
                timeout=15.0,
            )
            raw = resp.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            subtag = result.get("subtag", "기타")
            valid = SUBTAGS.get(topic, [])
            if subtag not in valid:
                subtag = "기타"
            return subtag
        except Exception:
            return "기타"

    def classify(self, text: str) -> Dict[str, Any]:
        """전체 2-depth 분류"""
        # 1차: KcELECTRA
        result = self.classify_kcelectra(text)
        result["source"] = "kcelectra"

        # LLM fallback (confidence 낮으면)
        if result["confidence"] < self.confidence_threshold:
            llm_result = self.classify_llm_topic(text)
            result["topic"] = llm_result["topic"]
            result["source"] = "llm_fallback"

        # 2차: 서브태그
        if self.subtag_all or result["source"] == "llm_fallback":
            result["subtag"] = self.classify_subtag(text, result["topic"])

        return result

    def classify_batch(self, items: List[Dict], text_field: str = "text") -> List[Dict]:
        """배치 분류"""
        results = []
        for item in items:
            text = item.get(text_field, "")
            classification = self.classify(text)
            item["classification"] = classification
            results.append(item)
        return results
