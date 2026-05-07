"""BedrockTwoDepthClassifier — KcELECTRA + Bedrock Sonnet 서브태그

TwoDepthClassifier의 LLM 호출을 Bedrock으로 교체.
KcELECTRA 1차 분류는 동일, LLM fallback과 서브태그만 Bedrock 사용.
"""
import json
import time
import logging

import boto3

from .two_depth_classifier import TwoDepthClassifier, strip_workflow_buttons, ID_TO_CATEGORY
from .subtag_prompt import SUBTAG_SYSTEM_PROMPT, SUBTAGS

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class BedrockTwoDepthClassifier(TwoDepthClassifier):
    """KcELECTRA + Bedrock Sonnet 2-depth 분류기"""

    def __init__(
        self,
        model_dir="models/channel_3class/kcelectra-base-v2022-v9-boost/final_model",
        bedrock_region="us-west-2",
        bedrock_model_id=None,
        confidence_threshold=0.9,
        subtag_all=True,
    ):
        # KcELECTRA 로드 (부모 클래스)
        super().__init__(
            model_dir=model_dir,
            llm_model="unused",  # Bedrock 사용하므로 무시
            confidence_threshold=confidence_threshold,
            subtag_all=subtag_all,
        )
        # Bedrock 클라이언트
        self.bedrock = boto3.client("bedrock-runtime", region_name=bedrock_region)
        self.bedrock_model_id = bedrock_model_id or BEDROCK_MODEL_ID
        self._llm_client = None  # 부모의 Anthropic 클라이언트 비활성화

    def _bedrock_call(self, system, user_content, max_tokens=50):
        """Bedrock API 호출 래퍼"""
        for attempt in range(3):
            try:
                resp = self.bedrock.invoke_model(
                    modelId=self.bedrock_model_id,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "system": system,
                        "messages": [{"role": "user", "content": user_content}],
                    }),
                )
                result = json.loads(resp["body"].read())
                return result["content"][0]["text"].strip()
            except Exception as e:
                if "Throttl" in str(e) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                logger.warning(f"Bedrock 호출 실패: {e}")
                return None

    def classify_llm_topic(self, text):
        """LLM fallback: 1차 분류 재판단 (Bedrock)"""
        prompt = f"""채널톡 CS 문의를 3개 카테고리 중 하나로 분류하세요.

카테고리:
- 결제·구독: 결제, 환불, 구독, 해지, 카드 변경 등 돈 관련
- 콘텐츠·수강: 강의 접근, 수강 방법, 오프라인 참석, 자료 요청
- 기술·오류: 로그인 안됨, 앱 오류, 콘텐츠 접근 차단, 기능 건의

분류 기준: "발생한 현상 / 고객이 요청한 사항"
JSON만: {{"topic": "결제·구독"}}

문의: {text[:500]}"""

        raw = self._bedrock_call("", prompt)
        if not raw:
            return {"topic": "콘텐츠·수강"}

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw[start:end])
            else:
                result = json.loads(raw)
            topic = result.get("topic", "콘텐츠·수강")
            if topic not in ID_TO_CATEGORY.values():
                topic = "콘텐츠·수강"
            return {"topic": topic}
        except Exception:
            return {"topic": "콘텐츠·수강"}

    def classify_subtag(self, text, topic):
        """2차: 서브태그 부여 (Bedrock)"""
        if topic not in SUBTAG_SYSTEM_PROMPT:
            return "기타"

        prompt_text = strip_workflow_buttons(text)[:500]
        raw = self._bedrock_call(SUBTAG_SYSTEM_PROMPT[topic], prompt_text)

        if not raw:
            return "기타"

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw[start:end])
            else:
                result = json.loads(raw)
            subtag = result.get("subtag", "기타")
            valid = SUBTAGS.get(topic, [])
            if subtag not in valid:
                subtag = "기타"
            return subtag
        except Exception:
            return "기타"
