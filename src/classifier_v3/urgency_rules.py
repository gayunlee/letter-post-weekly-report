"""긴급도(Urgency) 규칙 엔진

Topic + Sentiment + Intent + 키워드 조합으로 긴급도를 판별합니다.
모델 학습 없이 규칙 기반으로 동작합니다.

규칙 우선순위:
1. 긴급: 서비스이슈 + 부정 + 긴급 키워드
2. 높음: 서비스이슈 + (질문/요청 or 제보/건의)
3. 보통: 부정 감성 or 질문/요청
4. 낮음: 기본값
"""
import re
from typing import Dict, Any


# ── 긴급 키워드 목록 ────────────────────────────────────────────
# 운영팀과 협의하여 지속 업데이트
URGENT_KEYWORDS = re.compile(
    r"결제\s*오류|결제\s*실패|결제\s*안[돼됨되]|이중\s*결제|"
    r"접속\s*불가|접속\s*안[돼됨되]|로그인\s*안[돼됨되]|로그인\s*실패|"
    r"사칭|해킹|개인정보\s*유출|보안|피싱|"
    r"서비스\s*중단|서버\s*[다오]운|먹통|장애|"
    r"환불\s*불가|환불\s*안[돼됨되]|"
    r"데이터\s*[삭소]실|계정\s*[삭소]제|계정\s*정지",
    re.IGNORECASE
)


def classify_urgency(
    topic: str,
    sentiment: str,
    intent: str,
    text: str = "",
) -> Dict[str, Any]:
    """긴급도 분류

    Args:
        topic: 주제 분류 결과
        sentiment: 감성 분류 결과
        intent: 의도 분류 결과
        text: 원문 텍스트 (키워드 매칭용)

    Returns:
        {"urgency": str, "urgency_method": "rule_based", "matched_keywords": list}
    """
    matched_keywords = []
    if text:
        matches = URGENT_KEYWORDS.finditer(text)
        matched_keywords = list(set(m.group() for m in matches))

    is_service = topic == "서비스 이슈"
    is_negative = sentiment == "부정"
    is_question_or_request = intent == "질문/요청"
    is_report = intent == "제보/건의"

    # 규칙 1: 긴급 — 서비스이슈 + 부정 + 긴급 키워드
    if is_service and is_negative and matched_keywords:
        return {
            "urgency": "긴급",
            "urgency_method": "rule_based",
            "matched_keywords": matched_keywords,
        }

    # 규칙 2: 높음 — 서비스이슈 + (질문/요청 or 제보/건의)
    if is_service and (is_question_or_request or is_report):
        return {
            "urgency": "높음",
            "urgency_method": "rule_based",
            "matched_keywords": matched_keywords,
        }

    # 규칙 3: 보통 — 부정 감성 or 질문/요청
    if is_negative or is_question_or_request:
        return {
            "urgency": "보통",
            "urgency_method": "rule_based",
            "matched_keywords": matched_keywords,
        }

    # 규칙 4: 낮음 — 기본값
    return {
        "urgency": "낮음",
        "urgency_method": "rule_based",
        "matched_keywords": [],
    }
