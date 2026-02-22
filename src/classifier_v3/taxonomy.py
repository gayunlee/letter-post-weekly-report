"""4축 분류 체계 정의

축1 (주제/Topic): 무엇에 대한 글인가 — 파인튜닝 KcBERT (기존 v2)
축2 (감성/Sentiment): 어떤 톤인가 — 파인튜닝 KcBERT (기존 v2)
축3 (의도/Intent): 어떤 목적의 글인가 — LLM(Haiku), 이후 파인튜닝 전환
축4 (긴급도/Urgency): 얼마나 빠른 대응이 필요한가 — 규칙 기반
"""
from typing import Dict, List, Tuple

# ── 축1: 주제 (Topic) — 기존 v2와 동일 ──────────────────────────
TOPIC_CATEGORIES: Dict[str, str] = {
    "콘텐츠 반응": "마스터의 콘텐츠(강의, 리포트, 방송)에 대한 직접적 반응",
    "투자 이야기": "투자 전략, 종목, 포트폴리오, 시장 분석 관련 글",
    "서비스 이슈": "플랫폼/앱 기능, 결제, 배송, 구독 등 서비스 운영 관련",
    "커뮤니티 소통": "인사, 안부, 축하, 일상 공유, 커뮤니티 교류",
}

TOPICS: List[str] = list(TOPIC_CATEGORIES.keys())

# ── 축2: 감성 (Sentiment) — 기존 v2와 동일 ──────────────────────
SENTIMENT_CATEGORIES: Dict[str, str] = {
    "긍정": "감사, 만족, 기쁨, 칭찬, 격려",
    "부정": "불만, 실망, 걱정, 비판, 답답함",
    "중립": "질문, 정보 전달, 사실 기술, 요청",
}

SENTIMENTS: List[str] = list(SENTIMENT_CATEGORIES.keys())

# ── 축3: 의도 (Intent) — 신규 ───────────────────────────────────
INTENT_CATEGORIES: Dict[str, str] = {
    "질문/요청": "답변이나 조치를 기대하는 글 (CS팀, 콘텐츠팀)",
    "피드백/의견": "경험·감정 공유, 답변 불필요 (사업팀, 경영진)",
    "제보/건의": "문제 신고, 기능 요청, 정책 제안 (개발팀, 운영팀)",
    "정보공유": "뉴스, 분석, 경험 등 정보 전달 목적 (콘텐츠팀)",
}

INTENTS: List[str] = list(INTENT_CATEGORIES.keys())

# ── 축4: 긴급도 (Urgency) — 신규, 규칙 기반 ────────────────────
URGENCY_LEVELS: Dict[str, str] = {
    "긴급": "서비스이슈 + 부정 + 키워드(결제오류, 접속불가, 사칭 등)",
    "높음": "서비스이슈 + (질문/요청 or 제보/건의)",
    "보통": "부정 감성 or 질문/요청",
    "낮음": "기본값 (인사, 잡담 등)",
}

URGENCIES: List[str] = list(URGENCY_LEVELS.keys())


# ── 분류 결과 타입 정의 ──────────────────────────────────────────
def make_classification(
    topic: str,
    topic_confidence: float,
    sentiment: str,
    sentiment_confidence: float,
    intent: str = "",
    intent_confidence: float = 0.0,
    urgency: str = "낮음",
    urgency_method: str = "rule_based",
    department_route: List[str] = None,
    needs_review: bool = False,
    method: str = "four_axis_v3",
) -> Dict:
    """4축 분류 결과 딕셔너리 생성"""
    return {
        "topic": topic,
        "topic_confidence": topic_confidence,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "intent": intent,
        "intent_confidence": intent_confidence,
        "urgency": urgency,
        "urgency_method": urgency_method,
        "department_route": department_route or [],
        "needs_review": needs_review,
        "method": method,
    }
