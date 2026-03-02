"""기능/대응 주체 + 분석 목적 기반 분류 체계 정의 (v3)

기존 v2의 소재 기반 Topic(콘텐츠 반응/투자 이야기/서비스 이슈/커뮤니티 소통)을
5분류 체계(운영 피드백/서비스 피드백/콘텐츠 반응/투자 담론/기타)로 전환.

핵심 질문:
1. "이 글에 누가 대응해야 하나?" → 운영 피드백 / 서비스 피드백
2. "글의 주된 대상이 마스터/콘텐츠인가, 시장/투자인가?" → 콘텐츠 반응 / 투자 담론
3. 위 어디에도 해당하지 않는 인사·잡담 → 기타
"""
from typing import Dict, List, Tuple


# ── 축1: 주제 (Topic) — 기능/대응 주체 + 분석 목적 기반 ───────────────
TOPIC_CATEGORIES: Dict[str, str] = {
    "운영 피드백": "운영팀/사업팀이 사람 대 사람으로 처리할 요청·이슈 (세미나, 환불, 멤버십, 배송)",
    "서비스 피드백": "개발팀이 시스템을 수정해야 하는 기술적 이슈·요청 (앱 버그, 결제 오류, 기능 요청)",
    "콘텐츠 반응": "마스터/콘텐츠가 주된 대상인 감상·반응·피드백 (강의 후기, 마스터 칭찬/불만, 콘텐츠 품질)",
    "투자 담론": "시장/종목/투자전략이 주된 대상인 의견·질문·공유 (투자 질문, 수익 공유, 시장 분석, 종목 토론)",
    "기타": "분석 가치가 낮은 인사·잡담·테스트",
}

TOPICS: List[str] = list(TOPIC_CATEGORIES.keys())

TOPIC_TO_ID: Dict[str, int] = {cat: i for i, cat in enumerate(TOPICS)}
ID_TO_TOPIC: Dict[int, str] = {i: cat for cat, i in TOPIC_TO_ID.items()}

# ── 축2: 감성 (Sentiment) — v2와 동일 ───────────────────────────────
SENTIMENT_CATEGORIES: Dict[str, str] = {
    "긍정": "감사, 만족, 기쁨, 칭찬, 격려",
    "부정": "불만, 실망, 걱정, 비판, 답답함",
    "중립": "질문, 정보 전달, 사실 기술, 요청",
}

SENTIMENTS: List[str] = list(SENTIMENT_CATEGORIES.keys())

# ── 축3: 의도 (Intent) — v2 detail_tags에서 추출, 동일 ──────────────
INTENT_CATEGORIES: Dict[str, str] = {
    "질문/요청": "답변이나 조치를 기대하는 글",
    "피드백/의견": "경험·감정 공유, 답변 불필요",
    "제보/건의": "문제 신고, 기능 요청, 정책 제안",
    "정보공유": "뉴스, 분석, 경험 등 정보 전달",
}

INTENTS: List[str] = list(INTENT_CATEGORIES.keys())

# ── 축4: 긴급도 (Urgency) — 규칙 기반, 동일 ────────────────────────
URGENCY_LEVELS: Dict[str, str] = {
    "긴급": "서비스 피드백 + 부정 + 키워드(결제오류, 접속불가, 사칭 등)",
    "높음": "서비스 피드백 + (질문/요청 or 제보/건의)",
    "보통": "부정 감성 or 질문/요청",
    "낮음": "기본값 (인사, 잡담 등)",
}

URGENCIES: List[str] = list(URGENCY_LEVELS.keys())


# ── v2 → v3 Topic 매핑 (근사치, 실제 재분류 필요) ────────────────────
V2_TO_V3_TOPIC: Dict[str, str] = {
    "콘텐츠 반응": "콘텐츠 반응",    # 61%가 콘텐츠 반응, 13% 투자 담론
    "투자 이야기": "투자 담론",      # 65%가 투자 담론, 17% 콘텐츠 반응
    "서비스 이슈": "운영 피드백",    # 43% 운영, 9% 서비스, 나머지 혼재
    "커뮤니티 소통": "기타",         # 39% 기타, 33% 콘텐츠 반응, 24% 투자 담론
}


# ── 카테고리 태그 (28종, 5분류 체계 배치) ─────────────────────────────
CATEGORY_TAGS: Dict[str, List[str]] = {
    "운영 피드백": [
        "결제/환불 정책",
        "구독/멤버십 관리",
        "세미나/이벤트",
        "배송/일정",
        "가격/프로모션 정책",
        "온보딩/접근성",
        "기타 운영",
    ],
    "서비스 피드백": [
        "앱/기능 오류",
        "결제 시스템 오류",
        "로그인/계정 문제",
        "콘텐츠 접근 문제",
        "보안/사칭 제보",
        "기능/UX 개선 요청",
        "기타 서비스",
    ],
    "콘텐츠 반응": [
        "콘텐츠 품질/깊이",
        "마스터 소통/태도",
        "강의/수업 반응",
        "리포트/브리핑 반응",
        "기타 콘텐츠",
    ],
    "투자 담론": [
        "포트폴리오/종목 전략",
        "시장 전망/매크로",
        "수익/손실 공유",
        "매매 타이밍",
        "투자 학습 질문",
        "기타 투자",
    ],
    "기타": [
        "인사/안부",
        "축하/격려/응원",
        "일상 공유",
        "테스트/기타",
    ],
}

ALL_VALID_TAGS = set()
for tags in CATEGORY_TAGS.values():
    ALL_VALID_TAGS.update(tags)


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
    method: str = "v3",
) -> Dict:
    """분류 결과 딕셔너리 생성"""
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
        "method": method,
    }
