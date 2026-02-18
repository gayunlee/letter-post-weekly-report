"""2축 분류 체계 정의

축1 (주제/Topic): 무엇에 대한 글인가
축2 (감성/Sentiment): 어떤 톤인가

기존 1축 체계와의 매핑도 제공합니다.
"""
from typing import Dict, Tuple, Optional


# ── 축1: 주제 (Topic) ──────────────────────────────────────────────
TOPIC_CATEGORIES = {
    "콘텐츠 반응": "마스터의 콘텐츠(강의, 리포트, 방송)에 대한 직접적 반응",
    "투자 이야기": "투자 전략, 종목, 포트폴리오, 시장 분석 관련 글",
    "서비스 이슈": "플랫폼/앱 기능, 결제, 배송, 구독 등 서비스 운영 관련",
    "커뮤니티 소통": "인사, 안부, 축하, 일상 공유, 커뮤니티 교류",
}

TOPIC_TO_ID = {cat: i for i, cat in enumerate(TOPIC_CATEGORIES)}
ID_TO_TOPIC = {i: cat for cat, i in TOPIC_TO_ID.items()}

# ── 축2: 감성 (Sentiment) ──────────────────────────────────────────
SENTIMENT_CATEGORIES = {
    "긍정": "감사, 만족, 기쁨, 칭찬, 격려",
    "부정": "불만, 실망, 걱정, 비판, 답답함",
    "중립": "질문, 정보 전달, 사실 기술, 요청",
}

SENTIMENT_TO_ID = {cat: i for i, cat in enumerate(SENTIMENT_CATEGORIES)}
ID_TO_SENTIMENT = {i: cat for cat, i in SENTIMENT_TO_ID.items()}


# ── 1축 → 2축 기본 매핑 ────────────────────────────────────────────
# 기존 7개 카테고리에서 2축으로의 기본 변환 (키워드 분석 전 기본값)
ONE_AXIS_TO_TWO_AXIS: Dict[str, Tuple[str, str]] = {
    "감사·후기":     ("콘텐츠 반응", "긍정"),
    "질문·토론":     ("투자 이야기", "중립"),
    "정보성 글":     ("투자 이야기", "중립"),
    "서비스 피드백":  ("서비스 이슈", "중립"),
    "서비스 불편사항": ("서비스 이슈", "부정"),
    "서비스 제보/건의": ("서비스 이슈", "중립"),
    "일상·공감":     ("커뮤니티 소통", "긍정"),
}


# ── 2축 → 1축 역매핑 ───────────────────────────────────────────────
# 리포트 호환용: 2축 조합에서 기존 카테고리로 복원
TWO_AXIS_TO_ONE_AXIS: Dict[Tuple[str, str], str] = {
    ("콘텐츠 반응", "긍정"):  "감사·후기",
    ("콘텐츠 반응", "부정"):  "일상·공감",      # 콘텐츠 불만 → 일상·공감
    ("콘텐츠 반응", "중립"):  "감사·후기",      # 콘텐츠 중립 반응 → 감사·후기
    ("투자 이야기", "긍정"):  "감사·후기",      # 수익 자랑 등
    ("투자 이야기", "부정"):  "일상·공감",      # 투자 불만
    ("투자 이야기", "중립"):  "질문·토론",      # 질문/정보 공유
    ("서비스 이슈", "긍정"):  "서비스 피드백",
    ("서비스 이슈", "부정"):  "서비스 불편사항",
    ("서비스 이슈", "중립"):  "서비스 피드백",
    ("커뮤니티 소통", "긍정"): "일상·공감",
    ("커뮤니티 소통", "부정"): "일상·공감",
    ("커뮤니티 소통", "중립"): "일상·공감",
}


def to_two_axis(one_axis_category: str) -> Tuple[str, str]:
    """1축 카테고리를 2축(topic, sentiment)으로 변환 (기본 매핑)"""
    return ONE_AXIS_TO_TWO_AXIS.get(one_axis_category, ("커뮤니티 소통", "중립"))


def to_one_axis(topic: str, sentiment: str) -> str:
    """2축(topic, sentiment)을 1축 카테고리로 변환"""
    return TWO_AXIS_TO_ONE_AXIS.get((topic, sentiment), "일상·공감")


# ── 키워드 기반 감성 보정 규칙 ──────────────────────────────────────
# 1축→2축 기본 매핑 후, 텍스트 키워드로 sentiment를 보정
import re

_POSITIVE_KEYWORDS = re.compile(
    r'감사|고마|고맙|덕분|덕에|감동|최고|짱|좋은|대박|수익|올라|불을 뿜|'
    r'리스펙|존경|응원|화이팅|파이팅|축하|행복|기쁨|사랑|멋지|훌륭'
)

_NEGATIVE_KEYWORDS = re.compile(
    r'불만|실망|답답|환불|불편|짜증|화[가나]|속상|걱정|무서|불안|'
    r'물렸|손실|하락|폭락|안나|안 나|못 받|안 받|안돼|안 돼|오류|버그'
)

_QUESTION_KEYWORDS = re.compile(
    r'\?|인가요|할까요|일까요|나요|인지요|주실|줄 수|어떤가요|어떨까|'
    r'궁금|문의|여쭤|질문|알려주|어떻게'
)


def refine_sentiment(text: str, default_sentiment: str) -> str:
    """텍스트 키워드를 분석하여 sentiment를 보정"""
    if not text:
        return default_sentiment

    has_positive = bool(_POSITIVE_KEYWORDS.search(text))
    has_negative = bool(_NEGATIVE_KEYWORDS.search(text))
    has_question = bool(_QUESTION_KEYWORDS.search(text))

    # 질문이 있으면 중립 우선
    if has_question:
        return "중립"

    # 긍정+부정 동시 → 기본값 유지
    if has_positive and has_negative:
        return default_sentiment

    if has_positive:
        return "긍정"
    if has_negative:
        return "부정"

    return default_sentiment


def classify_two_axis(text: str, one_axis_category: str) -> Tuple[str, str]:
    """1축 카테고리 + 텍스트로 2축 분류 수행

    기존 1축 분류 결과를 기반으로, 텍스트 키워드 분석을 추가하여
    topic과 sentiment를 결정합니다.
    """
    topic, default_sentiment = to_two_axis(one_axis_category)
    sentiment = refine_sentiment(text, default_sentiment)
    return topic, sentiment
