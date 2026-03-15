"""채널톡 3분류 — Claude Code 판단 규칙 기반 라벨링

Claude가 100건 샘플을 직접 검토하여 도출한 분류 규칙.
3분류: 결제·구독 / 콘텐츠·수강 / 기술·오류

우선순위:
1. 결제·구독: 돈/결제/환불/구독 키워드가 있으면 무조건
2. 기술·오류: 시스템 문제가 명확한 경우만
3. 콘텐츠·수강: 나머지 전부
"""
import json
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent

# === 워크플로우 버튼 제거 ===
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
    "🏠처음으로",
]


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


# === 분류 규칙 ===

# 결제·구독 키워드 (최우선)
PAYMENT_KEYWORDS = [
    "결제", "결재", "환불", "구독", "해지", "탈퇴", "카드",
    "자동결제", "자동결재", "자동이체", "계좌이체", "이체",
    "영수증", "할부", "정기구독", "상품변경", "상품 변경",
    "멤버십", "멤버쉽", "맴버십", "맴버쉽",
    "가격", "요금", "수강료", "구독료", "회비",
    "연장", "갱신", "재구독", "재결제", "재결재",
    "취소", "반환", "입금", "출금", "통장",
    "만원", "원 ", "5천원", "10만", "50만", "85000", "100000",
    "월로", "6개월", "1개월", "한달", "신청",
    "결제일", "만료", "해지일",
    "무통장", "법인카드",
]

# 결제·구독 패턴 (정규식)
PAYMENT_PATTERNS = [
    r"구독.{0,5}(해지|취소|변경|정지)",
    r"(해지|취소|변경).{0,5}(부탁|요청|하고|하려|싶)",
    r"자동.{0,3}(결제|결재|이체)",
    r"(환불|반환).{0,5}(해|부탁|요청|처리|확인|가능)",
    r"(결제|결재).{0,5}(취소|안|변경|확인|방법|하려|했는데|됐|됨)",
    r"(카드|계좌).{0,5}(변경|등록|이체)",
    r"(탈퇴|해지).{0,5}(하려|하고|부탁|요청|싶|합니다)",
    r"(가입|신청).{0,5}(취소|철회)",
    r"(연장|갱신).{0,5}(하고|하려|싶|합니다|부탁)",
    r"\d+만\s*원",
    r"(얼마|가격|요금)",
]

# 기술·오류 키워드
TECH_KEYWORDS = [
    "오류", "에러", "error", "버그",
    "로그인", "로그 인", "접속",
    "안됨", "안돼", "안되", "안 됨", "안 돼", "안 되",
    "안열", "안 열", "열리지",
    "크래시", "다운", "멈춤", "튕김",
    "화면이 않", "화면이 안",
    "비활성화", "비번", "비밀번호",
    "뺑뺑이", "무한로딩", "로딩",
    "unknown_failure", "설치",
]

# 기술·오류 패턴
TECH_PATTERNS = [
    r"(앱|어플|사이트|화면|페이지|링크).{0,5}(안|오류|에러|멈|튕|않)",
    r"(로그인|접속|입장).{0,5}(안|불가|못|실패|어려)",
    r"(시스템|서버).{0,5}(오류|에러|문제|장애)",
    r"(기능|프린트|검색|자막).{0,5}(추가|개선|건의|요청)",
    r"(업데이트|설치).{0,5}(이후|후|오류)",
    r"(안\s?보|못\s?봄|안\s?열|못\s?들어)",
]

# 기술 오류처럼 보이지만 결제 맥락인 경우 제외할 패턴
TECH_BUT_PAYMENT = [
    r"결제.{0,10}(안|오류|실패)",  # "결제 안 됨"은 결제·구독
    r"(결재|결제).{0,5}(안됨|안돼|안 됨)",
]

# 콘텐츠·수강으로 가야 할 기술 키워드 (콘텐츠 접근 문의)
CONTENT_OVERRIDE = [
    r"(강의|수업|녹화|라이브).{0,5}(안|못|어디|방법|시청)",
    r"(수강|시청).{0,5}(방법|어떻게)",
    r"줌.{0,5}(다운|설치|방법|어떻게)",
]


def classify(text: str) -> str:
    """텍스트를 3분류 중 하나로 분류."""
    cleaned = strip_workflow_buttons(text)
    if len(cleaned.strip()) < 3:
        # 워크플로우 버튼만 있는 경우
        # 버튼 내용으로 판단
        if any(kw in text for kw in ["결제", "결재", "구독", "환불"]):
            return "결제·구독"
        return "콘텐츠·수강"

    t = cleaned.lower()

    # 1단계: 결제·구독 키워드 체크 (최우선)
    payment_score = 0
    for kw in PAYMENT_KEYWORDS:
        if kw in t:
            payment_score += 1

    for pattern in PAYMENT_PATTERNS:
        if re.search(pattern, t):
            payment_score += 2

    # 2단계: 기술·오류 키워드 체크
    tech_score = 0
    for kw in TECH_KEYWORDS:
        if kw in t:
            tech_score += 1

    for pattern in TECH_PATTERNS:
        if re.search(pattern, t):
            tech_score += 2

    # 기술처럼 보이지만 결제 맥락인 경우
    for pattern in TECH_BUT_PAYMENT:
        if re.search(pattern, t):
            tech_score -= 2
            payment_score += 1

    # 콘텐츠 접근 문의 (기술이 아닌 수강 방법)
    for pattern in CONTENT_OVERRIDE:
        if re.search(pattern, t):
            tech_score -= 1

    # 3단계: 판정
    if payment_score >= 1:
        return "결제·구독"

    if tech_score >= 2:
        return "기술·오류"

    return "콘텐츠·수강"


def main():
    # 데이터 로드
    with open(PROJECT_ROOT / "data/channel_io/training_data/_unlabeled_8798.json") as f:
        items = json.load(f)

    print(f"총 {len(items)}건 분류 시작...")

    results = []
    for item in items:
        topic = classify(item["text"])
        results.append({
            "chatId": item["chatId"],
            "text": item["text"],
            "topic": topic,
        })

    # 통계
    dist = Counter(r["topic"] for r in results)
    print(f"\n분류 완료: {len(results)}건")
    print("분포:")
    for topic, count in dist.most_common():
        print(f"  {topic}: {count}건 ({count/len(results)*100:.1f}%)")

    # 저장
    output_path = PROJECT_ROOT / "data/channel_io/training_data/labeled_claude_8798.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {output_path}")

    # 기존 학습 데이터와 합치기
    with open(PROJECT_ROOT / "data/channel_io/training_data/labeled_1500_v5.json") as f:
        existing = json.load(f)

    combined = existing + results
    combined_dist = Counter(r["topic"] for r in combined)
    print(f"\n=== 합산 ===")
    print(f"기존: {len(existing)} + 신규: {len(results)} = 총 {len(combined)}건")
    print("분포:")
    for topic, count in combined_dist.most_common():
        print(f"  {topic}: {count}건 ({count/len(combined)*100:.1f}%)")

    combined_path = PROJECT_ROOT / "data/channel_io/training_data/labeled_combined_10126.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"저장: {combined_path}")


if __name__ == "__main__":
    main()
