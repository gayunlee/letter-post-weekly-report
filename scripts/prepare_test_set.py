#!/usr/bin/env python3
"""테스트셋 준비: 100건 균등 추출 + 도메인 이해도 테스트 20건

사용법:
    python scripts/prepare_test_set.py
"""
import json
import random
from pathlib import Path
from collections import Counter

# 5개 카테고리
CATEGORIES = ["서비스 이슈", "서비스 칭찬", "투자 질문", "정보/의견", "일상 소통"]


def load_labeled_data(path: Path) -> list:
    """라벨링된 데이터 로드"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_balanced_testset(data: list, samples_per_category: int = 20) -> list:
    """카테고리별 균등 추출"""
    random.seed(42)  # 재현성

    testset = []
    for category in CATEGORIES:
        cat_items = [d for d in data if d.get("category") == category]
        if len(cat_items) < samples_per_category:
            print(f"  경고: {category}는 {len(cat_items)}건만 있음 (전체 추출)")
            sampled = cat_items
        else:
            sampled = random.sample(cat_items, samples_per_category)
        testset.extend(sampled)

    return testset


def create_domain_test_cases() -> list:
    """도메인 이해도 테스트 케이스 (직접 라벨링)

    이 케이스들은 도메인 특수성으로 인해 혼동하기 쉬운 예시들임
    """
    return [
        # 1-4: 증시 불만 vs 서비스 불만 구분
        {
            "id": "domain_001",
            "text": "요즘 장이 너무 힘드네요. 계좌가 녹아요 ㅠㅠ",
            "category": "정보/의견",
            "reason": "증시/투자 상황에 대한 감정 토로 (서비스 이슈 ❌)"
        },
        {
            "id": "domain_002",
            "text": "최근 시장이 미쳐 돌아가네요. 손절이 답인가요...",
            "category": "정보/의견",
            "reason": "투자 심리 표현, 반문형 수사 (질문 ❌)"
        },
        {
            "id": "domain_003",
            "text": "하락장에 멘탈이 흔들리네요. 버티기 힘들어요",
            "category": "정보/의견",
            "reason": "투자 심리 토로 (서비스 이슈 ❌)"
        },
        {
            "id": "domain_004",
            "text": "오늘도 마이너스네요 ㅠㅠ 언제쯤 반등할까요",
            "category": "정보/의견",
            "reason": "투자 심리 + 수사적 질문 (서비스 이슈/투자 질문 ❌)"
        },

        # 5-8: 형식적 감사 vs 구체적 감사 구분
        {
            "id": "domain_005",
            "text": "감사합니다",
            "category": "일상 소통",
            "reason": "형식적 감사 한마디 (서비스 칭찬 ❌)"
        },
        {
            "id": "domain_006",
            "text": "항상 감사합니다~",
            "category": "일상 소통",
            "reason": "형식적 감사 표현 (서비스 칭찬 ❌)"
        },
        {
            "id": "domain_007",
            "text": "오늘도 좋은 콘텐츠 감사합니다. 덕분에 시장 흐름을 잘 파악할 수 있었어요",
            "category": "서비스 칭찬",
            "reason": "구체적 도움 언급한 감사 (일상 소통 ❌)"
        },
        {
            "id": "domain_008",
            "text": "레포트 보고 삼전 매수했는데 수익났습니다. 감사합니다!",
            "category": "서비스 칭찬",
            "reason": "콘텐츠 활용 결과 언급 (정보/의견 ❌)"
        },

        # 9-12: 서비스 이슈 (운영/기능 문제)
        {
            "id": "domain_009",
            "text": "레포트 언제 올라오나요?",
            "category": "서비스 이슈",
            "reason": "콘텐츠 업로드 지연 문의 (투자 질문 ❌)"
        },
        {
            "id": "domain_010",
            "text": "링크가 깨져서 영상이 안 보여요",
            "category": "서비스 이슈",
            "reason": "기술 오류 신고"
        },
        {
            "id": "domain_011",
            "text": "교재가 아직 배송 안 왔는데 언제쯤 받을 수 있나요?",
            "category": "서비스 이슈",
            "reason": "배송 지연 문의"
        },
        {
            "id": "domain_012",
            "text": "결제했는데 결제 내역이 안 보여요",
            "category": "서비스 이슈",
            "reason": "결제/시스템 오류"
        },

        # 13-16: 투자 질문 (실제 답변 기대)
        {
            "id": "domain_013",
            "text": "삼성전자 지금 추매해도 될까요? 비중 어느 정도가 적당할까요?",
            "category": "투자 질문",
            "reason": "매매 타이밍 + 비중 질문"
        },
        {
            "id": "domain_014",
            "text": "현재 포트폴리오에서 반도체 비중이 50%인데 줄여야 할까요?",
            "category": "투자 질문",
            "reason": "포트폴리오 조언 요청"
        },
        {
            "id": "domain_015",
            "text": "두산에너지빌리티랑 한전 중에 어떤 게 더 좋을까요?",
            "category": "투자 질문",
            "reason": "종목 비교 질문"
        },
        {
            "id": "domain_016",
            "text": "선생님 의견 여쭤봐도 될까요? SK하이닉스 목표가가 어디쯤일까요?",
            "category": "투자 질문",
            "reason": "목표가 질문"
        },

        # 17-20: 복합적/경계 케이스
        {
            "id": "domain_017",
            "text": "새해 복 많이 받으세요! 올해도 잘 부탁드립니다~",
            "category": "일상 소통",
            "reason": "인사/안부"
        },
        {
            "id": "domain_018",
            "text": "오늘 뉴스 보니까 금리 동결이라는데 시장 영향이 어떨까요?",
            "category": "정보/의견",
            "reason": "뉴스 언급 + 수사적 의문문 (투자 질문 ❌)"
        },
        {
            "id": "domain_019",
            "text": "이번 강의에서 말씀하신 차트 분석 방법대로 해봤는데 잘 됩니다",
            "category": "서비스 칭찬",
            "reason": "콘텐츠 활용 후기 (정보/의견 ❌)"
        },
        {
            "id": "domain_020",
            "text": "영상 퀄리티가 예전보다 많이 안 좋아진 것 같아요",
            "category": "서비스 이슈",
            "reason": "콘텐츠 품질 피드백 (정보/의견 ❌)"
        },
    ]


def main():
    project_root = Path(__file__).parent.parent
    labeled_path = project_root / "data" / "labeling" / "gpt4o_labeled.json"
    test_path = project_root / "data" / "test" / "test_set_100.json"
    domain_test_path = project_root / "data" / "test" / "domain_test_20.json"

    print("=" * 60)
    print("테스트셋 준비")
    print("=" * 60)

    # 1. 라벨링 데이터 로드
    print("\n1. 라벨링 데이터 로드")
    data = load_labeled_data(labeled_path)
    print(f"  총 {len(data)}건")

    # 분포 확인
    dist = Counter(d.get("category") for d in data)
    print("\n  카테고리 분포:")
    for cat in CATEGORIES:
        print(f"    {cat}: {dist.get(cat, 0)}건")

    # 2. 균등 추출 (100건)
    print("\n2. 테스트셋 추출 (카테고리당 20건)")
    testset = extract_balanced_testset(data, samples_per_category=20)

    test_dist = Counter(d.get("category") for d in testset)
    print(f"  추출 결과: {len(testset)}건")
    for cat in CATEGORIES:
        print(f"    {cat}: {test_dist.get(cat, 0)}건")

    # 저장
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {test_path}")

    # 3. 도메인 이해도 테스트 (20건)
    print("\n3. 도메인 이해도 테스트 케이스 생성")
    domain_tests = create_domain_test_cases()

    domain_dist = Counter(d.get("category") for d in domain_tests)
    print(f"  생성: {len(domain_tests)}건")
    for cat in CATEGORIES:
        print(f"    {cat}: {domain_dist.get(cat, 0)}건")

    with open(domain_test_path, "w", encoding="utf-8") as f:
        json.dump(domain_tests, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {domain_test_path}")

    # 4. 샘플 출력
    print("\n4. 도메인 테스트 케이스 샘플")
    for item in domain_tests[:5]:
        print(f"\n  [{item['category']}]")
        print(f"    텍스트: {item['text']}")
        print(f"    근거: {item['reason']}")

    print("\n" + "=" * 60)
    print("완료!")
    print(f"  - 메인 테스트셋: {test_path} ({len(testset)}건)")
    print(f"  - 도메인 테스트: {domain_test_path} ({len(domain_tests)}건)")


if __name__ == "__main__":
    main()
