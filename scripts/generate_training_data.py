#!/usr/bin/env python3
"""이유 포함 학습 데이터 생성

Claude API를 사용해서 각 케이스별 분류 이유 생성
"""
import json
import os
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# 시스템 프롬프트 (파인튜닝용)
SYSTEM_PROMPT = """너는 금융 콘텐츠 플랫폼 '어스(us)'의 VOC 분석가야.

[플랫폼 개요]
- 마스터: 투자 전문가, 콘텐츠 크리에이터 (예: 박두환, 서재형 등)
- 오피셜클럽: 마스터가 운영하는 유료 구독 커뮤니티
- 편지글: 구독자→마스터 1:1 메시지 (감사, 일상, 소통이 주)
- 게시글: 커뮤니티 게시판 (정보 공유, 수익 후기 등)
- 서비스: 온라인 콘텐츠(강의, 레포트) + 오프라인 세미나

[분류 카테고리]
1. 서비스 이슈: 플랫폼 운영/기능 문제 (링크오류, 결제, 배송, 신청방법, 자료지연, 답변지연)
2. 서비스 칭찬: 콘텐츠/마스터에 대한 구체적 감사 ("덕분에 배웠다", "수익났다")
3. 투자 질문: 종목, 비중, 매매 타이밍 질문
4. 정보/의견: 증시 뉴스, 분석, 수익/손실 후기, 투자 심리
5. 일상 소통: 인사, 안부, 축하, 조문, 형식적 감사

[핵심 구분]
- 증시/투자 이야기 ≠ 서비스 이슈 (아무리 부정적이어도)
- 구체적 감사 = 서비스 칭찬, 형식적 감사 = 일상 소통
- 운영/기능 문의 = 서비스 이슈"""

# 이유 생성용 프롬프트
REASON_PROMPT = """다음 텍스트를 분류하고 짧은 이유를 작성해줘.

[텍스트 유형]: {source}
[텍스트]: {text}

다음 형식으로 답변해:
카테고리: [서비스 이슈/서비스 칭찬/투자 질문/정보/의견/일상 소통]
이유: [1문장으로 간결하게]

분류 기준:
- 서비스 이슈: 플랫폼 기능/운영 문제 (링크, 결제, 배송, 신청, 자료, 답변지연)
- 서비스 칭찬: 콘텐츠/마스터에 대한 구체적 감사
- 투자 질문: 종목, 비중, 매매 타이밍 질문
- 정보/의견: 증시 뉴스, 분석, 수익/손실 후기, 투자 심리 표현
- 일상 소통: 인사, 안부, 형식적 감사

주의: 증시/투자 관련 불만은 서비스 이슈가 아님!"""


def load_labeled_data(path: str):
    """라벨링된 데이터 로드"""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def select_samples(data: list, samples_per_cat: int = 50):
    """카테고리별 균형 잡힌 샘플 선택"""
    by_category = defaultdict(list)

    for item in data:
        cat = item.get('category')
        if cat:
            by_category[cat].append(item)

    selected = []
    for cat, items in by_category.items():
        random.seed(42)
        random.shuffle(items)

        # 다양한 길이 선택
        short = [d for d in items if len(d['text']) < 50][:samples_per_cat//3]
        medium = [d for d in items if 50 <= len(d['text']) < 200][:samples_per_cat//3]
        long_items = [d for d in items if len(d['text']) >= 200][:samples_per_cat//3]

        cat_selected = short + medium + long_items

        # 부족하면 추가
        remaining = samples_per_cat - len(cat_selected)
        if remaining > 0:
            others = [d for d in items if d not in cat_selected][:remaining]
            cat_selected.extend(others)

        selected.extend(cat_selected)
        print(f"  {cat}: {len(cat_selected)}개 선택")

    return selected


def generate_reason(client: Anthropic, text: str, source: str) -> tuple:
    """Claude로 분류 이유 생성"""
    source_kr = "편지글" if source == "letter" else "게시글"

    prompt = REASON_PROMPT.format(source=source_kr, text=text[:300])

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text.strip()

        # 파싱
        lines = response.split('\n')
        category = None
        reason = None

        for line in lines:
            if line.startswith('카테고리:'):
                category = line.replace('카테고리:', '').strip()
            elif line.startswith('이유:'):
                reason = line.replace('이유:', '').strip()

        return category, reason

    except Exception as e:
        print(f"  오류: {e}")
        return None, None


def create_training_example(text: str, source: str, category: str, reason: str):
    """학습 예시 생성"""
    source_kr = "편지글" if source == "letter" else "게시글"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[{source_kr}] {text[:300]}"},
            {"role": "assistant", "content": f"{category}\n\n이유: {reason}"}
        ]
    }


def main():
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "labeling" / "gpt4o_labeled.json"
    output_path = project_root / "data" / "finetune" / "training_data_with_reasons.jsonl"

    print("=" * 60)
    print("이유 포함 학습 데이터 생성")
    print("=" * 60)

    # 데이터 로드
    print("\n1. 데이터 로드")
    data = load_labeled_data(str(input_path))
    print(f"  총 {len(data)}건")

    # 샘플 선택
    print("\n2. 샘플 선택 (카테고리당 50개)")
    samples = select_samples(data, samples_per_cat=50)
    print(f"  총 {len(samples)}개 선택")

    # Claude로 이유 생성
    print("\n3. 분류 이유 생성 중...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    training_data = []

    for i, item in enumerate(samples):
        text = item['text'].replace('\n', ' ').strip()
        source = item['source']
        original_cat = item['category']

        # Claude로 분류 + 이유 생성
        category, reason = generate_reason(client, text, source)

        if category and reason:
            # Claude가 다르게 분류한 경우 로그
            if category != original_cat:
                print(f"  [{i+1}] 분류 변경: {original_cat} → {category}")

            example = create_training_example(text, source, category, reason)
            training_data.append(example)
        else:
            # 실패 시 원본 카테고리 사용 + 기본 이유
            reason = f"{original_cat} 카테고리에 해당하는 내용."
            example = create_training_example(text, source, original_cat, reason)
            training_data.append(example)

        if (i + 1) % 25 == 0:
            print(f"  진행: {i+1}/{len(samples)}")

    # 저장
    print(f"\n4. 저장: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"  총 {len(training_data)}개 학습 예시 생성")

    # 분포 확인
    from collections import Counter
    cats = Counter()
    for item in training_data:
        cat = item['messages'][2]['content'].split('\n')[0]
        cats[cat] += 1

    print("\n카테고리 분포:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}개")


if __name__ == "__main__":
    main()
