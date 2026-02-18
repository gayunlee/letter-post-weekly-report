#!/usr/bin/env python3
"""GPT-4o-mini 파인튜닝용 데이터 준비

사용법:
    python scripts/prepare_finetune_data.py
"""
import json
import random
from pathlib import Path

# 카테고리 정의
CATEGORIES = ["서비스 이슈", "서비스 칭찬", "투자 질문", "정보/의견", "일상 소통"]

SYSTEM_PROMPT = """너는 금융 콘텐츠 플랫폼의 VOC 분류 전문가야. 다음 5개 카테고리 중 하나로 분류해:
1. 서비스 이슈: 서비스/운영에 대한 불만, 문의, 개선요청
2. 서비스 칭찬: 콘텐츠/마스터/플랫폼에 대한 구체적 감사, 칭찬
3. 투자 질문: 종목, 비중, 매매 타이밍 등 투자 관련 질문
4. 정보/의견: 뉴스, 분석, 시장 전망, 수익/손실 후기, 투자 심리
5. 일상 소통: 인사, 안부, 축하, 조문, 가입인사"""


def main():
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "labeling" / "gpt4o_labeled.json"
    output_dir = project_root / "data" / "finetune"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GPT-4o-mini 파인튜닝 데이터 준비")
    print("=" * 60)

    # 데이터 로드
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"로드: {len(data)}건")

    # 카테고리별 분리
    by_category = {cat: [] for cat in CATEGORIES}
    for item in data:
        cat = item.get("category")
        if cat in by_category:
            by_category[cat].append(item)

    print("\n카테고리별 데이터:")
    for cat, items in by_category.items():
        print(f"  {cat}: {len(items)}건")

    # 균형 잡힌 샘플링 (카테고리당 최대 150건)
    max_per_category = 150
    balanced_data = []
    for cat, items in by_category.items():
        random.seed(42)
        sampled = random.sample(items, min(len(items), max_per_category))
        balanced_data.extend(sampled)
        print(f"  {cat}: {len(sampled)}건 샘플링")

    random.shuffle(balanced_data)

    # Train/Validation 분리 (90/10)
    split_idx = int(len(balanced_data) * 0.9)
    train_data = balanced_data[:split_idx]
    val_data = balanced_data[split_idx:]

    print(f"\n훈련: {len(train_data)}건, 검증: {len(val_data)}건")

    # JSONL 형식으로 변환
    def to_finetune_format(item):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["text"][:500]},  # 500자 제한
                {"role": "assistant", "content": item["category"]}
            ]
        }

    # 저장
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(to_finetune_format(item), ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(to_finetune_format(item), ensure_ascii=False) + "\n")

    print(f"\n저장 완료:")
    print(f"  훈련: {train_path}")
    print(f"  검증: {val_path}")

    # 샘플 확인
    print("\n샘플 데이터:")
    sample = to_finetune_format(train_data[0])
    print(json.dumps(sample, ensure_ascii=False, indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
