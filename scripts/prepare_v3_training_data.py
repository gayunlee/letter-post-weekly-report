"""v3 훈련 데이터 준비 스크립트

LLM 라벨링 결과를 KcBERT 파인튜닝용 train/val/test split으로 변환.

사용법:
    python3 scripts/prepare_v3_training_data.py
    python3 scripts/prepare_v3_training_data.py --test-ratio 0.15 --val-ratio 0.1
"""
import sys
import os
import json
import argparse
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

V3_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠 반응", "투자 담론", "기타"]
TOPIC_TO_ID = {t: i for i, t in enumerate(V3_TOPICS)}
ID_TO_TOPIC = {i: t for t, i in TOPIC_TO_ID.items()}


def main():
    parser = argparse.ArgumentParser(description="v3 훈련 데이터 준비")
    parser.add_argument("--input", default="./data/training_data/v3/labeled_all.json")
    parser.add_argument("--output-dir", default="./data/training_data/v3")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="최소 신뢰도 (이하는 제외)")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("  v3 훈련 데이터 준비")
    print("=" * 70)

    # 라벨링 데이터 로드
    with open(args.input, encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"\n  전체 라벨링 데이터: {len(all_data)}건")

    # 필터링: 최소 신뢰도 + 텍스트 길이
    filtered = []
    excluded_low_conf = 0
    excluded_short = 0
    for item in all_data:
        if item["v3_confidence"] < args.min_confidence:
            excluded_low_conf += 1
            continue
        if not item.get("text") or len(item["text"].strip()) < 20:
            excluded_short += 1
            continue
        filtered.append(item)

    print(f"  신뢰도 < {args.min_confidence} 제외: {excluded_low_conf}건")
    print(f"  텍스트 짧은 건 제외: {excluded_short}건")
    print(f"  훈련 대상: {len(filtered)}건")

    # 분포 확인
    dist = Counter(item["v3_topic"] for item in filtered)
    print(f"\n  분포:")
    for topic in V3_TOPICS:
        count = dist.get(topic, 0)
        pct = count / len(filtered) * 100 if filtered else 0
        print(f"    {topic}: {count}건 ({pct:.1f}%)")

    # Stratified split
    by_topic = {}
    for item in filtered:
        by_topic.setdefault(item["v3_topic"], []).append(item)

    train_data, val_data, test_data = [], [], []

    for topic, items in by_topic.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * args.test_ratio))
        n_val = max(1, int(n * args.val_ratio))

        test_items = items[:n_test]
        val_items = items[n_test:n_test + n_val]
        train_items = items[n_test + n_val:]

        for item in train_items:
            train_data.append({
                "text": item["text"],
                "label": TOPIC_TO_ID[topic],
                "category": topic,
            })
        for item in val_items:
            val_data.append({
                "text": item["text"],
                "label": TOPIC_TO_ID[topic],
                "category": topic,
            })
        for item in test_items:
            test_data.append({
                "text": item["text"],
                "label": TOPIC_TO_ID[topic],
                "category": topic,
            })

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print(f"\n  Split:")
    print(f"    Train: {len(train_data)}건")
    print(f"    Val:   {len(val_data)}건")
    print(f"    Test:  {len(test_data)}건")

    # 저장
    topic_dir = Path(args.output_dir) / "topic"
    topic_dir.mkdir(parents=True, exist_ok=True)

    with open(topic_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(topic_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(topic_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    mapping = {
        "category_to_id": TOPIC_TO_ID,
        "id_to_category": {str(v): k for k, v in TOPIC_TO_ID.items()},
        "num_labels": len(V3_TOPICS),
    }
    with open(topic_dir / "category_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 분포 리포트
    print(f"\n  Train 분포:")
    train_dist = Counter(item["category"] for item in train_data)
    for topic in V3_TOPICS:
        print(f"    {topic}: {train_dist.get(topic, 0)}건")

    print(f"\n  저장: {topic_dir}/")
    print("  파일: train.json, val.json, test.json, category_mapping.json")


if __name__ == "__main__":
    main()
