"""v3 훈련 데이터 준비 스크립트

LLM 라벨링 결과를 KcBERT 파인튜닝용 train/val/test split으로 변환.
Golden set 텍스트를 split 전에 제외하여 학습/평가 데이터 오염을 방지.

사용법:
    python3 scripts/prepare_v3_training_data.py
    python3 scripts/prepare_v3_training_data.py \
      --input ./data/training_data/v3/labeled_all_v3c.json \
      --topic-field v3c_topic \
      --output-dir ./data/training_data/v3/topic_v3c_clean
"""
import sys
import os
import json
import argparse
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 기본 5분류 (v3 원본). --topic-field에 따라 데이터에서 자동 감지.
V3_TOPICS_DEFAULT = ["운영 피드백", "서비스 피드백", "콘텐츠 반응", "투자 담론", "기타"]

# v3c 5분류
V3C_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "일상·감사", "기타"]

TOPIC_PRESETS = {
    "v3_topic": V3_TOPICS_DEFAULT,
    "v3c_topic": V3C_TOPICS,
}


def main():
    parser = argparse.ArgumentParser(description="v3 훈련 데이터 준비")
    parser.add_argument("--input", default="./data/training_data/v3/labeled_all.json")
    parser.add_argument("--output-dir", default="./data/training_data/v3/topic")
    parser.add_argument("--golden-dir", default="./data/gold_dataset",
                        help="Golden set 디렉토리 (*_golden_set.json 자동 탐색)")
    parser.add_argument("--topic-field", default="v3_topic",
                        help="라벨링 JSON의 topic 필드명 (예: v3_topic, v3c_topic)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="최소 신뢰도 (이하는 제외)")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    topic_field = args.topic_field

    print("=" * 70)
    print("  v3 훈련 데이터 준비")
    print("=" * 70)

    # 라벨링 데이터 로드
    with open(args.input, encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"\n  전체 라벨링 데이터: {len(all_data)}건")
    print(f"  Topic 필드: {topic_field}")

    # 필터링: 최소 신뢰도 + 텍스트 길이
    filtered = []
    excluded_low_conf = 0
    excluded_short = 0
    for item in all_data:
        if item.get("v3_confidence", 1.0) < args.min_confidence:
            excluded_low_conf += 1
            continue
        if not item.get("text") or len(item["text"].strip()) < 20:
            excluded_short += 1
            continue
        filtered.append(item)

    print(f"  신뢰도 < {args.min_confidence} 제외: {excluded_low_conf}건")
    print(f"  텍스트 짧은 건 제외: {excluded_short}건")

    # Golden set 제외 (split 전 — 학습/평가 데이터 오염 방지)
    golden_dir = Path(args.golden_dir)
    golden_texts = set()
    if golden_dir.exists():
        for gf in sorted(golden_dir.glob("*_golden_set.json")):
            with open(gf, encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                golden_texts.add(item["text"].strip())
            print(f"    {gf.name}: {len(items)}건 로드")

    if golden_texts:
        before = len(filtered)
        filtered = [item for item in filtered if item["text"].strip() not in golden_texts]
        excluded_golden = before - len(filtered)
        print(f"  Golden set 제외: {excluded_golden}건 (golden 전체 {len(golden_texts)}건 중)")
    else:
        print(f"  Golden set: 없음 ({golden_dir})")

    print(f"  훈련 대상: {len(filtered)}건")

    # 카테고리 매핑 결정 (프리셋 또는 데이터에서 자동 감지)
    if topic_field in TOPIC_PRESETS:
        topics = TOPIC_PRESETS[topic_field]
    else:
        topics = sorted(set(item[topic_field] for item in filtered))
    topic_to_id = {t: i for i, t in enumerate(topics)}

    # 분포 확인
    dist = Counter(item[topic_field] for item in filtered)
    print(f"\n  분포:")
    for topic in topics:
        count = dist.get(topic, 0)
        pct = count / len(filtered) * 100 if filtered else 0
        print(f"    {topic}: {count}건 ({pct:.1f}%)")

    # 데이터에 없는 카테고리 경고
    data_topics = set(dist.keys())
    unknown = data_topics - set(topics)
    if unknown:
        print(f"\n  ⚠ 미등록 카테고리 발견: {unknown}")
        print(f"    이 항목들은 제외됩니다.")
        filtered = [item for item in filtered if item[topic_field] in topic_to_id]

    # Stratified split
    by_topic = {}
    for item in filtered:
        by_topic.setdefault(item[topic_field], []).append(item)

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
                "label": topic_to_id[topic],
                "category": topic,
            })
        for item in val_items:
            val_data.append({
                "text": item["text"],
                "label": topic_to_id[topic],
                "category": topic,
            })
        for item in test_items:
            test_data.append({
                "text": item["text"],
                "label": topic_to_id[topic],
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(out_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    mapping = {
        "category_to_id": topic_to_id,
        "id_to_category": {str(v): k for k, v in topic_to_id.items()},
        "num_labels": len(topics),
    }
    with open(out_dir / "category_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 분포 리포트
    print(f"\n  Train 분포:")
    train_dist = Counter(item["category"] for item in train_data)
    for topic in topics:
        print(f"    {topic}: {train_dist.get(topic, 0)}건")

    print(f"\n  저장: {out_dir}/")
    print("  파일: train.json, val.json, test.json, category_mapping.json")


if __name__ == "__main__":
    main()
