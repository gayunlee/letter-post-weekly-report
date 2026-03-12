"""Gemma LoRA 학습용 데이터 변환

train_3class.json → Gemma 학습 포맷 (text + label)
- train/val split (90/10)
- Golden set leak check
- category_mapping.json 생성

사용법:
    python3 scripts/prepare_gemma_ft_data.py
    python3 scripts/prepare_gemma_ft_data.py --val-ratio 0.1
"""
import json
import argparse
import random
from pathlib import Path
from collections import Counter


LABEL_MAP = {
    "대응 필요": 0,
    "콘텐츠·투자": 1,
    "노이즈": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def main():
    parser = argparse.ArgumentParser(description="Gemma 학습 데이터 준비")
    parser.add_argument("--input", default=None, help="입력 파일 (기본: train_3class.json)")
    parser.add_argument("--output-dir", default=None, help="출력 디렉토리")
    parser.add_argument("--golden-dir", default=None, help="Golden set 디렉토리")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_path = Path(args.input) if args.input else project_root / "data" / "training_data" / "v3" / "train_3class.json"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "training_data" / "v3" / "gemma_3class"
    golden_dir = Path(args.golden_dir) if args.golden_dir else project_root / "data" / "gold_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Gemma 학습 데이터 준비")
    print("=" * 60)

    # 입력 데이터 로드
    with open(input_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"\n  입력: {input_path.name} ({len(raw_data)}건)")

    # Golden set 로드 (leak check용)
    golden_texts = set()
    golden_ids = set()
    if golden_dir.exists():
        for gf in sorted(golden_dir.glob("*_golden_set.json")):
            with open(gf, encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                golden_texts.add(item["text"].strip())
                if "_id" in item:
                    golden_ids.add(item["_id"])
            print(f"    Golden: {gf.name} ({len(items)}건)")
    print(f"  Golden set 총: {len(golden_texts)}건 (텍스트), {len(golden_ids)}건 (ID)")

    # Leak check
    leaks = []
    for item in raw_data:
        if item["text"].strip() in golden_texts or item.get("_id") in golden_ids:
            leaks.append(item)

    if leaks:
        print(f"\n  ⚠ Golden set 누수 {len(leaks)}건 발견 — 제거합니다.")
        leak_ids = {item.get("_id") for item in leaks}
        leak_texts = {item["text"].strip() for item in leaks}
        raw_data = [
            item for item in raw_data
            if item.get("_id") not in leak_ids and item["text"].strip() not in leak_texts
        ]
        print(f"  제거 후: {len(raw_data)}건")
    else:
        print(f"  Golden set 누수 검사: 통과 (0건)")

    # v4_3class → label 변환
    converted = []
    skipped = 0
    for item in raw_data:
        v4 = item.get("v4_3class", "")
        if v4 not in LABEL_MAP:
            skipped += 1
            continue
        converted.append({
            "text": item["text"],
            "label": LABEL_MAP[v4],
        })

    if skipped:
        print(f"  v4_3class 누락/미매칭: {skipped}건 건너뜀")

    # 분포 확인
    label_dist = Counter(item["label"] for item in converted)
    print(f"\n  전체 분포 ({len(converted)}건):")
    for lid in sorted(label_dist.keys()):
        count = label_dist[lid]
        pct = count / len(converted) * 100
        print(f"    {lid} ({ID_TO_LABEL[lid]}): {count}건 ({pct:.1f}%)")

    # train/val split
    random.seed(args.seed)
    random.shuffle(converted)
    val_size = int(len(converted) * args.val_ratio)
    val_data = converted[:val_size]
    train_data = converted[val_size:]

    # train/val 분포
    for name, dataset in [("Train", train_data), ("Val", val_data)]:
        dist = Counter(item["label"] for item in dataset)
        print(f"\n  {name}: {len(dataset)}건")
        for lid in sorted(dist.keys()):
            print(f"    {lid} ({ID_TO_LABEL[lid]}): {dist[lid]}건")

    # 저장
    with open(output_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(output_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    # category_mapping.json
    mapping = {
        "category_to_id": LABEL_MAP,
        "id_to_category": {str(v): k for k, v in LABEL_MAP.items()},
        "num_labels": len(LABEL_MAP),
    }
    with open(output_dir / "category_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n  저장 완료:")
    print(f"    {output_dir / 'train.json'} ({len(train_data)}건)")
    print(f"    {output_dir / 'val.json'} ({len(val_data)}건)")
    print(f"    {output_dir / 'category_mapping.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
