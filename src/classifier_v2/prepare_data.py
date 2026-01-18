"""훈련 데이터 준비 스크립트

기존 분류된 데이터를 읽어서 train/val/test로 분할합니다.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import random


# 카테고리 매핑 (라벨 ID)
CATEGORY_TO_ID = {
    "감사·후기": 0,
    "질문·토론": 1,
    "정보성 글": 2,
    "서비스 피드백": 3,
    "불편사항": 4,
    "일상·공감": 5,
}

ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}


def load_classified_data(data_dir: Path) -> List[Dict]:
    """분류된 데이터 로드"""
    all_data = []

    for f in sorted(data_dir.glob("*.json")):
        with open(f, encoding="utf-8") as file:
            data = json.load(file)

            # 편지 데이터
            for item in data.get("letters", []):
                text = item.get("message", "").strip()
                category = item.get("classification", {}).get("category")
                if text and category in CATEGORY_TO_ID:
                    all_data.append({
                        "text": text,
                        "category": category,
                        "label": CATEGORY_TO_ID[category],
                        "source": "letter",
                        "file": f.name
                    })

            # 게시글 데이터
            for item in data.get("posts", []):
                # 게시글은 textBody 또는 body 필드 사용
                text = item.get("textBody") or item.get("body") or ""
                text = text.strip()
                category = item.get("classification", {}).get("category")
                if text and category in CATEGORY_TO_ID:
                    all_data.append({
                        "text": text,
                        "category": category,
                        "label": CATEGORY_TO_ID[category],
                        "source": "post",
                        "file": f.name
                    })

    return all_data


def stratified_split(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """카테고리 비율을 유지하면서 데이터 분할"""
    random.seed(seed)

    # 카테고리별로 그룹화
    by_category = {}
    for item in data:
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)

    train, val, test = [], [], []

    for cat, items in by_category.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # 섞기
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_splits(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    output_dir: Path
):
    """분할된 데이터 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        output_path = output_dir / f"{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(data)}건 → {output_path}")


def print_distribution(data: List[Dict], name: str):
    """카테고리 분포 출력"""
    counts = Counter(item["category"] for item in data)
    print(f"\n{name} 분포:")
    for cat, count in counts.most_common():
        pct = count / len(data) * 100
        print(f"  {cat}: {count}건 ({pct:.1f}%)")


def main():
    # 경로 설정
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "classified_data"
    output_dir = project_root / "data" / "training_data"

    print("=" * 50)
    print("훈련 데이터 준비")
    print("=" * 50)

    # 데이터 로드
    print(f"\n데이터 로드 중: {data_dir}")
    all_data = load_classified_data(data_dir)
    print(f"총 {len(all_data)}건 로드됨")

    print_distribution(all_data, "전체 데이터")

    # 데이터 분할
    print("\n데이터 분할 중 (80/10/10)...")
    train, val, test = stratified_split(all_data)

    print(f"\n분할 결과:")
    print(f"  Train: {len(train)}건")
    print(f"  Val: {len(val)}건")
    print(f"  Test: {len(test)}건")

    print_distribution(train, "Train")

    # 저장
    print(f"\n저장 중: {output_dir}")
    save_splits(train, val, test, output_dir)

    print("\n완료!")


if __name__ == "__main__":
    main()
