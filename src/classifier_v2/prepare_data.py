"""훈련 데이터 준비 스크립트

라벨링된 데이터를 읽어서 train/val/test로 분할합니다.

사용법:
    # 새 카테고리 (5개) - 라벨링 데이터 사용
    python -m src.classifier_v2.prepare_data

    # 기존 카테고리 (6개) - 기존 분류 데이터 사용
    python -m src.classifier_v2.prepare_data --legacy
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import random


# 새로운 5개 카테고리 매핑 (라벨 ID)
CATEGORY_TO_ID_V2 = {
    "긍정 피드백": 0,
    "부정 피드백": 1,
    "질문/문의": 2,
    "정보 공유": 3,
    "일상 소통": 4,
}

ID_TO_CATEGORY_V2 = {v: k for k, v in CATEGORY_TO_ID_V2.items()}

# 기존 6개 카테고리 (레거시)
CATEGORY_TO_ID_LEGACY = {
    "감사·후기": 0,
    "질문·토론": 1,
    "정보성 글": 2,
    "서비스 피드백": 3,
    "불편사항": 4,
    "일상·공감": 5,
}

ID_TO_CATEGORY_LEGACY = {v: k for k, v in CATEGORY_TO_ID_LEGACY.items()}

# 기본값: 새 카테고리 사용
CATEGORY_TO_ID = CATEGORY_TO_ID_V2
ID_TO_CATEGORY = ID_TO_CATEGORY_V2


def load_labeling_data(labeling_file: Path) -> List[Dict]:
    """라벨링된 데이터 로드 (새 카테고리)

    Args:
        labeling_file: 라벨링 데이터 파일 경로
    """
    if not labeling_file.exists():
        raise FileNotFoundError(f"라벨링 파일을 찾을 수 없습니다: {labeling_file}")

    with open(labeling_file, encoding="utf-8") as f:
        raw_data = json.load(f)

    all_data = []
    skipped = 0

    for item in raw_data:
        text = item.get("text", "").strip()
        category = item.get("new_label")  # 새로 라벨링된 카테고리

        if not text:
            skipped += 1
            continue

        if category is None:
            skipped += 1
            continue

        if category not in CATEGORY_TO_ID_V2:
            print(f"  경고: 알 수 없는 카테고리 '{category}'")
            skipped += 1
            continue

        all_data.append({
            "text": text,
            "category": category,
            "label": CATEGORY_TO_ID_V2[category],
            "source": item.get("source", "unknown"),
            "id": item.get("id", ""),
        })

    if skipped > 0:
        print(f"  건너뛴 항목: {skipped}건 (미라벨링 또는 빈 텍스트)")

    return all_data


def load_classified_data(data_dir: Path, trusted_files: List[str] = None) -> List[Dict]:
    """분류된 데이터 로드 (기존 카테고리 - 레거시)

    Args:
        data_dir: 데이터 디렉토리
        trusted_files: 신뢰할 수 있는 파일명 리스트 (None이면 전체)
    """
    all_data = []

    for f in sorted(data_dir.glob("*.json")):
        # 신뢰 파일 필터링
        if trusted_files and f.name not in trusted_files:
            continue
        with open(f, encoding="utf-8") as file:
            data = json.load(file)

            # 편지 데이터
            for item in data.get("letters", []):
                text = item.get("message", "").strip()
                category = item.get("classification", {}).get("category")
                if text and category in CATEGORY_TO_ID_LEGACY:
                    all_data.append({
                        "text": text,
                        "category": category,
                        "label": CATEGORY_TO_ID_LEGACY[category],
                        "source": "letter",
                        "file": f.name
                    })

            # 게시글 데이터
            for item in data.get("posts", []):
                # 게시글은 textBody 또는 body 필드 사용
                text = item.get("textBody") or item.get("body") or ""
                text = text.strip()
                category = item.get("classification", {}).get("category")
                if text and category in CATEGORY_TO_ID_LEGACY:
                    all_data.append({
                        "text": text,
                        "category": category,
                        "label": CATEGORY_TO_ID_LEGACY[category],
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
    parser = argparse.ArgumentParser(description="훈련 데이터 준비")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="기존 6개 카테고리 사용 (레거시 모드)"
    )
    parser.add_argument(
        "--labeling-file",
        type=str,
        default=None,
        help="라벨링 데이터 파일 경로 (기본: data/labeling/labeling_data.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉토리 (기본: data/training_data_v2)"
    )
    args = parser.parse_args()

    # 경로 설정
    project_root = Path(__file__).parent.parent.parent

    if args.legacy:
        # 레거시 모드: 기존 분류 데이터 사용
        data_dir = project_root / "data" / "classified_data"
        output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "training_data"
        trusted_files = ["2026-01-05.json"]

        print("=" * 50)
        print("훈련 데이터 준비 (레거시 모드 - 6개 카테고리)")
        print("=" * 50)
        print(f"신뢰 파일: {trusted_files}")

        # 데이터 로드
        print(f"\n데이터 로드 중: {data_dir}")
        all_data = load_classified_data(data_dir, trusted_files=trusted_files)
    else:
        # 새 모드: 라벨링 데이터 사용
        labeling_file = Path(args.labeling_file) if args.labeling_file else project_root / "data" / "labeling" / "labeling_data.json"
        output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "training_data_v2"

        print("=" * 50)
        print("훈련 데이터 준비 (새 카테고리 - 5개)")
        print("=" * 50)
        print(f"라벨링 파일: {labeling_file}")

        # 데이터 로드
        print(f"\n데이터 로드 중...")
        all_data = load_labeling_data(labeling_file)

    print(f"총 {len(all_data)}건 로드됨")

    if len(all_data) == 0:
        print("\n오류: 로드된 데이터가 없습니다.")
        if not args.legacy:
            print("라벨링을 먼저 완료하세요: python scripts/labeling_tool.py")
        return

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

    # 카테고리 매핑 저장
    category_mapping = CATEGORY_TO_ID_LEGACY if args.legacy else CATEGORY_TO_ID_V2
    mapping_file = output_dir / "category_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump({
            "category_to_id": category_mapping,
            "id_to_category": {v: k for k, v in category_mapping.items()},
            "num_labels": len(category_mapping),
            "mode": "legacy" if args.legacy else "v2"
        }, f, ensure_ascii=False, indent=2)
    print(f"  카테고리 매핑 → {mapping_file}")

    print("\n완료!")


if __name__ == "__main__":
    main()
