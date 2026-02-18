#!/usr/bin/env python3
"""라벨링용 데이터 추출 스크립트

기존 분류 데이터에서 라벨링용 샘플을 추출합니다.

사용법:
    python scripts/export_for_labeling.py [--sample-size 750] [--output OUTPUT_PATH]
"""
import json
import argparse
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict

# 기존 카테고리 → 새 카테고리 매핑 (참고용)
OLD_TO_NEW_MAPPING = {
    "감사·후기": "긍정 피드백",
    "질문·토론": "질문/문의",
    "정보성 글": "정보 공유",
    "서비스 피드백": "질문/문의",
    "불편사항": "부정 피드백",
    "일상·공감": "일상 소통",
}

# 새로운 5개 카테고리
NEW_CATEGORIES = [
    "긍정 피드백",
    "부정 피드백",
    "질문/문의",
    "정보 공유",
    "일상 소통",
]


def load_all_items(data_dir: Path) -> List[Dict]:
    """모든 분류 데이터에서 항목 로드"""
    all_items = []

    for json_file in sorted(data_dir.glob("*.json")):
        print(f"  로딩: {json_file.name}")
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # 편지
        for item in data.get("letters", []):
            text = item.get("message", "").strip()
            old_cat = item.get("classification", {}).get("category", "")
            confidence = item.get("classification", {}).get("confidence", 0)
            if text:
                all_items.append({
                    "id": item.get("_id", ""),
                    "text": text,
                    "source": "letter",
                    "old_category": old_cat,
                    "confidence": confidence,
                    "suggested_label": OLD_TO_NEW_MAPPING.get(old_cat, ""),
                    "new_label": None,
                    "file": json_file.name,
                    "master_name": item.get("masterName", ""),
                })

        # 게시글
        for item in data.get("posts", []):
            text = (item.get("textBody") or item.get("body") or "").strip()
            old_cat = item.get("classification", {}).get("category", "")
            confidence = item.get("classification", {}).get("confidence", 0)
            if text:
                all_items.append({
                    "id": item.get("_id", ""),
                    "text": text,
                    "source": "post",
                    "old_category": old_cat,
                    "confidence": confidence,
                    "suggested_label": OLD_TO_NEW_MAPPING.get(old_cat, ""),
                    "new_label": None,
                    "file": json_file.name,
                    "master_name": item.get("masterName", ""),
                })

    return all_items


def stratified_sample(
    items: List[Dict],
    sample_size: int,
    seed: int = 42
) -> List[Dict]:
    """카테고리별로 균형있게 샘플링

    새 카테고리 기준으로 균형을 맞춤.
    """
    random.seed(seed)

    # 새 카테고리 기준으로 그룹화
    by_new_category = {cat: [] for cat in NEW_CATEGORIES}

    for item in items:
        suggested = item.get("suggested_label", "")
        if suggested in by_new_category:
            by_new_category[suggested].append(item)

    # 각 카테고리에서 균등 샘플링
    per_category = sample_size // len(NEW_CATEGORIES)
    sampled = []

    print(f"\n카테고리별 샘플링 (목표: 카테고리당 {per_category}건)")
    for cat in NEW_CATEGORIES:
        available = by_new_category[cat]
        random.shuffle(available)
        take = min(per_category, len(available))
        sampled.extend(available[:take])
        print(f"  {cat}: {len(available)}건 중 {take}건 샘플링")

    # 부족분 채우기 (낮은 신뢰도 우선)
    remaining = sample_size - len(sampled)
    if remaining > 0:
        used_ids = {item["id"] for item in sampled}
        extras = [item for item in items if item["id"] not in used_ids]
        # 낮은 신뢰도 순 정렬 (분류가 애매한 케이스 우선)
        extras.sort(key=lambda x: x.get("confidence", 0))
        sampled.extend(extras[:remaining])
        print(f"\n  추가 샘플링 (낮은 신뢰도): {remaining}건")

    random.shuffle(sampled)
    return sampled


def print_stats(items: List[Dict], title: str = "통계"):
    """통계 출력"""
    print(f"\n{title}:")
    print(f"  총 항목: {len(items)}건")

    # 출처별
    by_source = Counter(item["source"] for item in items)
    print(f"  출처별: 편지 {by_source.get('letter', 0)}건, 게시글 {by_source.get('post', 0)}건")

    # 기존 카테고리별
    print("\n  기존 카테고리 분포:")
    by_old = Counter(item["old_category"] for item in items)
    for cat, count in by_old.most_common():
        print(f"    {cat or '(미분류)'}: {count}건")

    # 새 카테고리 예상 분포
    print("\n  새 카테고리 예상 분포:")
    by_new = Counter(item["suggested_label"] for item in items)
    for cat, count in by_new.most_common():
        print(f"    {cat or '(미분류)'}: {count}건")


def main():
    parser = argparse.ArgumentParser(description="라벨링용 데이터 추출")
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=750,
        help="샘플 크기 (기본: 750)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="출력 파일 경로"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "classified_data"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "data" / "labeling" / "labeling_data.json"

    print("=" * 60)
    print("라벨링용 데이터 추출")
    print("=" * 60)
    print(f"데이터 디렉토리: {data_dir}")
    print(f"샘플 크기: {args.sample_size}")
    print(f"출력 경로: {output_path}")

    # 데이터 로드
    print("\n데이터 로드 중...")
    all_items = load_all_items(data_dir)
    print_stats(all_items, "전체 데이터 통계")

    # 샘플링
    print(f"\n샘플링 중 (시드: {args.seed})...")
    sampled = stratified_sample(all_items, args.sample_size, seed=args.seed)
    print_stats(sampled, "샘플 데이터 통계")

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {output_path}")
    print(f"총 {len(sampled)}건")

    print("\n다음 단계:")
    print(f"  python scripts/labeling_tool.py")


if __name__ == "__main__":
    main()
