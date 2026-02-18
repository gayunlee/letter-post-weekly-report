"""2축 분류 체계용 데이터 준비 스크립트

labeled_2axis_reviewed.json을 읽어서:
1. Topic 분류기용 train/val/test split
2. Sentiment 분류기용 train/val/test split
3. 벡터 분류기용 샘플 선별

모든 데이터는 data/training_data/two_axis/ 아래에 저장 (1축 데이터와 분리)
"""
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple


# 2축 카테고리 매핑
TOPIC_TO_ID = {
    "콘텐츠 반응": 0,
    "투자 이야기": 1,
    "서비스 이슈": 2,
    "커뮤니티 소통": 3,
}
ID_TO_TOPIC = {v: k for k, v in TOPIC_TO_ID.items()}

SENTIMENT_TO_ID = {
    "긍정": 0,
    "부정": 1,
    "중립": 2,
}
ID_TO_SENTIMENT = {v: k for k, v in SENTIMENT_TO_ID.items()}


def load_reviewed_data(filepath: Path) -> List[Dict]:
    with open(filepath, encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    skipped = 0
    for item in raw:
        text = item.get("text", "").strip()
        topic = item.get("topic", "")
        sentiment = item.get("sentiment", "")

        if not text or topic not in TOPIC_TO_ID or sentiment not in SENTIMENT_TO_ID:
            skipped += 1
            continue

        data.append({
            "text": text,
            "topic": topic,
            "topic_label": TOPIC_TO_ID[topic],
            "sentiment": sentiment,
            "sentiment_label": SENTIMENT_TO_ID[sentiment],
            "combo": f"{topic}_{sentiment}",
        })

    if skipped:
        print(f"  건너뛴 항목: {skipped}건")
    return data


def stratified_split(
    data: List[Dict],
    group_key: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """group_key 기준으로 stratified split"""
    random.seed(seed)

    by_group = {}
    for item in data:
        g = item[group_key]
        by_group.setdefault(g, []).append(item)

    train, val, test = [], [], []
    for g, items in by_group.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        # 최소 1개씩 보장
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = max(1, n - n_train - 1)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def save_for_axis(data_list: List[Dict], axis: str, output_dir: Path):
    """특정 축(topic/sentiment)용 데이터 저장"""
    axis_dir = output_dir / axis
    axis_dir.mkdir(parents=True, exist_ok=True)

    label_key = f"{axis}_label"
    category_key = axis

    for name, items in data_list:
        # 해당 축 기준으로 재구성
        formatted = []
        for item in items:
            formatted.append({
                "text": item["text"],
                "category": item[category_key],
                "label": item[label_key],
            })

        filepath = axis_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
        print(f"  {axis}/{name}: {len(formatted)}건")

    # 카테고리 매핑 저장
    if axis == "topic":
        mapping = {"category_to_id": TOPIC_TO_ID, "id_to_category": ID_TO_TOPIC, "num_labels": len(TOPIC_TO_ID)}
    else:
        mapping = {"category_to_id": SENTIMENT_TO_ID, "id_to_category": ID_TO_SENTIMENT, "num_labels": len(SENTIMENT_TO_ID)}

    mapping_file = axis_dir / "category_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  {axis}/category_mapping.json 저장")


def select_vector_samples(data: List[Dict], per_combo: int = 15) -> List[Dict]:
    """벡터 분류기용 대표 샘플 선별 (Topic×Sentiment 조합별)"""
    by_combo = {}
    for item in data:
        combo = item["combo"]
        by_combo.setdefault(combo, []).append(item)

    samples = []
    print("\n벡터 분류기 샘플 선별:")
    for combo, items in sorted(by_combo.items()):
        # 텍스트 길이가 적당한 것 우선 (50~300자)
        good_length = [i for i in items if 50 <= len(i["text"]) <= 300]
        rest = [i for i in items if i not in good_length]

        pool = good_length + rest
        selected = pool[:per_combo]
        samples.extend(selected)
        print(f"  {combo}: {len(selected)}건 (전체 {len(items)}건 중)")

    return samples


def print_distribution(data: List[Dict], name: str):
    print(f"\n{name}:")
    topic_counts = Counter(item["topic"] for item in data)
    sentiment_counts = Counter(item["sentiment"] for item in data)

    print("  Topic:")
    for cat, cnt in topic_counts.most_common():
        print(f"    {cat}: {cnt}건 ({cnt/len(data)*100:.1f}%)")

    print("  Sentiment:")
    for cat, cnt in sentiment_counts.most_common():
        print(f"    {cat}: {cnt}건 ({cnt/len(data)*100:.1f}%)")


def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "training_data" / "labeled_2axis_reviewed.json"
    output_dir = project_root / "data" / "training_data" / "two_axis"

    print("=" * 60)
    print("2축 분류 체계 데이터 준비")
    print("=" * 60)

    # 데이터 로드
    print(f"\n데이터 로드: {input_file}")
    all_data = load_reviewed_data(input_file)
    print(f"총 {len(all_data)}건 로드")
    print_distribution(all_data, "전체 분포")

    # combo 기준으로 stratified split (두 축 동시 고려)
    print("\n\nStratified Split (Topic×Sentiment 조합 기준, 80/10/10)...")
    train, val, test = stratified_split(all_data, group_key="combo")

    print(f"\n분할 결과:")
    print(f"  Train: {len(train)}건")
    print(f"  Val: {len(val)}건")
    print(f"  Test: {len(test)}건")

    print_distribution(train, "Train 분포")
    print_distribution(val, "Val 분포")
    print_distribution(test, "Test 분포")

    # Topic 분류기용 저장
    print(f"\n\nTopic 분류기용 데이터 저장:")
    save_for_axis(
        [("train", train), ("val", val), ("test", test)],
        axis="topic",
        output_dir=output_dir
    )

    # Sentiment 분류기용 저장
    print(f"\nSentiment 분류기용 데이터 저장:")
    save_for_axis(
        [("train", train), ("val", val), ("test", test)],
        axis="sentiment",
        output_dir=output_dir
    )

    # 벡터 분류기용 샘플 선별
    samples = select_vector_samples(train, per_combo=15)
    samples_file = output_dir / "vector_samples.json"
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"text": s["text"], "topic": s["topic"], "sentiment": s["sentiment"]} for s in samples],
            f, ensure_ascii=False, indent=2
        )
    print(f"\n벡터 샘플 저장: {samples_file} ({len(samples)}건)")

    print("\n" + "=" * 60)
    print("완료!")
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
