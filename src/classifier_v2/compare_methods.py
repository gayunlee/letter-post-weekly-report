"""분류 방식 비교 스크립트

기존 방식(벡터 유사도 + LLM fallback) vs 새 방식(Fine-tuned KcBERT)을 비교합니다.

사용법:
    # Ground truth 없이 일치율만 비교
    python -m src.classifier_v2.compare_methods

    # Ground truth 대비 정확도 평가 (라벨링 데이터 사용)
    python -m src.classifier_v2.compare_methods --with-ground-truth
"""
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

# 기존 분류기
from ..classifier.vector_classifier import VectorContentClassifier

# 새 분류기
from .finetuned_classifier import FinetunedClassifier

# 새 카테고리 정의
NEW_CATEGORIES = [
    "긍정 피드백",
    "부정 피드백",
    "질문/문의",
    "정보 공유",
    "일상 소통",
]

# 기존 → 새 카테고리 매핑
OLD_TO_NEW_MAPPING = {
    "감사·후기": "긍정 피드백",
    "질문·토론": "질문/문의",
    "정보성 글": "정보 공유",
    "서비스 피드백": "질문/문의",
    "불편사항": "부정 피드백",
    "일상·공감": "일상 소통",
}

# Claude API 비용 (Sonnet 기준, 2024)
CLAUDE_INPUT_COST_PER_1K = 0.003  # $0.003/1K input tokens
CLAUDE_OUTPUT_COST_PER_1K = 0.015  # $0.015/1K output tokens
AVG_INPUT_TOKENS_PER_CALL = 800  # 프롬프트 + 콘텐츠
AVG_OUTPUT_TOKENS_PER_CALL = 100  # 분류 결과


def load_raw_data(data_file: Path) -> tuple:
    """원본 데이터 로드 (기존 분류 결과 유지)"""
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    letters = data.get("letters", [])
    posts = data.get("posts", [])

    # 기존 분류 결과를 그대로 유지 (재분류 안함)
    return letters, posts


def load_ground_truth(labeling_file: Path) -> Dict[str, str]:
    """라벨링 데이터에서 Ground Truth 로드

    Returns:
        Dict[id, category]: ID별 정답 카테고리
    """
    if not labeling_file.exists():
        return {}

    with open(labeling_file, encoding="utf-8") as f:
        data = json.load(f)

    ground_truth = {}
    for item in data:
        item_id = item.get("id")
        label = item.get("new_label")
        if item_id and label:
            ground_truth[item_id] = label

    return ground_truth


def map_to_new_category(old_category: str) -> str:
    """기존 카테고리를 새 카테고리로 매핑"""
    return OLD_TO_NEW_MAPPING.get(old_category, old_category)


def run_vector_classifier(
    letters: List[Dict],
    posts: List[Dict],
    use_llm_fallback: bool = True
) -> tuple:
    """기존 벡터 분류기 실행"""
    classifier = VectorContentClassifier(
        use_llm_fallback=use_llm_fallback,
        confidence_threshold=0.3
    )

    start_time = time.time()

    # 편지 분류
    classified_letters = classifier.classify_batch(letters, content_field="message")

    # 게시글 분류 (textBody 필드 사용)
    classified_posts = classifier.classify_batch(posts, content_field="textBody")

    elapsed_time = time.time() - start_time
    llm_calls = classifier.llm_fallback_count

    return classified_letters, classified_posts, elapsed_time, llm_calls


def run_finetuned_classifier(
    letters: List[Dict],
    posts: List[Dict]
) -> tuple:
    """Fine-tuned 분류기 실행"""
    classifier = FinetunedClassifier()

    start_time = time.time()

    # 편지 분류
    classified_letters = classifier.classify_batch(letters, content_field="message")

    # 게시글 분류
    classified_posts = classifier.classify_batch(posts, content_field="textBody")

    elapsed_time = time.time() - start_time

    return classified_letters, classified_posts, elapsed_time


def calculate_agreement(
    results1: List[Dict],
    results2: List[Dict]
) -> Dict[str, Any]:
    """두 분류 결과의 일치율 계산"""
    agreements = 0
    disagreements = []

    for r1, r2 in zip(results1, results2):
        cat1 = r1.get("classification", {}).get("category")
        cat2 = r2.get("classification", {}).get("category")

        if cat1 == cat2:
            agreements += 1
        else:
            text = r1.get("message") or r1.get("textBody") or r1.get("body") or ""
            disagreements.append({
                "text": text[:100],
                "vector": cat1,
                "finetuned": cat2
            })

    total = len(results1)
    agreement_rate = agreements / total if total > 0 else 0

    return {
        "total": total,
        "agreements": agreements,
        "agreement_rate": agreement_rate,
        "disagreements_sample": disagreements[:10]  # 샘플 10개
    }


def calculate_accuracy(
    results: List[Dict],
    ground_truth: Dict[str, str]
) -> Dict[str, Any]:
    """Ground truth 대비 정확도 계산"""
    correct = 0
    total = 0

    category_stats = {}

    for item in results:
        item_id = item.get("_id")
        predicted = item.get("classification", {}).get("category")

        if item_id in ground_truth:
            actual = ground_truth[item_id]
            total += 1

            if actual not in category_stats:
                category_stats[actual] = {"correct": 0, "total": 0}
            category_stats[actual]["total"] += 1

            if predicted == actual:
                correct += 1
                category_stats[actual]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    # 카테고리별 정확도
    category_accuracy = {}
    for cat, stats in category_stats.items():
        category_accuracy[cat] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "category_accuracy": category_accuracy
    }


def estimate_cost(llm_calls: int) -> float:
    """LLM API 비용 추정"""
    input_cost = (AVG_INPUT_TOKENS_PER_CALL * llm_calls / 1000) * CLAUDE_INPUT_COST_PER_1K
    output_cost = (AVG_OUTPUT_TOKENS_PER_CALL * llm_calls / 1000) * CLAUDE_OUTPUT_COST_PER_1K
    return input_cost + output_cost


def generate_report(
    vector_results: Dict,
    finetuned_results: Dict,
    agreement: Dict,
    output_path: Path
):
    """비교 리포트 생성"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# 분류 방식 비교 리포트

생성일시: {now}

## 1. 개요

| 항목 | 기존 방식 (Vector + LLM) | 새 방식 (Fine-tuned KcBERT) |
|------|--------------------------|----------------------------|
| 처리 건수 | {vector_results['total']:,}건 | {finetuned_results['total']:,}건 |
| 총 처리 시간 | {vector_results['elapsed_time']:.2f}초 | {finetuned_results['elapsed_time']:.2f}초 |
| 건당 평균 시간 | {vector_results['avg_time_per_item']*1000:.2f}ms | {finetuned_results['avg_time_per_item']*1000:.2f}ms |
| LLM API 호출 | {vector_results['llm_calls']:,}회 | 0회 |
| 예상 API 비용 | ${vector_results['estimated_cost']:.4f} | $0.0000 |

## 2. 성능 비교

### 속도
- **기존 방식**: {vector_results['elapsed_time']:.2f}초 ({vector_results['avg_time_per_item']*1000:.2f}ms/건)
- **새 방식**: {finetuned_results['elapsed_time']:.2f}초 ({finetuned_results['avg_time_per_item']*1000:.2f}ms/건)
- **속도 향상**: {vector_results['elapsed_time'] / finetuned_results['elapsed_time']:.1f}배 빠름

### 비용
- **기존 방식**: LLM fallback {vector_results['llm_calls']}회 → 예상 비용 ${vector_results['estimated_cost']:.4f}
- **새 방식**: 로컬 추론 → 비용 $0

## 3. 분류 결과 비교

### 일치율
- **총 비교 건수**: {agreement['total']:,}건
- **일치 건수**: {agreement['agreements']:,}건
- **일치율**: {agreement['agreement_rate']*100:.1f}%

### 불일치 샘플 (상위 10건)

| 텍스트 (100자) | 기존 분류 | 새 분류 |
|---------------|----------|--------|
"""

    for d in agreement['disagreements_sample']:
        text = d['text'].replace('\n', ' ').replace('|', '\\|')
        report += f"| {text} | {d['vector']} | {d['finetuned']} |\n"

    report += f"""
## 4. 카테고리별 분포

### 기존 방식
"""
    for cat, count in sorted(vector_results['category_dist'].items(), key=lambda x: -x[1]):
        pct = count / vector_results['total'] * 100
        report += f"- {cat}: {count:,}건 ({pct:.1f}%)\n"

    report += f"""
### 새 방식
"""
    for cat, count in sorted(finetuned_results['category_dist'].items(), key=lambda x: -x[1]):
        pct = count / finetuned_results['total'] * 100
        report += f"- {cat}: {count:,}건 ({pct:.1f}%)\n"

    report += f"""
## 5. 결론

| 평가 항목 | 우위 |
|----------|------|
| 처리 속도 | {'새 방식' if finetuned_results['elapsed_time'] < vector_results['elapsed_time'] else '기존 방식'} |
| 비용 효율 | 새 방식 (무료) |
| 일치율 | {agreement['agreement_rate']*100:.1f}% |

### 권장사항
"""
    if agreement['agreement_rate'] >= 0.9:
        report += "- 일치율 90% 이상으로 Fine-tuned 모델로 전환 권장\n"
    elif agreement['agreement_rate'] >= 0.8:
        report += "- 일치율 80% 이상으로 추가 훈련 데이터로 개선 후 전환 검토\n"
    else:
        report += "- 일치율이 낮아 훈련 데이터 검토 및 추가 수집 필요\n"

    if finetuned_results['elapsed_time'] < vector_results['elapsed_time']:
        speedup = vector_results['elapsed_time'] / finetuned_results['elapsed_time']
        report += f"- 처리 속도 {speedup:.1f}배 향상으로 대량 처리에 유리\n"

    report += f"- 예상 비용 절감: 월 1만건 처리 시 ${estimate_cost(1000):.2f} 절감\n"

    # 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n리포트 저장: {output_path}")


def generate_accuracy_report(
    accuracy_results: Dict[str, Dict],
    output_path: Path
):
    """Ground Truth 대비 정확도 리포트 생성"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Ground Truth 대비 정확도 리포트

생성일시: {now}

## 1. 전체 정확도

| 분류기 | 정확도 | 정답 | 전체 |
|--------|--------|------|------|
"""
    for name, result in accuracy_results.items():
        acc = result["accuracy"] * 100
        report += f"| {name} | {acc:.1f}% | {result['correct']} | {result['total']} |\n"

    report += """
## 2. 카테고리별 정확도

| 카테고리 | 기존 방식 | Fine-tuned |
|----------|----------|------------|
"""
    # 모든 카테고리 수집
    all_cats = set()
    for result in accuracy_results.values():
        all_cats.update(result.get("category_accuracy", {}).keys())

    for cat in sorted(all_cats):
        vector_acc = accuracy_results.get("기존 방식", {}).get("category_accuracy", {}).get(cat, 0) * 100
        finetuned_acc = accuracy_results.get("Fine-tuned", {}).get("category_accuracy", {}).get(cat, 0) * 100
        report += f"| {cat} | {vector_acc:.1f}% | {finetuned_acc:.1f}% |\n"

    report += """
## 3. 권장사항

"""
    best_method = max(accuracy_results.items(), key=lambda x: x[1]["accuracy"])
    report += f"- **최고 정확도**: {best_method[0]} ({best_method[1]['accuracy']*100:.1f}%)\n"

    if best_method[1]["accuracy"] >= 0.7:
        report += "- 목표 정확도 70% 달성\n"
    else:
        report += "- 목표 정확도 70% 미달, 추가 라벨링 및 재훈련 필요\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n정확도 리포트 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="분류 방식 비교")
    parser.add_argument(
        "--with-ground-truth",
        action="store_true",
        help="라벨링 데이터를 Ground Truth로 사용하여 정확도 평가"
    )
    parser.add_argument(
        "--labeling-file",
        type=str,
        default=None,
        help="라벨링 데이터 파일 경로"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    # 가장 최근 데이터 파일
    data_file = project_root / "data" / "classified_data" / "2026-01-05.json"
    output_path = project_root / "reports" / "classifier_comparison_report.md"

    print("=" * 60)
    print("분류 방식 비교")
    print("=" * 60)

    # Ground Truth 로드 (선택적)
    ground_truth = {}
    if args.with_ground_truth:
        labeling_file = Path(args.labeling_file) if args.labeling_file else project_root / "data" / "labeling" / "labeling_data.json"
        print(f"\nGround Truth 로드: {labeling_file}")
        ground_truth = load_ground_truth(labeling_file)
        print(f"  라벨링된 항목: {len(ground_truth)}건")

    # 데이터 로드
    print(f"\n데이터 로드: {data_file}")
    letters, posts = load_raw_data(data_file)
    total_items = len(letters) + len(posts)
    print(f"  편지: {len(letters)}건, 게시글: {len(posts)}건, 총: {total_items}건")

    # 기존 방식 실행
    print("\n[1/2] 기존 방식 (Vector + LLM fallback) 실행 중...")
    v_letters, v_posts, v_time, v_llm_calls = run_vector_classifier(
        [l.copy() for l in letters],
        [p.copy() for p in posts]
    )

    vector_results = {
        "total": total_items,
        "elapsed_time": v_time,
        "avg_time_per_item": v_time / total_items,
        "llm_calls": v_llm_calls,
        "estimated_cost": estimate_cost(v_llm_calls),
        "category_dist": Counter(
            item.get("classification", {}).get("category", "미분류")
            for item in v_letters + v_posts
        )
    }
    print(f"  완료: {v_time:.2f}초, LLM 호출: {v_llm_calls}회")

    # 새 방식 실행
    print("\n[2/2] 새 방식 (Fine-tuned KcBERT) 실행 중...")
    f_letters, f_posts, f_time = run_finetuned_classifier(
        [l.copy() for l in letters],
        [p.copy() for p in posts]
    )

    finetuned_results = {
        "total": total_items,
        "elapsed_time": f_time,
        "avg_time_per_item": f_time / total_items,
        "category_dist": Counter(
            item.get("classification", {}).get("category", "미분류")
            for item in f_letters + f_posts
        )
    }
    print(f"  완료: {f_time:.2f}초")

    # 일치율 계산
    print("\n결과 비교 중...")
    agreement = calculate_agreement(v_letters + v_posts, f_letters + f_posts)
    print(f"  일치율: {agreement['agreement_rate']*100:.1f}%")

    # 리포트 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_report(vector_results, finetuned_results, agreement, output_path)

    # Ground Truth 대비 정확도 평가
    if ground_truth:
        print("\nGround Truth 대비 정확도 계산 중...")

        # 기존 방식 정확도 (새 카테고리로 매핑 후 비교)
        vector_mapped = []
        for item in v_letters + v_posts:
            mapped_item = item.copy()
            old_cat = item.get("classification", {}).get("category", "")
            mapped_item["classification"] = {
                "category": map_to_new_category(old_cat)
            }
            vector_mapped.append(mapped_item)

        vector_accuracy = calculate_accuracy(vector_mapped, ground_truth)
        print(f"  기존 방식 정확도: {vector_accuracy['accuracy']*100:.1f}%")

        # Fine-tuned 정확도
        finetuned_accuracy = calculate_accuracy(f_letters + f_posts, ground_truth)
        print(f"  Fine-tuned 정확도: {finetuned_accuracy['accuracy']*100:.1f}%")

        # 정확도 리포트 생성
        accuracy_output = project_root / "reports" / "accuracy_report.md"
        generate_accuracy_report(
            {
                "기존 방식": vector_accuracy,
                "Fine-tuned": finetuned_accuracy
            },
            accuracy_output
        )

    print("\n완료!")


if __name__ == "__main__":
    main()
