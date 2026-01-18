"""분류 방식 비교 스크립트

기존 방식(벡터 유사도 + LLM fallback) vs 새 방식(Fine-tuned KcBERT)을 비교합니다.
"""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import Counter

# 기존 분류기
from ..classifier.vector_classifier import VectorContentClassifier

# 새 분류기
from .finetuned_classifier import FinetunedClassifier


# Claude API 비용 (Sonnet 기준, 2024)
CLAUDE_INPUT_COST_PER_1K = 0.003  # $0.003/1K input tokens
CLAUDE_OUTPUT_COST_PER_1K = 0.015  # $0.015/1K output tokens
AVG_INPUT_TOKENS_PER_CALL = 800  # 프롬프트 + 콘텐츠
AVG_OUTPUT_TOKENS_PER_CALL = 100  # 분류 결과


def load_raw_data(data_file: Path) -> tuple:
    """원본 데이터 로드 (분류 결과 제외)"""
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    letters = data.get("letters", [])
    posts = data.get("posts", [])

    # 분류 결과 저장 (ground truth로 사용)
    ground_truth = {}
    for item in letters + posts:
        item_id = item.get("_id")
        if item_id and "classification" in item:
            ground_truth[item_id] = item["classification"]["category"]

    # 분류 결과 제거 (재분류를 위해)
    for item in letters:
        item.pop("classification", None)
    for item in posts:
        item.pop("classification", None)

    return letters, posts, ground_truth


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


def main():
    project_root = Path(__file__).parent.parent.parent

    # 가장 최근 데이터 파일
    data_file = project_root / "data" / "classified_data" / "2026-01-05.json"
    output_path = project_root / "reports" / "classifier_comparison_report.md"

    print("=" * 60)
    print("분류 방식 비교")
    print("=" * 60)

    # 데이터 로드
    print(f"\n데이터 로드: {data_file}")
    letters, posts, ground_truth = load_raw_data(data_file)
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

    print("\n완료!")


if __name__ == "__main__":
    main()
