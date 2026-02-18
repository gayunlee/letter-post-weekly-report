#!/usr/bin/env python3
"""Few-shot vs Fine-tuned 결과 비교 및 리포트 생성

사용법:
    python scripts/compare_results.py
"""
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Optional

# 5개 카테고리
CATEGORIES = ["서비스 이슈", "서비스 칭찬", "투자 질문", "정보/의견", "일상 소통"]


def load_results(path: Path) -> Optional[dict]:
    """결과 파일 로드"""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_confusion_matrix(results: list, categories: list) -> dict:
    """혼동 행렬 계산"""
    matrix = {cat: {c: 0 for c in categories} for cat in categories}
    for item in results:
        expected = item["expected"]
        predicted = item["predicted"]
        if expected in categories and predicted in categories:
            matrix[expected][predicted] += 1
    return matrix


def compute_f1_scores(results: list, categories: list) -> dict:
    """카테고리별 F1 스코어 계산"""
    scores = {}

    for cat in categories:
        # True Positives, False Positives, False Negatives
        tp = sum(1 for r in results if r["expected"] == cat and r["predicted"] == cat)
        fp = sum(1 for r in results if r["expected"] != cat and r["predicted"] == cat)
        fn = sum(1 for r in results if r["expected"] == cat and r["predicted"] != cat)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        scores[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn
        }

    return scores


def format_confusion_matrix(matrix: dict, categories: list) -> str:
    """혼동 행렬 포맷팅"""
    # 헤더
    header = "실제\\예측  | " + " | ".join(f"{cat[:4]:>6}" for cat in categories)
    lines = [header, "-" * len(header)]

    for cat in categories:
        row = f"{cat[:6]:>10} | "
        row += " | ".join(f"{matrix[cat][c]:>6}" for c in categories)
        lines.append(row)

    return "\n".join(lines)


def analyze_disagreements(fewshot_results: list, finetuned_results: list) -> list:
    """두 방식이 다르게 분류한 케이스 분석"""
    disagreements = []

    fewshot_map = {r["id"]: r for r in fewshot_results}
    finetuned_map = {r["id"]: r for r in finetuned_results}

    for id_val in fewshot_map:
        if id_val not in finetuned_map:
            continue

        few = fewshot_map[id_val]
        fine = finetuned_map[id_val]

        if few["predicted"] != fine["predicted"]:
            disagreements.append({
                "id": id_val,
                "text": few["text"],
                "expected": few["expected"],
                "fewshot_pred": few["predicted"],
                "finetuned_pred": fine["predicted"],
                "fewshot_correct": few["correct"],
                "finetuned_correct": fine["correct"]
            })

    return disagreements


def generate_report(fewshot: dict, finetuned: dict) -> str:
    """비교 리포트 생성"""
    report = []
    report.append("=" * 70)
    report.append("VOC 분류: Few-shot vs Fine-tuning 비교 리포트")
    report.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)

    # 1. 요약
    report.append("\n## 1. 전체 요약\n")
    report.append("| 항목 | Few-shot | Fine-tuned | 차이 |")
    report.append("|------|----------|------------|------|")

    few_main = fewshot["summary"]
    fine_main = finetuned["summary"]

    main_diff = fine_main["main_accuracy"] - few_main["main_accuracy"]
    domain_diff = fine_main["domain_accuracy"] - few_main["domain_accuracy"]

    report.append(f"| 메인 테스트 정확도 | {few_main['main_accuracy']:.1f}% | {fine_main['main_accuracy']:.1f}% | {main_diff:+.1f}% |")
    report.append(f"| 도메인 테스트 정확도 | {few_main['domain_accuracy']:.1f}% | {fine_main['domain_accuracy']:.1f}% | {domain_diff:+.1f}% |")

    # 2. 카테고리별 F1 스코어
    report.append("\n## 2. 카테고리별 F1 스코어\n")

    few_f1 = compute_f1_scores(fewshot["main_test"], CATEGORIES)
    fine_f1 = compute_f1_scores(finetuned["main_test"], CATEGORIES)

    report.append("| 카테고리 | Few-shot F1 | Fine-tuned F1 | 차이 |")
    report.append("|----------|-------------|---------------|------|")

    for cat in CATEGORIES:
        few_score = few_f1[cat]["f1"]
        fine_score = fine_f1[cat]["f1"]
        diff = fine_score - few_score
        report.append(f"| {cat} | {few_score:.3f} | {fine_score:.3f} | {diff:+.3f} |")

    # 3. 혼동 행렬
    report.append("\n## 3. 혼동 행렬\n")

    report.append("### Few-shot")
    few_cm = compute_confusion_matrix(fewshot["main_test"], CATEGORIES)
    report.append("```")
    report.append(format_confusion_matrix(few_cm, CATEGORIES))
    report.append("```")

    report.append("\n### Fine-tuned")
    fine_cm = compute_confusion_matrix(finetuned["main_test"], CATEGORIES)
    report.append("```")
    report.append(format_confusion_matrix(fine_cm, CATEGORIES))
    report.append("```")

    # 4. 도메인 이해도 테스트 상세
    report.append("\n## 4. 도메인 이해도 테스트 상세\n")

    report.append("| 텍스트 | 정답 | Few-shot | Fine-tuned |")
    report.append("|--------|------|----------|------------|")

    for few_item, fine_item in zip(fewshot["domain_test"], finetuned["domain_test"]):
        text = few_item["text"][:30] + "..." if len(few_item["text"]) > 30 else few_item["text"]
        expected = few_item["expected"]
        few_pred = few_item["predicted"]
        fine_pred = fine_item["predicted"]

        few_mark = "✓" if few_item["correct"] else "✗"
        fine_mark = "✓" if fine_item["correct"] else "✗"

        report.append(f"| {text} | {expected} | {few_mark} {few_pred} | {fine_mark} {fine_pred} |")

    # 5. 불일치 분석
    report.append("\n## 5. 두 방식이 다르게 분류한 케이스\n")

    disagreements = analyze_disagreements(fewshot["main_test"], finetuned["main_test"])
    report.append(f"총 {len(disagreements)}건의 불일치\n")

    # 불일치 유형 분석
    patterns = Counter()
    for d in disagreements:
        key = f"{d['fewshot_pred']} vs {d['finetuned_pred']}"
        patterns[key] += 1

    report.append("### 불일치 패턴")
    report.append("| Few-shot | Fine-tuned | 건수 |")
    report.append("|----------|------------|------|")
    for pattern, count in patterns.most_common(10):
        parts = pattern.split(" vs ")
        report.append(f"| {parts[0]} | {parts[1]} | {count} |")

    # 누가 더 맞았는지
    few_better = sum(1 for d in disagreements if d["fewshot_correct"] and not d["finetuned_correct"])
    fine_better = sum(1 for d in disagreements if d["finetuned_correct"] and not d["fewshot_correct"])
    both_wrong = sum(1 for d in disagreements if not d["fewshot_correct"] and not d["finetuned_correct"])

    report.append(f"\n불일치 중:")
    report.append(f"  - Few-shot만 정답: {few_better}건")
    report.append(f"  - Fine-tuned만 정답: {fine_better}건")
    report.append(f"  - 둘 다 오답: {both_wrong}건")

    # 6. 비용 효율 분석
    report.append("\n## 6. 비용 효율 분석\n")
    report.append("| 항목 | Few-shot | Fine-tuned |")
    report.append("|------|----------|------------|")
    report.append("| 학습 비용 | $0 | ~$5.00 |")
    report.append("| 추론 비용 (100건) | ~$0.03 | ~$0.02 |")
    report.append("| 1000건 처리 총 비용 | ~$0.30 | ~$5.20 |")
    report.append("| 10000건 처리 총 비용 | ~$3.00 | ~$7.00 |")

    accuracy_gain = fine_main["main_accuracy"] - few_main["main_accuracy"]
    report.append(f"\n정확도 향상: {accuracy_gain:+.1f}%p")
    if accuracy_gain > 0:
        cost_per_point = 5.0 / accuracy_gain
        report.append(f"정확도 1%p 향상당 비용: ${cost_per_point:.2f}")

    # 7. 결론
    report.append("\n## 7. 결론\n")

    if fine_main["main_accuracy"] > few_main["main_accuracy"]:
        report.append(f"Fine-tuning이 Few-shot 대비 메인 테스트에서 {main_diff:.1f}%p 높은 정확도를 보임.")
    else:
        report.append(f"Few-shot이 Fine-tuning 대비 메인 테스트에서 {-main_diff:.1f}%p 높은 정확도를 보임.")

    if fine_main["domain_accuracy"] > few_main["domain_accuracy"]:
        report.append(f"도메인 이해도 테스트에서도 Fine-tuning이 {domain_diff:.1f}%p 높음.")
    else:
        report.append(f"도메인 이해도 테스트에서는 Few-shot이 {-domain_diff:.1f}%p 높음.")

    # 권장사항
    report.append("\n### 권장사항")
    if accuracy_gain >= 5:
        report.append("- 5%p 이상 개선: Fine-tuning 적용 권장")
    elif accuracy_gain >= 2:
        report.append("- 2-5%p 개선: 대량 처리 시 Fine-tuning 고려")
    else:
        report.append("- 2%p 미만 개선: Few-shot으로 충분, 추가 비용 불필요")

    return "\n".join(report)


def main():
    project_root = Path(__file__).parent.parent
    fewshot_path = project_root / "data" / "results" / "fewshot_results.json"
    finetuned_path = project_root / "data" / "results" / "finetuned_results.json"
    report_path = project_root / "data" / "results" / "comparison_report.md"

    print("=" * 60)
    print("Few-shot vs Fine-tuned 비교 분석")
    print("=" * 60)

    # 결과 로드
    fewshot = load_results(fewshot_path)
    finetuned = load_results(finetuned_path)

    if not fewshot:
        print(f"Few-shot 결과 없음: {fewshot_path}")
        print("먼저 실행: python scripts/classify_fewshot.py")
        return

    if not finetuned:
        print(f"Fine-tuned 결과 없음: {finetuned_path}")
        print("먼저 실행: python scripts/classify_finetuned.py")
        return

    # 기본 정보
    print(f"\nFew-shot 모델: {fewshot.get('model')}")
    print(f"Fine-tuned 모델: {finetuned.get('model')}")

    # 비교 요약
    print("\n[비교 요약]")
    print(f"메인 테스트 정확도:")
    print(f"  Few-shot:   {fewshot['summary']['main_accuracy']:.1f}%")
    print(f"  Fine-tuned: {finetuned['summary']['main_accuracy']:.1f}%")

    print(f"\n도메인 테스트 정확도:")
    print(f"  Few-shot:   {fewshot['summary']['domain_accuracy']:.1f}%")
    print(f"  Fine-tuned: {finetuned['summary']['domain_accuracy']:.1f}%")

    # 리포트 생성
    report = generate_report(fewshot, finetuned)
    print("\n" + report)

    # 저장
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
