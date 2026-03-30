"""v5 4분류 벤치마크 — V5Classifier + v7 golden set

목적: 4분류 체계(피드백/마스터 반응/시장·투자/일상) 정확도 측정.

비교 대상:
  - 78.9% (v3c 4분류 Haiku zero-shot)
  - 93.9% (v4 3분류 Haiku)

사용법:
    python3 scripts/benchmark_v5_4class_golden.py
"""
import sys, os, json, argparse, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier_v5.classifier import V5Classifier, V5_TOPICS

V5_4CLASS = V5_TOPICS


def main():
    parser = argparse.ArgumentParser(description="v5 4분류 벤치마크")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--golden-path", default="data/gold_dataset/v7_golden_set.json")
    args = parser.parse_args()

    with open(args.golden_path, encoding='utf-8') as f:
        golden = json.load(f)

    # v5_4class 라벨 있는 항목만
    golden = [item for item in golden if 'v5_4class' in item]

    print("=" * 60)
    print(f"  v5 4분류 벤치마크 — V5Classifier")
    print(f"  모델: {args.model}")
    print(f"  Golden set: {len(golden)}건")
    print("=" * 60)

    # 정답 분포
    y_true = [item['v5_4class'] for item in golden]
    true_dist = Counter(y_true)
    print(f"\n  정답 분포:")
    for t in V5_4CLASS:
        print(f"    {t}: {true_dist.get(t, 0)}건")

    # V5Classifier로 분류
    classifier = V5Classifier(model=args.model, max_workers=args.max_workers)

    results = [None] * len(golden)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {}
        for i, item in enumerate(golden):
            future = executor.submit(classifier.classify_single, item['text'])
            future_map[future] = i

        done_count = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
            done_count += 1
            if done_count % 50 == 0 or done_count == len(golden):
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"    {done_count}/{len(golden)} 완료 ({rate:.1f}건/초)")

    elapsed = time.time() - start_time
    print(f"\n  총 소요: {elapsed:.1f}초 ({len(golden)/elapsed:.1f}건/초)")

    # 에러 체크
    errors = [(i, r['error']) for i, r in enumerate(results) if 'error' in r]
    if errors:
        print(f"\n  에러: {len(errors)}건")
        for idx, err in errors[:5]:
            print(f"    [{idx}] {err}")

    # 4분류 정확도
    y_pred = [r['topic'] for r in results]
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(golden)
    print(f"\n  4분류 정확도: {acc*100:.1f}% ({correct}/{len(golden)})")

    # Actionable-only (일상 제외)
    actionable_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != "일상"]
    if actionable_pairs:
        correct_act = sum(1 for t, p in actionable_pairs if t == p)
        acc_act = correct_act / len(actionable_pairs)
        print(f"  Actionable-only: {acc_act*100:.1f}% ({correct_act}/{len(actionable_pairs)})")

    # Per-category metrics
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    report = classification_report(y_true, y_pred, labels=V5_4CLASS, digits=4, zero_division=0)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=V5_4CLASS)
    print("  Confusion Matrix:")
    header = f"  {'':>14}"
    for t in V5_4CLASS:
        header += f"  {t[:6]:>6}"
    print(header)
    for i, row in enumerate(cm):
        line = f"  {V5_4CLASS[i]:>14}"
        for val in row:
            line += f"  {val:>6}"
        print(line)

    # 오분류 상세
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                'index': i,
                'text': golden[i]['text'][:150],
                'true': true,
                'pred': pred,
                'confidence': results[i].get('confidence', 0),
                'summary': results[i].get('summary', ''),
                'sentiment': results[i].get('sentiment', ''),
                'tags': results[i].get('tags', []),
            })

    print(f"\n  오분류: {len(misclassified)}건")
    for m in misclassified:
        print(f"    [{m['index']}] {m['true']} → {m['pred']} | conf={m['confidence']:.2f} | {m['summary'][:60]}")

    # 오분류 패턴 분석
    error_patterns = Counter()
    for m in misclassified:
        error_patterns[f"{m['true']}→{m['pred']}"] += 1
    if error_patterns:
        print(f"\n  오분류 패턴:")
        for pattern, count in error_patterns.most_common():
            print(f"    {pattern}: {count}건")

    # 비교
    print(f"\n{'='*60}")
    print(f"  비교:")
    print(f"    v3c 4분류 Haiku zero-shot:    78.9%")
    print(f"    v4 3분류 Haiku:               93.9%")
    print(f"    v5 4분류 (이번):              {acc*100:.1f}%")
    print(f"{'='*60}")

    # 비용 리포트
    cost = classifier.get_cost_report()
    print(f"\n  비용: ${cost['cost_usd']:.4f} (in:{cost['input_tokens']:,} / out:{cost['output_tokens']:,})")

    # 결과 저장
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=V5_4CLASS, zero_division=0
    )
    per_cat = {}
    for i, t in enumerate(V5_4CLASS):
        per_cat[t] = {
            'precision': round(float(p[i]), 4),
            'recall': round(float(r[i]), 4),
            'f1': round(float(f1[i]), 4),
            'support': int(support[i]),
        }

    output = {
        'experiment': 'v5_4class_classifier',
        'model': args.model,
        'taxonomy': '4분류 (피드백 / 마스터 반응 / 시장·투자 / 일상)',
        'golden_set': os.path.basename(args.golden_path),
        'golden_set_size': len(golden),
        'accuracy_4class': round(acc, 4),
        'accuracy_actionable': round(acc_act, 4) if actionable_pairs else None,
        'correct': correct,
        'baseline_comparison': {
            'v3c_5way': 0.789,
            'v4_3class': 0.9391,
            'v5_4class': round(acc, 4),
        },
        'per_category': per_cat,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'misclassified': misclassified,
        'error_patterns': dict(error_patterns),
        'cost': cost,
        'elapsed_seconds': round(elapsed, 1),
        'errors': len(errors),
    }

    output_path = "benchmarks/golden_benchmark_v5_4class.json"
    os.makedirs('benchmarks', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output_path}")


if __name__ == "__main__":
    main()
