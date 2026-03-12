"""v4 3분류 벤치마크 — UnifiedClassifier + v7 golden set

목적: 3분류 체계(대응 필요 / 콘텐츠·투자 / 노이즈)의 정확도 측정.
      UnifiedClassifier의 새 프롬프트(요약→분류 + 경계 규칙)로 93%+ 목표.

비교 대상:
  - 78.9% (v3c 5분류 Haiku zero-shot)
  - 84.1% (v4 R2 4분류 후처리)
  - 87.2% (R2→3분류 시뮬레이션)

사용법:
    python3 scripts/benchmark_v4_3class_golden.py
"""
import sys, os, json, argparse, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

from src.classifier_v4.unified_classifier import UnifiedClassifier, V4_TOPICS

V4_3CLASS = ["대응 필요", "콘텐츠·투자", "노이즈"]


def main():
    parser = argparse.ArgumentParser(description="v4 3분류 벤치마크")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--golden-path", default="data/gold_dataset/v7_golden_set.json")
    args = parser.parse_args()

    with open(args.golden_path, encoding='utf-8') as f:
        golden = json.load(f)

    print("=" * 60)
    print(f"  v4 3분류 벤치마크 — UnifiedClassifier")
    print(f"  모델: {args.model}")
    print(f"  Golden set: {len(golden)}건")
    print("=" * 60)

    # 정답 분포
    y_true = [item['v4_3class'] for item in golden]
    true_dist = Counter(y_true)
    print(f"\n  정답 분포:")
    for t in V4_3CLASS:
        print(f"    {t}: {true_dist.get(t, 0)}건")

    # UnifiedClassifier로 분류
    classifier = UnifiedClassifier(model=args.model, max_workers=args.max_workers)

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

    # 3분류 정확도
    y_pred = [r['topic'] for r in results]
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(golden)
    print(f"\n  3분류 정확도: {acc*100:.1f}% ({correct}/{len(golden)})")

    # Actionable-only (노이즈 제외)
    actionable_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != "노이즈"]
    if actionable_pairs:
        correct_act = sum(1 for t, p in actionable_pairs if t == p)
        acc_act = correct_act / len(actionable_pairs)
        print(f"  Actionable-only: {acc_act*100:.1f}% ({correct_act}/{len(actionable_pairs)})")

    # Per-category metrics
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    report = classification_report(y_true, y_pred, labels=V4_3CLASS, digits=4, zero_division=0)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=V4_3CLASS)
    print("  Confusion Matrix:")
    header = f"  {'':>14}"
    for t in V4_3CLASS:
        header += f"  {t[:6]:>6}"
    print(header)
    for i, row in enumerate(cm):
        line = f"  {V4_3CLASS[i]:>14}"
        for val in row:
            line += f"  {val:>6}"
        print(line)

    # 오분류 상세
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                'index': i,
                'text': golden[i]['text'][:100],
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
    print(f"    v3c 5분류 Haiku zero-shot:    78.9%")
    print(f"    v4 R2 4분류 후처리:           84.1%")
    print(f"    R2→3분류 시뮬레이션:          87.2%")
    print(f"    v4 3분류 (이번):              {acc*100:.1f}%")
    delta = acc * 100 - 87.2
    sign = "+" if delta >= 0 else ""
    print(f"    시뮬레이션 대비:              {sign}{delta:.1f}%p")
    print(f"{'='*60}")

    # 비용 리포트
    cost = classifier.get_cost_report()
    print(f"\n  비용: ${cost['cost_usd']:.4f} (in:{cost['input_tokens']:,} / out:{cost['output_tokens']:,})")

    # 결과 저장
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=V4_3CLASS, zero_division=0
    )
    per_cat = {}
    for i, t in enumerate(V4_3CLASS):
        per_cat[t] = {
            'precision': round(float(p[i]), 4),
            'recall': round(float(r[i]), 4),
            'f1': round(float(f1[i]), 4),
            'support': int(support[i]),
        }

    output = {
        'experiment': 'v4_3class_unified_classifier',
        'model': args.model,
        'taxonomy': '3분류 (대응 필요 / 콘텐츠·투자 / 노이즈)',
        'golden_set': 'v7_golden_set.json',
        'golden_set_size': len(golden),
        'accuracy_3class': round(acc, 4),
        'accuracy_actionable': round(acc_act, 4) if actionable_pairs else None,
        'correct': correct,
        'baseline_comparison': {
            'v3c_5way_haiku': 0.789,
            'v4_4way_r2': 0.841,
            'v4_3way_simulation': 0.872,
            'v4_3class_unified': round(acc, 4),
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

    output_path = "benchmarks/golden_benchmark_v4_3class.json"
    os.makedirs('benchmarks', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output_path}")


if __name__ == "__main__":
    main()
