"""Golden set v7 Opus 재라벨링 — 3분류 경계 재정비

Opus로 227건 전수 분류 → 기존 v4_3class 라벨과 비교 → 불일치 건 출력.
최종 라벨은 불일치 건 개별 검수 후 확정.

사용법:
    python3 scripts/relabel_golden_v7_opus.py
"""
import sys, os, json, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier_v4.unified_classifier import UnifiedClassifier

V4_3CLASS = ["대응 필요", "콘텐츠·투자", "노이즈"]


def main():
    golden_path = "data/gold_dataset/v7_golden_set.json"
    with open(golden_path, encoding='utf-8') as f:
        golden = json.load(f)

    print("=" * 60)
    print(f"  Golden set Opus 재라벨링 — 3분류")
    print(f"  모델: claude-opus-4-6")
    print(f"  Golden set: {len(golden)}건")
    print("=" * 60)

    # 기존 분포
    old_dist = Counter(item['v4_3class'] for item in golden)
    print(f"\n  기존 분포:")
    for t in V4_3CLASS:
        print(f"    {t}: {old_dist.get(t, 0)}건")

    # Opus로 분류
    classifier = UnifiedClassifier(model="claude-opus-4-6", max_workers=5)

    results = [None] * len(golden)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
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
    print(f"\n  총 소요: {elapsed:.1f}초")

    # 에러 체크
    errors = [(i, r.get('error', '')) for i, r in enumerate(results) if 'error' in r]
    if errors:
        print(f"\n  에러: {len(errors)}건")
        for idx, err in errors[:5]:
            print(f"    [{idx}] {err}")

    # Opus 분포
    opus_labels = [r['topic'] for r in results]
    opus_dist = Counter(opus_labels)
    print(f"\n  Opus 분포:")
    for t in V4_3CLASS:
        print(f"    {t}: {opus_dist.get(t, 0)}건")

    # 일치/불일치 분석
    agreements = []
    disagreements = []
    for i, (item, result) in enumerate(zip(golden, results)):
        old_label = item['v4_3class']
        new_label = result['topic']
        if old_label == new_label:
            agreements.append(i)
        else:
            disagreements.append({
                'index': i,
                'text': item['text'][:150],
                'v3c_topic': item.get('v3c_topic', ''),
                'current_label': old_label,
                'opus_label': new_label,
                'opus_summary': result.get('summary', ''),
                'opus_confidence': result.get('confidence', 0),
                'opus_sentiment': result.get('sentiment', ''),
                'opus_tags': result.get('tags', []),
            })

    agree_rate = len(agreements) / len(golden) * 100
    print(f"\n  일치: {len(agreements)}건 ({agree_rate:.1f}%)")
    print(f"  불일치: {len(disagreements)}건")

    # 불일치 패턴
    patterns = Counter()
    for d in disagreements:
        patterns[f"{d['current_label']}→{d['opus_label']}"] += 1
    print(f"\n  불일치 패턴:")
    for pattern, count in patterns.most_common():
        print(f"    {pattern}: {count}건")

    # 불일치 상세
    print(f"\n{'='*60}")
    print(f"  불일치 상세 (검수 대상)")
    print(f"{'='*60}")
    for d in disagreements:
        print(f"\n  [{d['index']}] {d['current_label']} → {d['opus_label']} (conf={d['opus_confidence']:.2f})")
        print(f"    text: {d['text']}")
        print(f"    opus요약: {d['opus_summary'][:80]}")
        print(f"    v3c: {d['v3c_topic']}")

    # 비용
    cost = classifier.get_cost_report()
    print(f"\n  비용: ${cost['cost_usd']:.4f} (in:{cost['input_tokens']:,} / out:{cost['output_tokens']:,})")

    # 결과 저장
    output = {
        'model': 'claude-opus-4-6',
        'golden_set_size': len(golden),
        'agreements': len(agreements),
        'disagreements_count': len(disagreements),
        'agreement_rate': round(agree_rate, 1),
        'old_distribution': dict(old_dist),
        'opus_distribution': dict(opus_dist),
        'disagreement_patterns': dict(patterns),
        'disagreements': disagreements,
        'all_results': [
            {
                'index': i,
                'text': golden[i]['text'][:200],
                'current_label': golden[i]['v4_3class'],
                'opus_label': r['topic'],
                'opus_summary': r.get('summary', ''),
                'opus_confidence': r.get('confidence', 0),
                'opus_sentiment': r.get('sentiment', ''),
                'opus_tags': r.get('tags', []),
            }
            for i, r in enumerate(results)
        ],
        'cost': cost,
        'elapsed_seconds': round(elapsed, 1),
    }

    output_path = "benchmarks/golden_v7_opus_relabel.json"
    os.makedirs('benchmarks', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output_path}")


if __name__ == "__main__":
    main()
