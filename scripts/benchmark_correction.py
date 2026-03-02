"""4분류 보정 파이프라인 벤치마크

Golden set (102건)에서 LLM 보정 효과 측정:
1. KcBERT 4분류 분류 (softmax margin 포함)
2. LLM 재검증 보정 적용 (margin 낮은 소수 카테고리 관련 건)
3. Opus 라벨(정답)과 비교: 보정 전/후 정확도, 개선/악화 건수

사용법:
    python3 scripts/benchmark_correction.py
"""
import sys
import os
import json
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 5분류 → 4분류 매핑 ──────────────────────────────────────────────
V5_TO_V4 = {
    "운영 피드백": "운영 피드백",
    "서비스 피드백": "서비스 피드백",
    "콘텐츠 반응": "콘텐츠·투자",
    "투자 담론": "콘텐츠·투자",
    "기타": "기타",
}

V4_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "기타"]


def load_golden(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["v4_topic"] = V5_TO_V4.get(item["v5_topic"], "기타")
    return data


def step1_classify(golden, classifier):
    """KcBERT 4분류 + margin/top2"""
    print("\n[Step 1] KcBERT 4분류 분류...")
    items_for_classify = [{"message": item["text"]} for item in golden]
    classified = classifier.classify_batch(items_for_classify, content_field="message")

    for i, item in enumerate(golden):
        item["classification"] = classified[i]["classification"]

    pred_dist = Counter(item["classification"]["topic"] for item in golden)
    print(f"  분류 분포: {dict(pred_dist)}")

    correct = sum(
        1 for item in golden
        if item["classification"]["topic"] == item["v4_topic"]
    )
    acc = correct / len(golden) * 100
    print(f"  보정 전 정확도: {acc:.1f}% ({correct}/{len(golden)})")

    # margin 분포 확인
    minority_topics = {"운영 피드백", "서비스 피드백"}
    ct_items = [item for item in golden if item["classification"]["topic"] == "콘텐츠·투자"]
    low_margin_minority = [
        item for item in ct_items
        if item["classification"].get("topic_margin", 1.0) < 0.3
        and item["classification"].get("topic_top2", "") in minority_topics
    ]
    print(f"  콘텐츠·투자 분류: {len(ct_items)}건, margin<0.3 AND top2∈소수: {len(low_margin_minority)}건")

    return golden, acc


def step2_correct(golden):
    """LLM 재검증 보정"""
    print("\n[Step 2] LLM 재검증 (top2 ∈ 소수 카테고리)...")
    from src.classifier_v3.topic_corrector import correct_topics

    result = correct_topics(golden, content_field="text")
    golden = result["items"]
    stats = result["stats"]

    correct = sum(
        1 for item in golden
        if item["classification"]["topic"] == item["v4_topic"]
    )
    acc = correct / len(golden) * 100
    print(f"  보정 후 정확도: {acc:.1f}% ({correct}/{len(golden)})")
    return golden, acc, stats


def analyze_changes(golden, pre_acc):
    """보정 전후 개선/악화 분석"""
    print("\n" + "=" * 60)
    print("  보정 결과 분석")
    print("=" * 60)

    improved = []
    worsened = []
    all_corrected = []

    for item in golden:
        cls = item["classification"]
        if not cls.get("topic_before_correction"):
            continue

        before = cls["topic_before_correction"]
        after = cls["topic"]
        true = item["v4_topic"]

        entry = {
            "text": item["text"][:60],
            "true": true,
            "before": before,
            "after": after,
            "reason": cls.get("correction_reason", ""),
            "confidence": cls.get("correction_confidence", 0),
        }
        all_corrected.append(entry)

        if before != true and after == true:
            improved.append(entry)
        elif before == true and after != true:
            worsened.append(entry)

    print(f"\n  총 보정 건수: {len(all_corrected)}")
    print(f"  개선 (오분류→정분류): {len(improved)}건")
    print(f"  악화 (정분류→오분류): {len(worsened)}건")
    print(f"  무변화 (오분류→오분류): {len(all_corrected) - len(improved) - len(worsened)}건")

    if improved:
        print("\n  [개선 사례]")
        for e in improved:
            print(f"    [{e['before']} → {e['after']}] conf={e['confidence']:.2f} | {e['reason']}")
            print(f"      {e['text']}")

    if worsened:
        print("\n  [악화 사례]")
        for e in worsened:
            print(f"    [{e['before']} → {e['after']}] conf={e['confidence']:.2f} | {e['reason']}")
            print(f"      {e['text']}")

    # 카테고리별 정확도
    print("\n  [카테고리별 정확도]")
    for topic in V4_TOPICS:
        topic_items = [item for item in golden if item["v4_topic"] == topic]
        if not topic_items:
            continue
        correct = sum(1 for item in topic_items if item["classification"]["topic"] == topic)
        acc = correct / len(topic_items) * 100
        print(f"    {topic}: {acc:.1f}% ({correct}/{len(topic_items)})")

    # 잔존 오분류
    misclassified = [
        item for item in golden
        if item["classification"]["topic"] != item["v4_topic"]
    ]
    if misclassified:
        print(f"\n  [잔존 오분류: {len(misclassified)}건]")
        for item in misclassified[:10]:
            cls = item["classification"]
            print(f"    [{item['v4_topic']} → {cls['topic']}] "
                  f"conf={cls.get('topic_confidence', 0):.2f} "
                  f"margin={cls.get('topic_margin', 0):.2f} "
                  f"top2={cls.get('topic_top2', '')} | "
                  f"{item['text'][:60]}")


def main():
    parser = argparse.ArgumentParser(description="4분류 보정 파이프라인 벤치마크")
    parser.add_argument("--golden-set", default="./data/gold_dataset/v5_golden_set.json")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  4분류 보정 파이프라인 벤치마크")
    print("=" * 60)

    # Golden set 로드
    golden = load_golden(args.golden_set)
    true_dist = Counter(item["v4_topic"] for item in golden)
    print(f"\n  Golden set: {len(golden)}건")
    print(f"  4분류 정답 분포: {dict(true_dist)}")

    # Step 1: KcBERT 4분류
    from src.classifier_v3.v3_topic_classifier import V3TopicClassifier
    kwargs = {}
    if args.model_dir:
        kwargs["topic_model_dir"] = args.model_dir
    classifier = V3TopicClassifier(**kwargs)
    golden, pre_acc = step1_classify(golden, classifier)

    # Step 2: LLM 보정
    golden, post_acc, correction_stats = step2_correct(golden)

    # 분석
    analyze_changes(golden, pre_acc)

    # 요약
    print("\n" + "=" * 60)
    print("  최종 요약")
    print("=" * 60)
    print(f"  보정 전 정확도:    {pre_acc:.1f}%")
    print(f"  보정 후 정확도:    {post_acc:.1f}% ({post_acc - pre_acc:+.1f}%p)")
    print(f"  LLM 검토 건수:    {correction_stats['candidates']}건")
    print(f"  LLM 보정 건수:    {correction_stats['corrected']}건")
    if correction_stats.get("llm_cost"):
        cost = correction_stats["llm_cost"]
        print(f"  LLM 비용:          ${cost.get('estimated_cost_usd', 0):.4f}")
    print(f"  v2 기준:           69.3%")
    print("=" * 60)

    # 결과 저장
    result_path = Path("models/v3/topic_4cat/correction_benchmark.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "golden_set_size": len(golden),
            "pre_correction_accuracy": pre_acc,
            "post_correction_accuracy": post_acc,
            "correction_stats": correction_stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {result_path}")


if __name__ == "__main__":
    main()
