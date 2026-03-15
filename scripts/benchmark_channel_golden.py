"""채널톡 CS 분류 벤치마크 — golden set 기반

2축 분류 평가:
  - Axis 1 (route): 규칙 기반 → 정확도 확인 (verify)
  - Axis 2 (topic): LLM 분류 → P/R/F1 측정

사용법:
    # ChannelClassifier (Haiku) 벤치마크
    python3 scripts/benchmark_channel_golden.py

    # 다른 모델
    python3 scripts/benchmark_channel_golden.py --model claude-sonnet-4-5-20250514

    # golden set 경로 지정
    python3 scripts/benchmark_channel_golden.py --golden data/channel_io/golden/channel_golden_v1.json

비교 전략:
    이 벤치마크로 Haiku API / 파인튜닝 / 벡터 분류기 / 하이브리드 비교 가능.
    --classifier 옵션으로 분류 백엔드 교체.
"""
import sys
import os
import json
import argparse
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

CHANNEL_TOPICS = ["결제·환불", "구독·멤버십", "콘텐츠·수강", "기술·오류", "기타"]
CHANNEL_ROUTES = ["manager_resolved", "bot_resolved", "abandoned"]


def load_golden(path):
    """Golden set 로드. verified_topic이 있는 항목만 반환."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", data) if isinstance(data, dict) else data
    # verified_topic이 채워진 것만 = 검수 완료
    verified = [i for i in items if i.get("verified_topic")]
    return verified


def run_haiku_benchmark(golden, model, max_workers):
    """ChannelClassifier (Haiku API) 벤치마크"""
    from src.classifier_v4.channel_classifier import ChannelClassifier

    classifier = ChannelClassifier(model=model, max_workers=max_workers)

    results = [None] * len(golden)
    start_time = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for i, item in enumerate(golden):
            buttons = item.get("workflow_buttons", [])
            future = executor.submit(
                classifier.classify_single, item["text"], buttons
            )
            future_map[future] = i

        done_count = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
            done_count += 1
            if done_count % 20 == 0 or done_count == len(golden):
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"    {done_count}/{len(golden)} 완료 ({rate:.1f}건/초)")

    elapsed = time.time() - start_time
    cost = classifier.get_cost_report()
    return results, elapsed, cost


def evaluate(golden, predictions, label_field="verified_topic"):
    """정확도, P/R/F1, confusion matrix 계산"""
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        accuracy_score,
    )

    y_true = [item[label_field] for item in golden]
    y_pred = [p["topic"] for p in predictions]

    # 정확도
    acc = accuracy_score(y_true, y_pred)

    # 분류 리포트
    report = classification_report(
        y_true, y_pred, labels=CHANNEL_TOPICS, digits=4, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=CHANNEL_TOPICS)

    # Per-category
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CHANNEL_TOPICS, zero_division=0
    )
    per_cat = {}
    for i, t in enumerate(CHANNEL_TOPICS):
        per_cat[t] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    # 오분류 상세
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                "index": i,
                "chatId": golden[i].get("chatId", ""),
                "text": golden[i]["text"][:120],
                "true": true,
                "pred": pred,
                "confidence": predictions[i].get("confidence", 0),
                "summary": predictions[i].get("summary", ""),
                "tags": predictions[i].get("tags", []),
                "route": golden[i].get("route", ""),
            })

    # 오분류 패턴
    error_patterns = Counter()
    for m in misclassified:
        error_patterns[f"{m['true']}→{m['pred']}"] += 1

    return {
        "accuracy": round(acc, 4),
        "correct": sum(1 for a, b in zip(y_true, y_pred) if a == b),
        "total": len(golden),
        "per_category": per_cat,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "misclassified": misclassified,
        "error_patterns": dict(error_patterns),
    }


def evaluate_route(golden, label_field="verified_route"):
    """route 축 정확도 검증 (규칙 기반이므로 높아야 함)"""
    verified = [i for i in golden if i.get(label_field)]
    if not verified:
        return None

    correct = sum(1 for i in verified if i["route"] == i[label_field])
    acc = correct / len(verified) if verified else 0

    errors = [
        {"chatId": i.get("chatId"), "auto_route": i["route"], "verified": i[label_field]}
        for i in verified
        if i["route"] != i[label_field]
    ]

    return {
        "accuracy": round(acc, 4),
        "correct": correct,
        "total": len(verified),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="채널톡 CS 분류 벤치마크")
    parser.add_argument(
        "--golden",
        default="data/channel_io/golden/channel_golden_v1.json",
        help="Golden set 경로",
    )
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument(
        "--classifier",
        default="haiku",
        choices=["haiku"],
        help="분류 백엔드 (향후 finetuned, vector, hybrid 추가)",
    )
    parser.add_argument("--output", default=None, help="결과 저장 경로")
    args = parser.parse_args()

    # Golden set 로드
    if not os.path.exists(args.golden):
        print(f"  Golden set 파일이 없습니다: {args.golden}")
        print(f"  먼저 golden set을 생성하세요:")
        print(f"    1. python3 scripts/classify_channel_talk.py --weeks 4")
        print(f"    2. python3 scripts/extract_channel_golden.py data/channel_io/classified_*.json")
        print(f"    3. 수동 검수 후 verified_topic 필드 채우기")
        return

    golden = load_golden(args.golden)
    if not golden:
        print(f"  검수된 항목이 없습니다 (verified_topic이 비어있음)")
        print(f"  golden set 파일에서 verified_topic 필드를 채워주세요.")
        return

    print("=" * 60)
    print(f"  채널톡 CS 분류 벤치마크")
    print(f"  분류기: {args.classifier} ({args.model})")
    print(f"  Golden set: {len(golden)}건 (검수 완료)")
    print("=" * 60)

    # 정답 분포
    true_topics = Counter(item["verified_topic"] for item in golden)
    print(f"\n  정답 (topic) 분포:")
    for t in CHANNEL_TOPICS:
        print(f"    {t:<12} {true_topics.get(t, 0):>4}건")

    true_routes = Counter(item.get("route", "unknown") for item in golden)
    print(f"\n  Route 분포:")
    for r in CHANNEL_ROUTES:
        print(f"    {r:<20} {true_routes.get(r, 0):>4}건")

    # 분류 실행
    print(f"\n  분류 실행 중...")
    if args.classifier == "haiku":
        predictions, elapsed, cost = run_haiku_benchmark(
            golden, args.model, args.max_workers
        )
    else:
        raise ValueError(f"미지원 분류기: {args.classifier}")

    print(f"\n  총 소요: {elapsed:.1f}초 ({len(golden)/elapsed:.1f}건/초)")

    # 에러 체크
    errors = [(i, r.get("error", "")) for i, r in enumerate(predictions) if "error" in r]
    if errors:
        print(f"\n  API 에러: {len(errors)}건")
        for idx, err in errors[:5]:
            print(f"    [{idx}] {err}")

    # Topic 평가
    print(f"\n{'='*60}")
    print(f"  [Axis 2] Topic 분류 성능")
    print(f"{'='*60}")
    eval_result = evaluate(golden, predictions)

    print(f"\n  정확도: {eval_result['accuracy']*100:.1f}% ({eval_result['correct']}/{eval_result['total']})")
    print(f"\n{eval_result['classification_report']}")

    # Confusion matrix
    cm = eval_result["confusion_matrix"]
    print("  Confusion Matrix:")
    cm_label = "실제\\예측"
    header = f"  {cm_label:>14}"
    for t in CHANNEL_TOPICS:
        header += f"  {t[:5]:>6}"
    print(header)
    for i, row in enumerate(cm):
        line = f"  {CHANNEL_TOPICS[i]:>14}"
        for val in row:
            line += f"  {val:>6}"
        print(line)

    # 오분류
    misclassified = eval_result["misclassified"]
    print(f"\n  오분류: {len(misclassified)}건")
    for m in misclassified[:15]:
        print(
            f"    [{m['index']}] {m['true']} → {m['pred']} "
            f"| route={m['route']} | conf={m['confidence']:.2f} "
            f"| {m['summary'][:50]}"
        )

    if eval_result["error_patterns"]:
        print(f"\n  오분류 패턴:")
        for pattern, count in sorted(
            eval_result["error_patterns"].items(), key=lambda x: -x[1]
        ):
            print(f"    {pattern}: {count}건")

    # Route 평가
    route_eval = evaluate_route(golden)
    if route_eval:
        print(f"\n{'='*60}")
        print(f"  [Axis 1] Route 판정 검증")
        print(f"{'='*60}")
        print(f"  정확도: {route_eval['accuracy']*100:.1f}% ({route_eval['correct']}/{route_eval['total']})")
        if route_eval["errors"]:
            print(f"  오류:")
            for e in route_eval["errors"]:
                print(f"    {e['chatId']}: {e['auto_route']} → {e['verified']}")

    # 비용
    if cost:
        print(f"\n  비용: ${cost['cost_usd']:.4f} (in:{cost['input_tokens']:,} / out:{cost['output_tokens']:,})")

    # 결과 저장
    output_path = args.output or f"benchmarks/golden_benchmark_channel_{args.classifier}.json"
    os.makedirs("benchmarks", exist_ok=True)

    output = {
        "experiment": f"channel_cs_{args.classifier}",
        "model": args.model,
        "classifier": args.classifier,
        "taxonomy": "채널톡 2축 (route + topic 5분류)",
        "golden_set": args.golden,
        "golden_set_size": len(golden),
        "topic_accuracy": eval_result["accuracy"],
        "topic_correct": eval_result["correct"],
        "per_category": eval_result["per_category"],
        "classification_report": eval_result["classification_report"],
        "confusion_matrix": eval_result["confusion_matrix"],
        "misclassified": eval_result["misclassified"],
        "error_patterns": eval_result["error_patterns"],
        "route_eval": route_eval,
        "cost": cost,
        "elapsed_seconds": round(elapsed, 1),
        "api_errors": len(errors),
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {output_path}")

    print(f"\n{'='*60}")
    print(f"  벤치마크 완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
