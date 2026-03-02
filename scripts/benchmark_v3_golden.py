"""v3 Golden Set 벤치마크 스크립트

파인튜닝된 KcBERT v3 모델을 golden set (102건)에 대해 평가.
v2 정확도 69.3% 대비 v3 개선 여부 확인.

사용법:
    python3 scripts/benchmark_v3_golden.py
    python3 scripts/benchmark_v3_golden.py --model-dir models/v3/topic/final_model
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


V3_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠 반응", "투자 담론", "기타"]


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    config_path = Path(model_dir) / "category_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        id_to_category = {int(k): v for k, v in config["id_to_category"].items()}
    else:
        id_to_category = {i: t for i, t in enumerate(V3_TOPICS)}

    return tokenizer, model, id_to_category


def predict(tokenizer, model, text: str, id_to_category: dict) -> tuple:
    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=256, return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    confidence = probs[pred_id].item()
    return id_to_category[pred_id], confidence


def main():
    parser = argparse.ArgumentParser(description="v3 Golden Set 벤치마크")
    parser.add_argument("--model-dir", default="./models/v3/topic/final_model")
    parser.add_argument("--golden-set", default="./data/gold_dataset/v5_golden_set.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  v3 Golden Set 벤치마크")
    print("=" * 60)

    # Golden set 로드
    with open(args.golden_set, encoding="utf-8") as f:
        golden = json.load(f)
    print(f"\n  Golden set: {len(golden)}건")

    # 모델 로드
    print(f"  모델: {args.model_dir}")
    tokenizer, model, id_to_category = load_model(args.model_dir)

    # 예측
    y_true = []
    y_pred = []
    misclassified = []

    for item in golden:
        true_topic = item["v5_topic"]
        pred_topic, confidence = predict(tokenizer, model, item["text"], id_to_category)

        y_true.append(true_topic)
        y_pred.append(pred_topic)

        if true_topic != pred_topic:
            misclassified.append({
                "text": item["text"][:80],
                "true": true_topic,
                "pred": pred_topic,
                "confidence": confidence,
            })

    # 결과
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=V3_TOPICS, digits=4, zero_division=0)

    print(f"\n  정확도: {acc*100:.1f}%")
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=V3_TOPICS)
    print("  Confusion Matrix:")
    print(f"  {'':>14}", end="")
    for name in V3_TOPICS:
        print(f"  {name[:6]:>6}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {V3_TOPICS[i]:>14}", end="")
        for val in row:
            print(f"  {val:>6}", end="")
        print()

    # 오분류 사례
    if misclassified:
        print(f"\n  오분류: {len(misclassified)}건")
        for m in misclassified[:10]:
            print(f"    [{m['true']} → {m['pred']}] conf={m['confidence']:.2f} | {m['text']}")

    # v2 대비 비교
    print(f"\n{'='*60}")
    print(f"  v2 Topic 정확도: 69.3%")
    print(f"  v3 Topic 정확도: {acc*100:.1f}%")
    diff = (acc - 0.693) * 100
    print(f"  차이: {diff:+.1f}%p")
    if diff > 0:
        print(f"  → v3 개선 확인!")
    elif diff == 0:
        print(f"  → 동일 수준")
    else:
        print(f"  → v3 하락 — 분석 필요")
    print(f"{'='*60}")

    # 결과 저장
    result_path = Path(args.model_dir).parent / "golden_benchmark.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "golden_set_size": len(golden),
            "accuracy": acc,
            "v2_accuracy": 0.693,
            "diff_pp": round(diff, 1),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "misclassified_count": len(misclassified),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {result_path}")


if __name__ == "__main__":
    main()
