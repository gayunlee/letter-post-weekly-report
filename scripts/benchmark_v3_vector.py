"""v3 분류기 성능 비교 벤치마크

3가지 분류기를 Test set(744건)과 Golden set(102건)에 대해 비교:
1. KcBERT 단독 (현재 81.9%)
2. 벡터 KNN 단독
3. 하이브리드: KcBERT 기본 + softmax 차이 < threshold일 때 벡터 보정

사용법:
    python3 scripts/benchmark_v3_vector.py
    python3 scripts/benchmark_v3_vector.py --threshold 0.3
"""
import sys
import os
import json
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.classifier_v3.vector_v3_classifier import VectorV3Classifier, VALID_TOPICS_4
from src.classifier_v3.taxonomy import TOPICS

TOPICS_4 = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "기타"]


def merge_topic(t):
    if t in ("콘텐츠 반응", "투자 담론"):
        return "콘텐츠·투자"
    return t


# ── KcBERT 예측 ──────────────────────────────────────────────────────

def load_kcbert(model_dir: str = "models/v3/topic/final_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    config_path = Path(model_dir) / "category_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        id_to_cat = {int(k): v for k, v in config["id_to_category"].items()}
    else:
        id_to_cat = {i: t for i, t in enumerate(TOPICS)}

    return tokenizer, model, id_to_cat


def predict_kcbert(tokenizer, model, text: str, id_to_cat: dict):
    """KcBERT 예측 → (label, confidence, softmax_probs)"""
    inputs = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=256, return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    return id_to_cat[pred_id], probs[pred_id].item(), probs


def predict_kcbert_batch(tokenizer, model, texts, id_to_cat, batch_size=32):
    """KcBERT 배치 예측"""
    results = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts, truncation=True, padding="max_length",
            max_length=256, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        for i in range(len(batch_texts)):
            pred_id = probs[i].argmax().item()
            results.append({
                "label": id_to_cat[pred_id],
                "confidence": probs[i][pred_id].item(),
                "probs": probs[i],
            })
    return results


# ── 하이브리드 분류 ──────────────────────────────────────────────────

def hybrid_predict(
    kcbert_label: str,
    kcbert_probs: torch.Tensor,
    vector_label: str,
    vector_confidence: float,
    threshold: float = 0.3,
) -> str:
    """KcBERT 기본 + softmax 차이 < threshold일 때 벡터 보정

    softmax 차이 = top1_prob - top2_prob
    차이가 작으면 KcBERT가 불확실 → 벡터 결과로 교체
    """
    sorted_probs = torch.sort(kcbert_probs, descending=True).values
    margin = (sorted_probs[0] - sorted_probs[1]).item()

    if margin < threshold:
        return vector_label
    return kcbert_label


# ── 평가 ─────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label: str, topics=None):
    """정확도, classification report, confusion matrix 출력"""
    if topics is None:
        topics = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=topics, digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=topics)

    print(f"\n{'─'*50}")
    print(f"  [{label}] 정확도: {acc*100:.1f}%")
    print(f"{'─'*50}")
    print(report)

    # Confusion matrix
    print(f"  {'':>14}", end="")
    for name in topics:
        print(f"  {name[:6]:>6}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {topics[i]:>14}", end="")
        for val in row:
            print(f"  {val:>6}", end="")
        print()

    return acc


def run_benchmark(dataset_name, texts, y_true, tokenizer, model, id_to_cat, vec_clf, threshold):
    """단일 데이터셋에 대해 3가지 분류기 벤치마크"""
    print(f"\n{'='*60}")
    print(f"  [{dataset_name}] {len(texts)}건 벤치마크")
    print(f"{'='*60}")

    # 1) KcBERT
    print("\n  KcBERT 예측 중...")
    kcbert_results = predict_kcbert_batch(tokenizer, model, texts, id_to_cat)
    y_kcbert = [r["label"] for r in kcbert_results]
    acc_kcbert = evaluate(y_true, y_kcbert, f"{dataset_name} / KcBERT")

    # 2) 벡터 KNN
    print("\n  벡터 KNN 예측 중...")
    vec_neighbors_all = []
    BATCH = 200
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        from src.classifier_v3.vector_v3_classifier import preprocess_text
        processed = [preprocess_text(t)[:500] for t in batch]
        neighbors = vec_clf.store.search_similar_batch(
            query_texts=processed, n_results=vec_clf.k,
        )
        vec_neighbors_all.extend(neighbors)

    y_vector = []
    vec_confidences = []
    for neighbors in vec_neighbors_all:
        result = vec_clf._classify_from_neighbors(neighbors)
        y_vector.append(result["topic"])
        vec_confidences.append(result["topic_confidence"])

    acc_vector = evaluate(y_true, y_vector, f"{dataset_name} / 벡터 KNN")

    # 3) 하이브리드
    print(f"\n  하이브리드 예측 중... (threshold={threshold})")
    y_hybrid = []
    corrections = 0
    for i in range(len(texts)):
        hybrid_label = hybrid_predict(
            kcbert_results[i]["label"],
            kcbert_results[i]["probs"],
            y_vector[i],
            vec_confidences[i],
            threshold=threshold,
        )
        if hybrid_label != kcbert_results[i]["label"]:
            corrections += 1
        y_hybrid.append(hybrid_label)

    acc_hybrid = evaluate(y_true, y_hybrid, f"{dataset_name} / 하이브리드 (t={threshold})")
    print(f"  벡터 보정 횟수: {corrections}/{len(texts)} ({corrections/len(texts)*100:.1f}%)")

    # 보정 효과 분석
    improved = 0
    degraded = 0
    for i in range(len(texts)):
        if y_hybrid[i] != kcbert_results[i]["label"]:
            if y_hybrid[i] == y_true[i]:
                improved += 1
            elif kcbert_results[i]["label"] == y_true[i]:
                degraded += 1
    print(f"  보정 결과: 개선 {improved}건, 악화 {degraded}건, 순효과 {improved - degraded:+d}건")

    return {
        "kcbert": acc_kcbert,
        "vector": acc_vector,
        "hybrid": acc_hybrid,
        "corrections": corrections,
        "improved": improved,
        "degraded": degraded,
    }


def main():
    parser = argparse.ArgumentParser(description="v3 분류기 비교 벤치마크")
    parser.add_argument("--model-dir", default="models/v3/topic/final_model")
    parser.add_argument("--vector-dir", default="data/vectorstore_v3")
    parser.add_argument("--test-set", default="data/training_data/v3/topic/test.json")
    parser.add_argument("--golden-set", default="data/gold_dataset/v5_golden_set.json")
    parser.add_argument("--threshold", type=float, default=0.3, help="하이브리드 softmax margin threshold")
    parser.add_argument("--k", type=int, default=7, help="KNN k값")
    args = parser.parse_args()

    print("=" * 60)
    print("  v3 분류기 비교 벤치마크")
    print("  KcBERT vs 벡터 KNN vs 하이브리드")
    print("=" * 60)

    # 모델 로드
    print(f"\n  KcBERT 로드: {args.model_dir}")
    tokenizer, model, id_to_cat = load_kcbert(args.model_dir)

    print(f"  벡터 인덱스 로드: {args.vector_dir}")
    vec_clf = VectorV3Classifier(persist_dir=args.vector_dir, k=args.k)
    vec_count = vec_clf.store.collection.count()
    if vec_count == 0:
        print("  벡터 인덱스가 비어있습니다. 먼저 build_v3_vector_index.py를 실행하세요.")
        sys.exit(1)
    print(f"  벡터 인덱스: {vec_count}건")

    # 4분류 모델인지 자동 감지
    model_categories = set(id_to_cat.values())
    is_4cat = "콘텐츠·투자" in model_categories
    if is_4cat:
        print("  모드: 4분류 (콘텐츠·투자 합산)")
    else:
        print("  모드: 5분류")

    all_results = {}

    # Test set
    test_path = Path(args.test_set)
    if test_path.exists():
        with open(test_path, encoding="utf-8") as f:
            test_data = json.load(f)
        texts = [d["text"] for d in test_data]
        y_true = [d["category"] for d in test_data]
        if is_4cat:
            y_true = [merge_topic(t) for t in y_true]
        all_results["test"] = run_benchmark(
            "Test", texts, y_true, tokenizer, model, id_to_cat, vec_clf, args.threshold,
        )
    else:
        print(f"\n  Test set 없음: {test_path}")

    # Golden set
    golden_path = Path(args.golden_set)
    if golden_path.exists():
        with open(golden_path, encoding="utf-8") as f:
            golden_data = json.load(f)
        texts = [d["text"] for d in golden_data]
        y_true = [d["v5_topic"] for d in golden_data]
        if is_4cat:
            y_true = [merge_topic(t) for t in y_true]
        all_results["golden"] = run_benchmark(
            "Golden", texts, y_true, tokenizer, model, id_to_cat, vec_clf, args.threshold,
        )
    else:
        print(f"\n  Golden set 없음: {golden_path}")

    # 종합 요약
    print(f"\n{'='*60}")
    print(f"  종합 요약 (threshold={args.threshold}, k={args.k})")
    print(f"{'='*60}")
    for name, r in all_results.items():
        print(f"\n  [{name}]")
        print(f"    KcBERT:    {r['kcbert']*100:.1f}%")
        print(f"    벡터 KNN:  {r['vector']*100:.1f}%")
        print(f"    하이브리드: {r['hybrid']*100:.1f}%")
        print(f"    보정: {r['corrections']}건 (개선 {r['improved']}, 악화 {r['degraded']}, 순효과 {r['improved']-r['degraded']:+d})")
    print(f"{'='*60}")

    # 결과 저장
    result_path = Path(args.vector_dir) / "benchmark_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": args.threshold,
            "k": args.k,
            "vector_count": vec_count,
            "results": {
                name: {k: v for k, v in r.items()}
                for name, r in all_results.items()
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {result_path}")


if __name__ == "__main__":
    main()
