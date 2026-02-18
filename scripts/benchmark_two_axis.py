"""2축 분류 벤치마크 스크립트

파인튜닝 모델 vs 벡터 분류기를 test set에서 비교합니다.

결과 (2026-02-19):
  Topic:     파인튜닝 69.3% vs 벡터 55.5% → 파인튜닝 우위
  Sentiment: 파인튜닝 84.1% vs 벡터 45.3% → 파인튜닝 압도적 우위
  → 2축 체계 프로덕션은 파인튜닝 모델만 사용 결정

사용법:
    python3 scripts/benchmark_two_axis.py
"""
import json
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.chroma_store import ChromaVectorStore
from src.classifier.vector_classifier import preprocess_text


def load_test_data(axis: str, data_dir: Path):
    with open(data_dir / axis / "test.json", encoding="utf-8") as f:
        return json.load(f)


def load_mapping(axis: str, data_dir: Path):
    with open(data_dir / axis / "category_mapping.json", encoding="utf-8") as f:
        return json.load(f)


def run_finetuned(test_data, model_dir: Path, mapping):
    id_to_cat = {int(k): v for k, v in mapping["id_to_category"].items()}
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)

    preds = []
    start = time.time()

    batch_size = 32
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        texts = [item["text"] for item in batch]
        encoding = tokenizer(
            texts, truncation=True, padding=True,
            max_length=256, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            batch_preds = outputs.logits.argmax(-1).cpu().tolist()
            preds.extend([id_to_cat[p] for p in batch_preds])

    elapsed = time.time() - start
    return preds, elapsed


def run_vector(test_data, collection_name: str, persist_dir: str, k: int = 3):
    store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    preds = []
    start = time.time()

    texts = [preprocess_text(item["text"])[:500] for item in test_data]
    batch_results = store.search_similar_batch(texts, n_results=k)

    for neighbors in batch_results:
        votes = Counter()
        for n in neighbors:
            cat = n["metadata"].get("category", "미분류")
            dist = n.get("distance", 1.0)
            sim = max(0.0, 1.0 - dist)
            votes[cat] += sim

        if votes:
            preds.append(votes.most_common(1)[0][0])
        else:
            preds.append("미분류")

    elapsed = time.time() - start
    return preds, elapsed


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "training_data" / "two_axis"
    model_dir = project_root / "models" / "two_axis"
    vector_dir = str(project_root / "chroma_db_two_axis")

    print("=" * 70)
    print("  2축 분류 벤치마크: 파인튜닝 vs 벡터 분류기")
    print("=" * 70)

    results = {}

    for axis in ["topic", "sentiment"]:
        print(f"\n{'─'*70}")
        print(f"  {axis.upper()} 분류 비교")
        print(f"{'─'*70}")

        test_data = load_test_data(axis, data_dir)
        mapping = load_mapping(axis, data_dir)
        id_to_cat = {int(k): v for k, v in mapping["id_to_category"].items()}
        ground_truth = [item["category"] for item in test_data]

        print(f"\n  Test set: {len(test_data)}건")
        gt_dist = Counter(ground_truth)
        for cat, cnt in gt_dist.most_common():
            print(f"    {cat}: {cnt}건")

        # 파인튜닝 모델
        print(f"\n  [1/2] 파인튜닝 모델...")
        ft_preds, ft_time = run_finetuned(
            test_data,
            model_dir / axis / "final_model",
            mapping
        )
        ft_acc = accuracy_score(ground_truth, ft_preds)
        print(f"    정확도: {ft_acc*100:.1f}% ({ft_time:.2f}초)")
        print(f"\n{classification_report(ground_truth, ft_preds, digits=4)}")

        # 벡터 분류기
        print(f"  [2/2] 벡터 분류기...")
        collection = f"two_axis_{axis}"
        vec_preds, vec_time = run_vector(test_data, collection, vector_dir)
        vec_acc = accuracy_score(ground_truth, vec_preds)
        print(f"    정확도: {vec_acc*100:.1f}% ({vec_time:.2f}초)")
        print(f"\n{classification_report(ground_truth, vec_preds, digits=4)}")

        # 비교 요약
        print(f"\n  ┌────────────────┬──────────────┬──────────────┐")
        print(f"  │ {axis.upper():14s} │ 파인튜닝     │ 벡터 분류기  │")
        print(f"  ├────────────────┼──────────────┼──────────────┤")
        print(f"  │ 정확도         │ {ft_acc*100:10.1f}% │ {vec_acc*100:10.1f}% │")
        print(f"  │ 처리 시간      │ {ft_time:10.2f}초 │ {vec_time:10.2f}초 │")
        print(f"  └────────────────┴──────────────┴──────────────┘")

        results[axis] = {
            "finetuned": {"accuracy": ft_acc, "time": ft_time},
            "vector": {"accuracy": vec_acc, "time": vec_time},
            "test_size": len(test_data),
        }

    # 최종 요약
    print(f"\n{'='*70}")
    print("  최종 요약")
    print(f"{'='*70}")
    for axis, res in results.items():
        ft = res["finetuned"]
        vec = res["vector"]
        winner = "파인튜닝" if ft["accuracy"] > vec["accuracy"] else "벡터"
        print(f"\n  {axis.upper()}: 파인튜닝 {ft['accuracy']*100:.1f}% vs 벡터 {vec['accuracy']*100:.1f}% → {winner} 우위")

    # 결과 저장
    report_file = project_root / "reports" / "two_axis_benchmark.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {report_file}")


if __name__ == "__main__":
    main()
