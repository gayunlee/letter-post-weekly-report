"""Gemma FT 모델 Golden set 벤치마크

파인튜닝된 Gemma 모델 또는 zero-shot Gemma로 Golden set 평가.

사용법:
    # Zero-shot (base 모델)
    python3 scripts/benchmark_gemma_golden.py --model google/gemma-3-1b-it

    # LoRA FT 모델 (merged)
    python3 scripts/benchmark_gemma_golden.py --model models/v4/gemma_3class/merged_model

    # LoRA 어댑터
    python3 scripts/benchmark_gemma_golden.py --model models/v4/gemma_3class/final_model --adapter
"""
import json
import argparse
import time
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)


V4_3CLASS = ["대응 필요", "콘텐츠·투자", "노이즈"]
LABEL_MAP = {"대응 필요": 0, "콘텐츠·투자": 1, "노이즈": 2}


def load_model(model_path, adapter=False, device="cpu"):
    """모델 로드 (merged or LoRA adapter)"""
    model_path = str(model_path)

    if adapter:
        from peft import PeftModel
        # LoRA 어댑터에서 base_model 경로 읽기
        config_path = Path(model_path) / "category_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            base_model_name = config.get("base_model", "google/gemma-3-1b-it")
        else:
            base_model_name = "google/gemma-3-1b-it"

        print(f"  Base 모델: {base_model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=3,
            torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=3,
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    model.to(device)
    return model, tokenizer


def classify_batch(model, tokenizer, texts, device="cpu", max_length=256, batch_size=16):
    """배치 추론"""
    all_preds = []
    all_confs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            confs = probs.max(dim=-1).values

        all_preds.extend(preds.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

    return all_preds, all_confs


def main():
    parser = argparse.ArgumentParser(description="Gemma Golden set 벤치마크")
    parser.add_argument("--model", required=True, help="모델 경로/이름")
    parser.add_argument("--adapter", action="store_true", help="LoRA 어댑터 모드")
    parser.add_argument("--golden-path", default=None, help="Golden set 경로")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None, help="결과 저장 경로")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    golden_path = Path(args.golden_path) if args.golden_path else project_root / "data" / "gold_dataset" / "v8_golden_set.json"

    # Golden set 로드
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    print("=" * 60)
    print(f"  Gemma Golden set 벤치마크")
    print(f"  모델: {args.model}")
    print(f"  Golden set: {golden_path.name} ({len(golden)}건)")
    print(f"  어댑터 모드: {args.adapter}")
    print("=" * 60)

    # 정답 분포
    y_true_labels = [item["v4_3class"] for item in golden]
    y_true = [LABEL_MAP[label] for label in y_true_labels]
    true_dist = Counter(y_true_labels)
    print(f"\n  정답 분포:")
    for t in V4_3CLASS:
        print(f"    {t}: {true_dist.get(t, 0)}건")

    # 디바이스
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"\n  Device: {device}")

    # 모델 로드
    print(f"  모델 로딩...")
    start_load = time.time()
    model, tokenizer = load_model(args.model, adapter=args.adapter, device=device)
    load_time = time.time() - start_load
    print(f"  모델 로드: {load_time:.1f}초")

    # 추론
    texts = [item["text"] for item in golden]
    start_infer = time.time()
    preds, confs = classify_batch(
        model, tokenizer, texts,
        device=device, max_length=args.max_length, batch_size=args.batch_size
    )
    infer_time = time.time() - start_infer
    print(f"  추론: {infer_time:.1f}초 ({len(golden)/infer_time:.1f}건/초)")

    # 예측 라벨
    id_to_label = {0: "대응 필요", 1: "콘텐츠·투자", 2: "노이즈"}
    y_pred_labels = [id_to_label[p] for p in preds]

    # 정확도
    correct = sum(1 for a, b in zip(y_true_labels, y_pred_labels) if a == b)
    acc = correct / len(golden)
    print(f"\n  3분류 정확도: {acc*100:.1f}% ({correct}/{len(golden)})")

    # Actionable-only (노이즈 제외)
    actionable_pairs = [(t, p) for t, p in zip(y_true_labels, y_pred_labels) if t != "노이즈"]
    if actionable_pairs:
        correct_act = sum(1 for t, p in actionable_pairs if t == p)
        acc_act = correct_act / len(actionable_pairs)
        print(f"  Actionable-only: {acc_act*100:.1f}% ({correct_act}/{len(actionable_pairs)})")

    # Classification report
    report = classification_report(
        y_true_labels, y_pred_labels, labels=V4_3CLASS, digits=4, zero_division=0
    )
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=V4_3CLASS)
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
    for i, (true, pred) in enumerate(zip(y_true_labels, y_pred_labels)):
        if true != pred:
            misclassified.append({
                "index": i,
                "text": golden[i]["text"][:100],
                "true": true,
                "pred": pred,
                "confidence": round(confs[i], 4),
            })

    print(f"\n  오분류: {len(misclassified)}건")
    for m in misclassified:
        print(f"    [{m['index']}] {m['true']} → {m['pred']} | conf={m['confidence']:.2f} | {m['text'][:60]}")

    # 오분류 패턴
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
    print(f"    KcBERT (5분류, golden 227):    59.0%")
    print(f"    Gemini Flash FT (5분류):       63.4%")
    print(f"    Haiku zero-shot (5분류):       78.9%")
    print(f"    Haiku prompt R5 (3분류):       93.9%")
    print(f"    Gemma (이번):                  {acc*100:.1f}%")
    print(f"{'='*60}")

    # 결과 저장
    p, r, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, labels=V4_3CLASS, zero_division=0
    )
    per_cat = {}
    for i, t in enumerate(V4_3CLASS):
        per_cat[t] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    output_data = {
        "experiment": "gemma_3class_golden",
        "model": args.model,
        "adapter": args.adapter,
        "taxonomy": "3분류 (대응 필요 / 콘텐츠·투자 / 노이즈)",
        "golden_set": golden_path.name,
        "golden_set_size": len(golden),
        "accuracy_3class": round(acc, 4),
        "accuracy_actionable": round(acc_act, 4) if actionable_pairs else None,
        "correct": correct,
        "baseline_comparison": {
            "kcbert_5way": 0.590,
            "gemini_flash_ft_5way": 0.634,
            "haiku_zeroshot_5way": 0.789,
            "haiku_prompt_r5_3way": 0.939,
            "gemma_this": round(acc, 4),
        },
        "per_category": per_cat,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "misclassified": misclassified,
        "error_patterns": dict(error_patterns),
        "inference_time_seconds": round(infer_time, 1),
        "model_load_time_seconds": round(load_time, 1),
    }

    # 출력 경로
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "benchmarks" / "golden_benchmark_gemma_3class.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output_path}")


if __name__ == "__main__":
    main()
