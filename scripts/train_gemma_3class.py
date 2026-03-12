"""Gemma LoRA 3분류 파인튜닝 스크립트

Gemma (1B/4B) + LoRA로 3분류(대응 필요/콘텐츠·투자/노이즈) 파인튜닝.
Mac MPS 48GB 환경 기준.

사용법:
    # 소량 테스트 (1,000건, 1 epoch)
    python3 scripts/train_gemma_3class.py --max-train-samples 1000 --epochs 1

    # 전량 학습 (3 epoch)
    python3 scripts/train_gemma_3class.py --epochs 3

    # 모델 변경
    python3 scripts/train_gemma_3class.py --model google/gemma-2-2b-it --epochs 3
"""
import json
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight


DEFAULT_MODEL = "google/gemma-3-1b-it"
MAX_LENGTH = 256


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": item["label"],
        }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Gemma LoRA 3분류 파인튜닝")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="모델 이름/경로")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="학습 데이터 제한 (소량 실험용)")
    parser.add_argument("--data", default=None, help="데이터 디렉토리")
    parser.add_argument("--output", default=None, help="출력 디렉토리")
    parser.add_argument("--no-weighted-loss", action="store_true",
                        help="클래스 가중치 비활성화")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = Path(args.data) if args.data else project_root / "data" / "training_data" / "v3" / "gemma_3class"

    # 출력 디렉토리: 소량 실험이면 별도 폴더
    if args.output:
        output_dir = Path(args.output)
    elif args.max_train_samples:
        output_dir = project_root / "models" / "v4" / f"gemma_3class_{args.max_train_samples}"
    else:
        output_dir = project_root / "models" / "v4" / "gemma_3class"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Gemma LoRA 3분류 파인튜닝")
    print("=" * 60)

    # 카테고리 매핑
    with open(data_dir / "category_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    id_to_category = {int(k): v for k, v in mapping["id_to_category"].items()}
    num_labels = mapping["num_labels"]

    print(f"\n  카테고리: {num_labels}개")
    for cid, cname in sorted(id_to_category.items()):
        print(f"    {cid}: {cname}")

    # 데이터 로드
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(data_dir / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)

    # 소량 실험: 학습 데이터 제한
    if args.max_train_samples and args.max_train_samples < len(train_data):
        import random
        random.seed(42)
        train_data = random.sample(train_data, args.max_train_samples)
        print(f"\n  소량 실험: {args.max_train_samples}건으로 제한")

    print(f"\n  Train: {len(train_data)}, Val: {len(val_data)}")

    # Golden set 데이터 누수 검사
    golden_dir = project_root / "data" / "gold_dataset"
    golden_texts = set()
    if golden_dir.exists():
        for gf in sorted(golden_dir.glob("*_golden_set.json")):
            with open(gf, encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                golden_texts.add(item["text"].strip())
            print(f"    Golden: {gf.name} ({len(items)}건)")

    if golden_texts:
        leaks = {"train": 0, "val": 0}
        for name, dataset in [("train", train_data), ("val", val_data)]:
            for item in dataset:
                if item["text"].strip() in golden_texts:
                    leaks[name] += 1
        total_leaks = sum(leaks.values())
        if total_leaks > 0:
            raise ValueError(
                f"Golden set과 학습 데이터가 {total_leaks}건 중복됩니다. "
                f"prepare_gemma_ft_data.py를 다시 실행하세요."
            )
        print(f"  Golden set 누수 검사: 통과 (0건)")

    # Train 분포
    train_dist = Counter(item["label"] for item in train_data)
    print(f"\n  Train 분포:")
    for lid in sorted(train_dist.keys()):
        count = train_dist[lid]
        pct = count / len(train_data) * 100
        print(f"    {lid} ({id_to_category[lid]}): {count}건 ({pct:.1f}%)")

    # 디바이스
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"\n  Device: {device}")

    # 토크나이저
    print(f"  모델: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 모델 + LoRA
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        torch_dtype=torch.float32,
        id2label=id_to_category,
        label2id=mapping["category_to_id"],
    )
    # pad_token_id 설정
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  학습 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Datasets
    train_dataset = TextDataset(train_data, tokenizer, args.max_length)
    val_dataset = TextDataset(val_data, tokenizer, args.max_length)

    # 클래스 가중치
    class_weights_tensor = None
    if not args.no_weighted_loss:
        train_labels = [item["label"] for item in train_data]
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels,
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        print(f"\n  클래스 가중치:")
        for i, w in enumerate(class_weights_tensor):
            print(f"    {id_to_category[i]}: {w:.3f}")

    # 훈련 설정
    gradient_accumulation = max(1, 16 // args.batch_size)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        use_mps_device=(device == "mps"),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\n  훈련 시작 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, max_len={args.max_length})")
    trainer.train()

    # Val 평가
    print(f"\n  Val 평가...")
    val_results = trainer.evaluate(val_dataset)
    print(f"  Val metrics:")
    for key, value in val_results.items():
        if key.startswith("eval_"):
            print(f"    {key}: {value:.4f}")

    # Val classification report
    val_preds = trainer.predict(val_dataset)
    preds = val_preds.predictions.argmax(-1)
    labels = val_preds.label_ids
    target_names = [id_to_category[i] for i in range(num_labels)]
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("  Confusion Matrix:")
    header = f"  {'':>14}"
    for name in target_names:
        header += f"  {name[:6]:>6}"
    print(header)
    for i, row in enumerate(cm):
        line = f"  {target_names[i]:>14}"
        for val in row:
            line += f"  {val:>6}"
        print(line)

    # LoRA 어댑터 + 토크나이저 저장
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    # LoRA 어댑터만 저장 (가벼움)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Merged 모델도 저장 (추론 편의)
    merged_dir = output_dir / "merged_model"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    # 카테고리 설정 저장
    for save_dir in [final_dir, merged_dir]:
        with open(save_dir / "category_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "axis": "topic",
                "version": "v4-3class-gemma-lora",
                "base_model": args.model,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "category_to_id": mapping["category_to_id"],
                "id_to_category": {str(k): v for k, v in id_to_category.items()},
            }, f, ensure_ascii=False, indent=2)

    # 결과 저장
    results_file = output_dir / "train_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v4-3class-gemma-lora",
            "base_model": args.model,
            "lora_config": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": ["q_proj", "v_proj"],
            },
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_length": args.max_length,
                "max_train_samples": args.max_train_samples,
                "weighted_loss": not args.no_weighted_loss,
            },
            "val_metrics": {k: float(v) for k, v in val_results.items()},
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "category_names": target_names,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "trainable_params": trainable,
            "total_params": total,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  LoRA 어댑터: {final_dir}")
    print(f"  Merged 모델: {merged_dir}")
    print(f"  결과: {results_file}")

    val_acc = val_results.get("eval_accuracy", 0)
    print(f"\n{'='*60}")
    print(f"  Val 정확도: {val_acc*100:.1f}%")
    print(f"  → benchmark_gemma_golden.py로 Golden set 벤치마크 필요")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
