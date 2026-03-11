"""v3 Topic 분류기 파인튜닝 스크립트

LLM 라벨링된 v3 데이터로 KcBERT를 파인튜닝합니다.
기존 train_two_axis.py와 동일 구조, v3 경로만 변경.

사용법:
    python3 scripts/train_v3_topic.py
    python3 scripts/train_v3_topic.py --epochs 8 --lr 3e-5
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


MODEL_NAME = "klue/roberta-base"
MAX_LENGTH = 128


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
    def __init__(self, data: List[Dict], tokenizer, max_length: int = MAX_LENGTH):
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="v3 Topic 파인튜닝")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "training_data" / "v3" / "topic"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / "v3" / "topic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  v3 Topic 분류기 파인튜닝")
    print("=" * 60)

    # 카테고리 매핑
    with open(data_dir / "category_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)

    category_to_id = mapping["category_to_id"]
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
    with open(data_dir / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"\n  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Golden set 데이터 누수 검사 (gold_dataset/ 내 모든 golden set 파일)
    golden_dir = project_root / "data" / "gold_dataset"
    golden_texts = set()
    golden_file_count = 0
    if golden_dir.exists():
        for gf in sorted(golden_dir.glob("*_golden_set.json")):
            with open(gf, encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                golden_texts.add(item["text"].strip())
            golden_file_count += 1
            print(f"    {gf.name}: {len(items)}건")

    if golden_texts:
        print(f"  Golden set 누수 검사: {len(golden_texts)}건 대상 ({golden_file_count}개 파일)")

        leaks = {"train": 0, "val": 0, "test": 0}
        for name, dataset in [("train", train_data), ("val", val_data), ("test", test_data)]:
            for item in dataset:
                if item["text"].strip() in golden_texts:
                    leaks[name] += 1

        total_leaks = sum(leaks.values())
        if total_leaks > 0:
            print(f"\n  ⚠ Golden set 데이터 누수 감지!")
            print(f"    Golden {len(golden_texts)}건 중 학습 데이터와 중복:")
            for name, count in leaks.items():
                if count > 0:
                    print(f"      {name}: {count}건")
            raise ValueError(
                f"Golden set과 학습 데이터가 {total_leaks}건 중복됩니다. "
                f"prepare_v3_training_data.py --golden-dir로 golden set을 제외한 데이터로 split하세요."
            )
        else:
            print(f"  Golden set 누수 검사: 통과 (중복 0건)")
    else:
        print(f"  Golden set 누수 검사: golden set 파일 없음 ({golden_dir})")

    # 모델 & 토크나이저
    print(f"  모델: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id_to_category,
        label2id=category_to_id,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")

    # Datasets
    train_dataset = TextDataset(train_data, tokenizer)
    val_dataset = TextDataset(val_data, tokenizer)
    test_dataset = TextDataset(test_data, tokenizer)

    # 클래스 가중치
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

    # 훈련
    gradient_accumulation = max(1, 16 // args.batch_size)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=args.lr,
        warmup_steps=50,
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
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
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

    print(f"\n  훈련 시작 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})")
    trainer.train()

    # 테스트 평가
    print(f"\n  테스트 평가...")
    test_results = trainer.evaluate(test_dataset)
    print(f"  Test metrics:")
    for key, value in test_results.items():
        if key.startswith("eval_"):
            print(f"    {key}: {value:.4f}")

    # Classification report
    test_preds = trainer.predict(test_dataset)
    preds = test_preds.predictions.argmax(-1)
    labels = test_preds.label_ids
    target_names = [id_to_category[i] for i in range(num_labels)]
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    print(f"\n{report}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    print("  Confusion Matrix:")
    print(f"  {'':>14}", end="")
    for name in target_names:
        print(f"  {name[:6]:>6}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:>14}", end="")
        for val in row:
            print(f"  {val:>6}", end="")
        print()

    # 모델 저장
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with open(final_dir / "category_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "axis": "topic",
            "version": "v3-5cat",
            "category_to_id": category_to_id,
            "id_to_category": {str(k): v for k, v in id_to_category.items()},
        }, f, ensure_ascii=False, indent=2)

    # 결과 저장
    results_file = output_dir / "test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v3-5cat",
            "test_metrics": {k: float(v) for k, v in test_results.items()},
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "category_names": target_names,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "v2_topic_accuracy": 0.693,  # 비교 기준
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  모델 저장: {final_dir}")
    print(f"  결과 저장: {results_file}")

    # v2 대비 비교
    v3_acc = test_results.get("eval_accuracy", 0)
    print(f"\n{'='*60}")
    print(f"  v2 Topic 정확도: 69.3%")
    print(f"  v3 Topic 정확도: {v3_acc*100:.1f}%")
    diff = (v3_acc - 0.693) * 100
    print(f"  차이: {diff:+.1f}%p")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
