"""2축 분류기 파인튜닝 스크립트

Topic 모델과 Sentiment 모델을 각각 파인튜닝합니다.
모델은 models/two_axis/{topic,sentiment}/ 에 저장됩니다.

사용법:
    # 둘 다 훈련
    python3 scripts/train_two_axis.py

    # Topic만
    python3 scripts/train_two_axis.py --axis topic

    # Sentiment만
    python3 scripts/train_two_axis.py --axis sentiment
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
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_axis(axis: str, data_dir: Path, output_dir: Path, epochs: int, batch_size: int, lr: float):
    """특정 축(topic/sentiment) 모델 훈련"""
    axis_data_dir = data_dir / axis
    axis_output_dir = output_dir / axis

    print(f"\n{'='*60}")
    print(f"  {axis.upper()} 분류기 파인튜닝")
    print(f"{'='*60}")

    # 카테고리 매핑 로드
    with open(axis_data_dir / "category_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)

    category_to_id = mapping["category_to_id"]
    id_to_category = {int(k): v for k, v in mapping["id_to_category"].items()}
    num_labels = mapping["num_labels"]

    print(f"  카테고리 수: {num_labels}")
    for cid, cname in sorted(id_to_category.items()):
        print(f"    {cid}: {cname}")

    # 데이터 로드
    with open(axis_data_dir / "train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(axis_data_dir / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(axis_data_dir / "test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"\n  Train: {len(train_data)}건, Val: {len(val_data)}건, Test: {len(test_data)}건")

    # 토크나이저 & 모델
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

    # Dataset
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

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=str(axis_output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=str(axis_output_dir / "logs"),
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

    # 훈련
    print(f"\n  훈련 시작...")
    trainer.train()

    # 테스트 평가
    print(f"\n  테스트 데이터 평가...")
    test_results = trainer.evaluate(test_dataset)
    print(f"  Test Results:")
    for key, value in test_results.items():
        if key.startswith("eval_"):
            print(f"    {key}: {value:.4f}")

    # 상세 classification report
    test_preds = trainer.predict(test_dataset)
    preds = test_preds.predictions.argmax(-1)
    labels = test_preds.label_ids
    target_names = [id_to_category[i] for i in range(num_labels)]
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    print(f"\n  Classification Report:\n{report}")

    # 모델 저장
    final_dir = axis_output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # 카테고리 매핑도 모델 디렉토리에 복사
    with open(final_dir / "category_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "axis": axis,
            "category_to_id": category_to_id,
            "id_to_category": id_to_category,
        }, f, ensure_ascii=False, indent=2)

    # 결과 저장
    results_file = axis_output_dir / "test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_metrics": {k: float(v) for k, v in test_results.items()},
            "classification_report": report,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  모델 저장: {final_dir}")
    print(f"  결과 저장: {results_file}")

    return test_results


def main():
    parser = argparse.ArgumentParser(description="2축 분류기 파인튜닝")
    parser.add_argument("--axis", type=str, default="both", choices=["topic", "sentiment", "both"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "training_data" / "two_axis"
    output_dir = project_root / "models" / "two_axis"
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = ["topic", "sentiment"] if args.axis == "both" else [args.axis]

    for axis in axes:
        train_axis(axis, data_dir, output_dir, args.epochs, args.batch_size, args.lr)

    print("\n\n모든 훈련 완료!")


if __name__ == "__main__":
    main()
