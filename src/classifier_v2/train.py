"""KcBERT Fine-tuning 스크립트

beomi/kcbert-base 모델을 VOC 분류 태스크에 맞게 fine-tuning합니다.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from .prepare_data import CATEGORY_TO_ID, ID_TO_CATEGORY


# 기본 설정
MODEL_NAME = "beomi/kcbert-base"
MAX_LENGTH = 256
NUM_LABELS = len(CATEGORY_TO_ID)


class VOCDataset(Dataset):
    """VOC 분류용 Dataset"""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_data(data_dir: Path) -> tuple:
    """훈련 데이터 로드"""
    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"
    test_path = data_dir / "test.json"

    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)

    return train_data, val_data, test_data


def compute_metrics(pred):
    """평가 메트릭 계산"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1
):
    """모델 훈련"""
    print("=" * 50)
    print("KcBERT Fine-tuning")
    print("=" * 50)

    # 데이터 로드
    print(f"\n데이터 로드 중: {data_dir}")
    train_data, val_data, test_data = load_data(data_dir)
    print(f"  Train: {len(train_data)}건")
    print(f"  Val: {len(val_data)}건")
    print(f"  Test: {len(test_data)}건")

    # 토크나이저 & 모델 로드
    print(f"\n모델 로드 중: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_CATEGORY,
        label2id=CATEGORY_TO_ID
    )

    # 디바이스 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset 생성
    train_dataset = VOCDataset(train_data, tokenizer)
    val_dataset = VOCDataset(val_data, tokenizer)
    test_dataset = VOCDataset(test_data, tokenizer)

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",  # wandb 등 비활성화
        fp16=False,  # MPS에서는 fp16 비활성화
        dataloader_num_workers=0,  # MPS 호환성
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 훈련
    print("\n훈련 시작...")
    trainer.train()

    # 테스트 평가
    print("\n테스트 데이터 평가...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results:")
    for key, value in test_results.items():
        if key.startswith("eval_"):
            print(f"  {key}: {value:.4f}")

    # 모델 저장
    final_model_dir = output_dir / "final_model"
    print(f"\n모델 저장 중: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # 카테고리 매핑 저장
    config_path = final_model_dir / "category_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "category_to_id": CATEGORY_TO_ID,
            "id_to_category": ID_TO_CATEGORY
        }, f, ensure_ascii=False, indent=2)

    print("\n훈련 완료!")
    print(f"모델 저장 위치: {final_model_dir}")

    return test_results


def main():
    parser = argparse.ArgumentParser(description="KcBERT Fine-tuning")
    parser.add_argument("--epochs", type=int, default=5, help="훈련 에폭 수")
    parser.add_argument("--batch-size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=2e-5, help="학습률")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "training_data"
    output_dir = project_root / "models" / "kcbert_voc_classifier"

    output_dir.mkdir(parents=True, exist_ok=True)

    train(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
