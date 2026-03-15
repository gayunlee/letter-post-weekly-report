"""채널톡 3분류 KcELECTRA 파인튜닝 스크립트

Opus 라벨링된 데이터로 KcELECTRA를 파인튜닝합니다.
3분류: 결제·구독 / 콘텐츠·수강 / 기술·오류 (기타는 노이즈로 제외)

사용법:
    python3 scripts/train_channel_4class.py --data data/channel_io/training_data/labeled_1500_v4.json
    python3 scripts/train_channel_4class.py --epochs 12 --lr 3e-5 --max-length 512
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np


DEFAULT_MODEL = "beomi/KcELECTRA-base-v2022"
MAX_LENGTH = 256  # CS 텍스트가 길 수 있으므로 128→256

CATEGORY_TO_ID = {
    "결제·구독": 0,
    "콘텐츠·수강": 1,
    "기술·오류": 2,
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}

WORKFLOW_NOISE = [
    "그 외 기타 문의(오류/구독해지/환불)",
    "그 외 기타 문의",
    "💬1:1 상담 문의하기",
    "💬 1:1 고객센터 문의하기",
    "💬상담매니저에게 직접 문의",
    "어스 구독신청/결제하기",
    "어스 이용방법",
    "사이트 및 동영상 오류",
    "수강 및 상품문의",
    "라이브 콘텐츠 참여 방법",
    "수강방법",
    "↩ 이전으로",
    "✅ 1:1 문의하기",
    "구독 상품변경/결제정보 확인",
    "결제실패 후 카드변경 방법",
    "💬 구독 결제/변경/정보 확인 직접 문의하기",
    "구독상품 변경",
    "구독 결제/변경/정보 직접 문의하기",
]


def strip_workflow_buttons(text: str) -> str:
    """워크플로우 버튼 텍스트 제거."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in WORKFLOW_NOISE:
            continue
        if stripped.startswith("👆🏻"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


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


def load_labeled_data(data_path: Path) -> List[Dict]:
    """Opus 라벨링 결과 로드 + 전처리."""
    with open(data_path, encoding="utf-8") as f:
        items = json.load(f)

    processed = []
    skipped = 0
    for item in items:
        topic = item.get("topic", "")
        if topic not in CATEGORY_TO_ID:
            skipped += 1
            continue

        text = strip_workflow_buttons(item.get("text", ""))
        if len(text.strip()) < 5:
            text = item.get("text", "")  # 원본 사용

        processed.append({
            "text": text,
            "label": CATEGORY_TO_ID[topic],
            "chatId": item.get("chatId", ""),
        })

    if skipped > 0:
        print(f"  경고: {skipped}건 스킵 (알 수 없는 토픽)")

    return processed


def check_golden_leak(train_data, val_data, test_data, golden_path: Path):
    """Golden set 데이터 누수 검사."""
    if not golden_path.exists():
        print(f"  Golden set 누수 검사: 파일 없음 ({golden_path})")
        return

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    golden_chatids = set(item["chatId"] for item in golden)
    golden_texts = set(item["text"].strip() for item in golden)

    leaks = {"train": 0, "val": 0, "test": 0}
    for name, dataset in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for item in dataset:
            if item.get("chatId", "") in golden_chatids or item["text"].strip() in golden_texts:
                leaks[name] += 1

    total_leaks = sum(leaks.values())
    if total_leaks > 0:
        print(f"\n  ⚠ Golden set 데이터 누수 감지!")
        for name, count in leaks.items():
            if count > 0:
                print(f"    {name}: {count}건")
        raise ValueError(
            f"Golden set과 학습 데이터가 {total_leaks}건 중복됩니다. "
            f"Golden set chatId를 제외하세요."
        )
    else:
        print(f"  Golden set 누수 검사: 통과 (중복 0건, golden {len(golden_chatids)}건)")


def main():
    parser = argparse.ArgumentParser(description="채널톡 4분류 KcELECTRA 파인튜닝")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="사전학습 모델")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--data", default="data/channel_io/training_data/labeled_1500.json",
                        help="Opus 라벨링 데이터 경로")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data
    model_name_short = args.model.split("/")[-1].lower()
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / "channel_3class" / model_name_short
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  채널톡 3분류 KcELECTRA 파인튜닝")
    print(f"  모델: {args.model}")
    print(f"  데이터: {data_path}")
    print("=" * 60)

    # 데이터 로드
    all_data = load_labeled_data(data_path)
    print(f"\n  전체 데이터: {len(all_data)}건")

    # 분포 확인
    dist = {}
    for item in all_data:
        cat = ID_TO_CATEGORY[item["label"]]
        dist[cat] = dist.get(cat, 0) + 1
    print("  분포:")
    for cat in CATEGORY_TO_ID:
        print(f"    {cat}: {dist.get(cat, 0)}건 ({dist.get(cat, 0)/len(all_data)*100:.1f}%)")

    # Stratified split: 80/10/10
    labels = [item["label"] for item in all_data]
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=args.seed, stratify=temp_labels
    )

    print(f"\n  Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Golden set 누수 검사
    golden_path = project_root / "data" / "channel_io" / "golden" / "golden_multilabel_270.json"
    check_golden_leak(train_data, val_data, test_data, golden_path)

    # 모델 & 토크나이저
    print(f"\n  모델 로드: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(CATEGORY_TO_ID),
        id2label=ID_TO_CATEGORY,
        label2id=CATEGORY_TO_ID,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"  Device: {device}")

    # Datasets
    train_dataset = TextDataset(train_data, tokenizer, args.max_length)
    val_dataset = TextDataset(val_data, tokenizer, args.max_length)
    test_dataset = TextDataset(test_data, tokenizer, args.max_length)

    # 클래스 가중치
    train_labels_arr = [item["label"] for item in train_data]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels_arr),
        y=train_labels_arr,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"\n  클래스 가중치:")
    for i, w in enumerate(class_weights_tensor):
        print(f"    {ID_TO_CATEGORY[i]}: {w:.3f}")

    # 훈련 설정
    gradient_accumulation = max(1, 16 // args.batch_size)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=0.1,
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
        seed=args.seed,
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
    labels_arr = test_preds.label_ids
    target_names = [ID_TO_CATEGORY[i] for i in range(len(CATEGORY_TO_ID))]
    report = classification_report(labels_arr, preds, target_names=target_names, digits=4)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds)
    print("  Confusion Matrix:")
    header = "".join(f"{name:>10s}" for name in target_names)
    print(f"{'':>12s}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{val:>10d}" for val in row)
        print(f"{target_names[i]:>12s}{row_str}")

    # 모델 저장
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with open(final_dir / "category_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "task": "channel_3class",
            "model": args.model,
            "category_to_id": CATEGORY_TO_ID,
            "id_to_category": {str(k): v for k, v in ID_TO_CATEGORY.items()},
            "workflow_noise": WORKFLOW_NOISE,
        }, f, ensure_ascii=False, indent=2)

    # 결과 저장
    results = {
        "task": "channel_4class",
        "model": args.model,
        "test_metrics": {k: float(v) for k, v in test_results.items()},
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "category_names": target_names,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "total_data": len(all_data),
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_length": args.max_length,
            "seed": args.seed,
        },
    }

    results_file = output_dir / "train_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  모델 저장: {final_dir}")
    print(f"  결과 저장: {results_file}")

    # 요약
    test_acc = test_results.get("eval_accuracy", 0)
    test_f1 = test_results.get("eval_f1", 0)
    print(f"\n{'='*60}")
    print(f"  Test Accuracy: {test_acc*100:.1f}%")
    print(f"  Test F1: {test_f1*100:.1f}%")
    print(f"  비교: EXAONE 32B zero-shot 83.0% (4분류+버튼제거)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
