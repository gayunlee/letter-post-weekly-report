#!/usr/bin/env python3
"""GPT-4o-mini 파인튜닝 실행

사용법:
    python scripts/run_finetune.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    project_root = Path(__file__).parent.parent
    train_path = project_root / "data" / "finetune" / "train.jsonl"
    val_path = project_root / "data" / "finetune" / "validation.jsonl"

    print("=" * 60)
    print("GPT-4o-mini 파인튜닝 실행")
    print("=" * 60)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. 파일 업로드
    print("\n1. 파일 업로드")

    with open(train_path, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    print(f"  훈련 파일: {train_file.id}")

    with open(val_path, "rb") as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    print(f"  검증 파일: {val_file.id}")

    # 2. 파인튜닝 작업 생성
    print("\n2. 파인튜닝 작업 생성")
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model="gpt-4o-mini-2024-07-18",
        suffix="voc-classifier",
        hyperparameters={
            "n_epochs": 3,
        }
    )

    print(f"  작업 ID: {job.id}")
    print(f"  상태: {job.status}")
    print(f"\n파인튜닝이 시작되었습니다.")
    print(f"진행 상황 확인: python scripts/check_finetune.py {job.id}")


if __name__ == "__main__":
    main()
