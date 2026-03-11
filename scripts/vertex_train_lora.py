"""
Vertex AI Custom Training — LoRA SFT for EXAONE / Solar

Vertex AI A100에서 실행되는 학습 스크립트.
submit_vertex_jobs.py가 이 스크립트를 Vertex AI에 제출함.
"""

import argparse
import json
import os
import subprocess
import sys

# 모듈 로딩 전에 의존성 업그레이드 (numpy 호환성 유지)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--upgrade",
    "transformers==4.54.0", "numpy<2.0", "scipy>=1.11.0",
    "peft>=0.12.0", "trl>=0.9.0",
    "datasets>=2.20.0", "bitsandbytes>=0.43.0", "accelerate>=0.33.0",
])

import transformers
print(f"transformers version: {transformers.__version__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="HuggingFace model ID")
    parser.add_argument("--train-data", required=True, help="GCS path to sft_train.jsonl")
    parser.add_argument("--val-data", required=True, help="GCS path to sft_val.jsonl")
    parser.add_argument("--output-dir", required=True, help="GCS path for output model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from google.cloud import storage

    print("=" * 60)
    print(f"  LoRA SFT Training")
    print(f"  Model: {args.model_name}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print("=" * 60)

    # --- 1. GCS에서 데이터 로드 ---
    def load_jsonl_from_gcs(gcs_path):
        """gs://bucket/path/file.jsonl → list[dict]"""
        path = gcs_path.replace("gs://", "")
        bucket_name = path.split("/")[0]
        blob_path = "/".join(path.split("/")[1:])
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        content = blob.download_as_text()
        return [json.loads(line) for line in content.strip().split("\n")]

    print("\n데이터 로드 중...")
    train_raw = load_jsonl_from_gcs(args.train_data)
    val_raw = load_jsonl_from_gcs(args.val_data)
    print(f"  Train: {len(train_raw)}건, Val: {len(val_raw)}건")

    # SFT 포맷 → 프롬프트 문자열로 변환
    def format_for_sft(item):
        return {
            "text": f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        }

    train_data = Dataset.from_list([format_for_sft(item) for item in train_raw])
    val_data = Dataset.from_list([format_for_sft(item) for item in val_raw])

    # --- 2. 모델 + 토크나이저 로드 (4bit 양자화) ---
    print(f"\n모델 로드 중: {args.model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"  모델 로드 완료. 파라미터: {model.num_parameters() / 1e9:.1f}B")

    # --- 3. LoRA 설정 ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  LoRA 학습 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # --- 4. 학습 ---
    local_output = "/tmp/lora_output"

    training_config = SFTConfig(
        output_dir=local_output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    print(f"\n학습 시작 (epochs={args.epochs}, batch={args.batch_size}, lr={args.learning_rate})")
    trainer.train()

    # --- 5. 결과 저장 → GCS ---
    print(f"\n모델 저장 중: {local_output}")
    trainer.save_model(local_output)
    tokenizer.save_pretrained(local_output)

    # GCS 업로드
    print(f"GCS 업로드 중: {args.output_dir}")
    output_path = args.output_dir.replace("gs://", "")
    bucket_name = output_path.split("/")[0]
    prefix = "/".join(output_path.split("/")[1:])

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_output):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_output)
            blob_path = f"{prefix}/{rel_path}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"  업로드: {blob_path}")

    print("\n학습 완료!")


if __name__ == "__main__":
    main()
