"""
EXAONE 7.8B 균형 데이터 파인튜닝 제출 — Vertex AI

2종 균형 데이터로 각각 EXAONE 7.8B LoRA 학습:
  1. balanced_strict (1:1 균형, ~1,500건)
  2. balanced_moderate (적당히 균형, ~5,300건)

사용법:
  python3 scripts/submit_balanced_jobs.py --variant strict
  python3 scripts/submit_balanced_jobs.py --variant moderate
  python3 scripts/submit_balanced_jobs.py --variant both
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.config/gcloud/legacy_credentials/gayunlee11@us-all.co.kr/adc.json"),
)

from google.cloud import aiplatform

PROJECT_ID = "us-service-data"
LOCATION = "us-east1"
BUCKET = "gs://us-service-data-vertex-ft-us"
STAGING_BUCKET = BUCKET

CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "vertex_train_lora.py")

HF_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

VARIANTS = {
    "strict": {
        "display_name": "voc-exaone-7.8b-balanced-strict",
        "train_data": f"{BUCKET}/data/sft_balanced_strict_train.jsonl",
        "val_data": f"{BUCKET}/data/sft_balanced_strict_val.jsonl",
        "output_dir": f"{BUCKET}/models/exaone_balanced_strict",
    },
    "moderate": {
        "display_name": "voc-exaone-7.8b-balanced-moderate",
        "train_data": f"{BUCKET}/data/sft_balanced_moderate_train.jsonl",
        "val_data": f"{BUCKET}/data/sft_balanced_moderate_val.jsonl",
        "output_dir": f"{BUCKET}/models/exaone_balanced_moderate",
    },
}


def submit_job(variant: str):
    config = VARIANTS[variant]

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

    args = [
        "--model-name", HF_MODEL,
        "--train-data", config["train_data"],
        "--val-data", config["val_data"],
        "--output-dir", config["output_dir"],
        "--epochs", "3",
        "--batch-size", "4",
        "--learning-rate", "2e-4",
        "--lora-r", "16",
        "--lora-alpha", "32",
        "--max-seq-length", "512",
    ]

    job = aiplatform.CustomJob.from_local_script(
        display_name=config["display_name"],
        script_path=SCRIPT_PATH,
        container_uri=CONTAINER_URI,
        requirements=[
            "transformers>=4.54.0",
            "peft>=0.12.0",
            "trl>=0.9.0",
            "datasets>=2.20.0",
            "bitsandbytes>=0.43.0",
            "accelerate>=0.33.0",
            "google-cloud-storage>=2.18.0",
            "sentencepiece",
            "protobuf",
        ],
        args=args,
        machine_type="g2-standard-12",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        replica_count=1,
        boot_disk_size_gb=200,
    )

    print(f"\n{'=' * 60}")
    print(f"  {config['display_name']} 제출")
    print(f"  모델: {HF_MODEL}")
    print(f"  학습 데이터: {config['train_data']}")
    print(f"  출력: {config['output_dir']}")
    print(f"{'=' * 60}")

    job.submit()

    print(f"\n  Job ID: {job.resource_name}")
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["strict", "moderate", "both"], required=True)
    args = parser.parse_args()

    if args.variant == "both":
        for v in ["strict", "moderate"]:
            submit_job(v)
    else:
        submit_job(args.variant)

    print("\n완료. check_vertex_jobs.py로 상태를 확인하세요.")


if __name__ == "__main__":
    main()
