"""
Vertex AI Custom Training Job 제출 — EXAONE 7.8B / Solar 10.7B LoRA SFT

사용법:
  # EXAONE 7.8B
  python3 scripts/submit_vertex_jobs.py --model exaone

  # Solar 10.7B
  python3 scripts/submit_vertex_jobs.py --model solar

  # 둘 다
  python3 scripts/submit_vertex_jobs.py --model both
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
LOCATION = "us-central1"
BUCKET = "gs://us-service-data-vertex-ft-us"
DATA_PATH = f"{BUCKET}/voc-classification"
STAGING_BUCKET = BUCKET

# PyTorch GPU 컨테이너 (CUDA 12.4, Python 3.10)
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

MODELS = {
    "exaone": {
        "display_name": "voc-exaone-7.8b-lora",
        "hf_model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "machine_type": "a2-highgpu-1g",  # 1x A100 40GB
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
        "batch_size": 4,
    },
    "solar": {
        "display_name": "voc-solar-10.7b-lora",
        "hf_model": "upstage/SOLAR-10.7B-Instruct-v1.0",
        "machine_type": "a2-highgpu-1g",  # 1x A100 40GB
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
        "batch_size": 4,
    },
}

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "vertex_train_lora.py")


def submit_job(model_key: str):
    config = MODELS[model_key]

    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

    output_dir = f"{BUCKET}/models/{model_key}"

    args = [
        "--model-name", config["hf_model"],
        "--train-data", f"{DATA_PATH}/sft_train.jsonl",
        "--val-data", f"{DATA_PATH}/sft_val.jsonl",
        "--output-dir", output_dir,
        "--epochs", "3",
        "--batch-size", str(config["batch_size"]),
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
        machine_type=config["machine_type"],
        accelerator_type=config["accelerator_type"],
        accelerator_count=config["accelerator_count"],
        replica_count=1,
        boot_disk_size_gb=200,
    )

    print(f"\n{'=' * 60}")
    print(f"  {config['display_name']} 제출")
    print(f"  모델: {config['hf_model']}")
    print(f"  머신: {config['machine_type']} ({config['accelerator_type']} x{config['accelerator_count']})")
    print(f"  출력: {output_dir}")
    print(f"{'=' * 60}")

    job.submit()

    print(f"\n  Job 제출됨: {job.resource_name}")
    print(f"  콘솔: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["exaone", "solar", "both"], required=True)
    args = parser.parse_args()

    if args.model == "both":
        for key in ["exaone", "solar"]:
            submit_job(key)
    else:
        submit_job(args.model)

    print("\n완료. 콘솔에서 진행 상태를 확인하세요.")


if __name__ == "__main__":
    main()
