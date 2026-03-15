"""Vertex AI Custom Training — KcELECTRA 3분류 제출

사용법:
    # 데이터 업로드 + 학습 제출
    python3 scripts/submit_vertex_kcelectra.py

    # T4 대신 A100 사용
    python3 scripts/submit_vertex_kcelectra.py --gpu a100

    # 커스텀 데이터
    python3 scripts/submit_vertex_kcelectra.py --data data/channel_io/training_data/labeled_1500_v5.json
"""
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.config/gcloud/legacy_credentials/gayunlee11@us-all.co.kr/adc.json"),
)

from google.cloud import aiplatform, storage

PROJECT_ID = "us-service-data"
LOCATION = "us-central1"
BUCKET = "gs://us-service-data-vertex-ft-us"
STAGING_BUCKET = BUCKET
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "vertex_train_kcelectra.py")

GPU_CONFIGS = {
    "t4": {
        "machine_type": "n1-standard-4",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1,
        "cost_per_hour": "$0.35",
    },
    "a100": {
        "machine_type": "a2-highgpu-1g",
        "accelerator_type": "NVIDIA_TESLA_A100",
        "accelerator_count": 1,
        "cost_per_hour": "$3.67",
    },
}


def upload_file_to_gcs(local_path: str, gcs_path: str):
    """로컬 파일을 GCS에 업로드."""
    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    blob_path = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"  Uploaded: {local_path} -> {gcs_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/channel_io/training_data/labeled_1500_v5.json")
    parser.add_argument("--golden", default="data/channel_io/golden/golden_multilabel_270.json")
    parser.add_argument("--gpu", choices=["t4", "a100"], default="t4")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--version", default="v6", help="모델 버전 태그")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_local = os.path.join(project_root, args.data)
    golden_local = os.path.join(project_root, args.golden)

    if not os.path.exists(data_local):
        raise FileNotFoundError(f"데이터 파일 없음: {data_local}")

    # GCS 경로
    gcs_data_dir = f"{BUCKET}/channel-3class"
    gcs_train_data = f"{gcs_data_dir}/{os.path.basename(args.data)}"
    gcs_golden = f"{gcs_data_dir}/{os.path.basename(args.golden)}"
    gcs_output = f"{gcs_data_dir}/models/kcelectra-{args.version}"

    gpu = GPU_CONFIGS[args.gpu]

    print("=" * 60)
    print("  KcELECTRA 3분류 — Vertex AI Custom Training")
    print(f"  GPU: {args.gpu.upper()} ({gpu['cost_per_hour']}/hr)")
    print(f"  데이터: {args.data}")
    print(f"  버전: {args.version}")
    print(f"  하이퍼파라미터: epochs={args.epochs}, lr={args.lr}, max_len={args.max_length}, batch={args.batch_size}")
    print("=" * 60)

    # 1. 데이터 업로드
    print("\n[1/2] 데이터 GCS 업로드...")
    upload_file_to_gcs(data_local, gcs_train_data)
    if os.path.exists(golden_local):
        upload_file_to_gcs(golden_local, gcs_golden)

    # 2. Job 제출
    print("\n[2/2] Training job 제출...")
    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=STAGING_BUCKET,
    )

    train_args = [
        "--model", "beomi/KcELECTRA-base-v2022",
        "--train-data", gcs_train_data,
        "--golden-data", gcs_golden,
        "--output-dir", gcs_output,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--max-length", str(args.max_length),
        "--batch-size", str(args.batch_size),
    ]

    job = aiplatform.CustomJob.from_local_script(
        display_name=f"kcelectra-3class-{args.version}",
        script_path=SCRIPT_PATH,
        container_uri=CONTAINER_URI,
        requirements=[
            "transformers>=4.35.0,<4.50.0",
            "accelerate>=0.26.0",
            "scikit-learn>=1.3.0",
            "google-cloud-storage>=2.18.0",
        ],
        args=train_args,
        machine_type=gpu["machine_type"],
        accelerator_type=gpu["accelerator_type"],
        accelerator_count=gpu["accelerator_count"],
        replica_count=1,
        boot_disk_size_gb=100,
    )

    job.submit()

    print(f"\n  Job 제출 완료!")
    print(f"  Resource: {job.resource_name}")
    print(f"  모델 출력: {gcs_output}")
    print(f"  콘솔: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"\n  완료 후 모델 다운로드:")
    print(f"    gsutil -m cp -r {gcs_output}/ models/channel_3class/kcelectra-base-v2022-{args.version}/final_model/")


if __name__ == "__main__":
    main()
