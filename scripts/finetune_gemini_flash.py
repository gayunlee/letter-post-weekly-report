"""
Gemini 2.5 Flash 파인튜닝 — Vertex AI Supervised Tuning

사용법:
  python3 scripts/finetune_gemini_flash.py

GCS 경로:
  gs://us-service-data-vertex-ft/voc-classification/gemini_ft_train.jsonl
  gs://us-service-data-vertex-ft/voc-classification/gemini_ft_val.jsonl
"""

import vertexai
from vertexai.tuning import sft

PROJECT_ID = "us-service-data"
LOCATION = "us-central1"
BUCKET = "gs://us-service-data-vertex-ft/voc-classification"


def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    print("=" * 60)
    print("  Gemini Flash 파인튜닝 시작")
    print("=" * 60)
    print(f"  프로젝트: {PROJECT_ID}")
    print(f"  리전: {LOCATION}")
    print(f"  학습 데이터: {BUCKET}/gemini_ft_train.jsonl")
    print(f"  검증 데이터: {BUCKET}/gemini_ft_val.jsonl")
    print()

    tuning_job = sft.train(
        source_model="gemini-2.0-flash-001",
        train_dataset=f"{BUCKET}/gemini_ft_train.jsonl",
        validation_dataset=f"{BUCKET}/gemini_ft_val.jsonl",
        tuned_model_display_name="voc-classifier-gemini-flash",
        epochs=3,
    )

    print(f"  튜닝 Job 생성됨: {tuning_job.resource_name}")
    print(f"  상태 확인: https://console.cloud.google.com/vertex-ai/generative/language/tuning?project={PROJECT_ID}")
    print()
    print("  완료까지 수십 분~수 시간 소요. 콘솔에서 상태 확인 가능.")
    print("  완료 후 엔드포인트 이름으로 API 호출 가능.")


if __name__ == "__main__":
    main()
