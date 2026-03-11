"""
Vertex AI Job 상태 확인

사용법: python3 scripts/check_vertex_jobs.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.config/gcloud/legacy_credentials/gayunlee11@us-all.co.kr/adc.json"),
)

from google.cloud import aiplatform
import vertexai
from vertexai.tuning import sft

PROJECT_ID = "us-service-data"
LOCATION = "us-central1"

JOBS = {
    "Gemini 2.5 Flash FT": {
        "type": "tuning",
        "id": "projects/306496779169/locations/us-central1/tuningJobs/3338612202419519488",
    },
    "EXAONE 불균형 (FAILED)": {
        "type": "custom",
        "id": "projects/306496779169/locations/us-central1/customJobs/5188325011873595392",
    },
    "EXAONE balanced_strict (L4 v3)": {
        "type": "custom",
        "id": "projects/306496779169/locations/us-central1/customJobs/8772134784097845248",
    },
}

STATE_LABELS = {
    1: "QUEUED",
    2: "PENDING",
    3: "RUNNING",
    4: "SUCCEEDED ✅",
    5: "FAILED ❌",
    6: "CANCELLING",
    7: "CANCELLED",
    8: "PAUSED",
    9: "EXPIRED",
}


def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    print("=" * 60)
    print("  Vertex AI Job 상태")
    print("=" * 60)

    for name, info in JOBS.items():
        try:
            if info["type"] == "tuning":
                job = sft.SupervisedTuningJob(info["id"])
                state = int(job.state)
                label = STATE_LABELS.get(state, f"UNKNOWN({state})")
                print(f"\n  [{name}]")
                print(f"    상태: {label}")
                if hasattr(job, "tuned_model_endpoint_name") and job.tuned_model_endpoint_name:
                    print(f"    엔드포인트: {job.tuned_model_endpoint_name}")
                if hasattr(job, "tuned_model_name") and job.tuned_model_name:
                    print(f"    모델 이름: {job.tuned_model_name}")
            else:
                job = aiplatform.CustomJob.get(info["id"])
                state = int(job.state)
                label = STATE_LABELS.get(state, f"UNKNOWN({state})")
                print(f"\n  [{name}]")
                print(f"    상태: {label}")
                if state == 5 and job.error:
                    print(f"    에러: {job.error.message[:200]}")
        except Exception as e:
            print(f"\n  [{name}]")
            print(f"    에러: {str(e)[:200]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
