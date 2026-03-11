"""
Vertex AI Job 자동 파이프라인 — 15분마다 확인, 실패 시 재시도, 성공 시 다음 단계 자동 진행

흐름:
  1. balanced_strict 학습 대기
  2. strict 성공 → balanced_moderate 자동 제출
  3. moderate 성공 → 벤치마크 준비 완료 로깅
  4. 실패 시 → 에러 기록 + 최대 2회 재시도 (다른 리전/설정)

로그: logs/vertex_monitor.log
"""

import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.config/gcloud/legacy_credentials/gayunlee11@us-all.co.kr/adc.json"),
)

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "vertex_monitor.log"
STATE_FILE = LOG_DIR / "pipeline_state.json"

CHECK_INTERVAL = 15 * 60  # 15분

PROJECT_ID = "us-service-data"
BUCKET = "gs://us-service-data-vertex-ft-us"
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
SCRIPT_PATH = str(BASE_DIR / "scripts" / "vertex_train_lora.py")
HF_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# 리전 순서 — 실패 시 다음 리전으로
REGIONS = ["us-east1", "us-central1", "europe-west4"]

# GPU 설정 순서 — 실패 시 다음 설정으로
GPU_CONFIGS = [
    {"machine_type": "g2-standard-12", "accelerator_type": "NVIDIA_L4"},
    {"machine_type": "a2-highgpu-1g", "accelerator_type": "NVIDIA_TESLA_A100"},
]

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

STATE_LABELS = {
    1: "QUEUED", 2: "PENDING", 3: "RUNNING",
    4: "SUCCEEDED", 5: "FAILED", 6: "CANCELLING",
    7: "CANCELLED", 8: "PAUSED", 9: "EXPIRED",
}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def check_job(job_id: str, location: str) -> tuple[int, str]:
    """Job 상태 확인. (state_code, error_message)"""
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=location)
    job = aiplatform.CustomJob.get(job_id)
    state = int(job.state)
    error = ""
    if state == 5 and job.error:
        error = job.error.message
    return state, error


def submit_variant(variant: str, location: str, gpu_config: dict) -> str:
    """variant 제출, job_id 반환"""
    from google.cloud import aiplatform

    config = VARIANTS[variant]
    aiplatform.init(project=PROJECT_ID, location=location, staging_bucket=BUCKET)

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
            "peft>=0.12.0", "trl>=0.9.0", "datasets>=2.20.0",
            "bitsandbytes>=0.43.0", "accelerate>=0.33.0",
            "google-cloud-storage>=2.18.0", "sentencepiece", "protobuf",
        ],
        args=args,
        machine_type=gpu_config["machine_type"],
        accelerator_type=gpu_config["accelerator_type"],
        accelerator_count=1,
        replica_count=1,
        boot_disk_size_gb=200,
    )
    job.submit()
    return job.resource_name


def is_infra_error(error_msg: str) -> bool:
    """인프라/환경 에러인지 판별 (재시도 가능)"""
    infra_keywords = [
        "quota", "resource", "capacity", "unavailable",
        "timeout", "internal", "RESOURCE_EXHAUSTED",
    ]
    lower = error_msg.lower()
    return any(kw in lower for kw in infra_keywords)


def handle_variant(variant: str, state: dict) -> str:
    """
    variant 하나를 끝까지 처리. 상태 반환: 'succeeded' | 'failed' | 'waiting'
    """
    key = f"{variant}_job_id"
    loc_key = f"{variant}_location"
    retry_key = f"{variant}_retry_count"
    gpu_key = f"{variant}_gpu_idx"

    # 아직 제출 안 됨 → 제출
    if key not in state:
        location = REGIONS[0]
        gpu_idx = 0
        gpu_config = GPU_CONFIGS[gpu_idx]
        log(f"[{variant}] 제출: {location}, {gpu_config['accelerator_type']}")
        try:
            job_id = submit_variant(variant, location, gpu_config)
            state[key] = job_id
            state[loc_key] = location
            state[retry_key] = 0
            state[gpu_key] = gpu_idx
            save_state(state)
            log(f"[{variant}] Job ID: {job_id}")
        except Exception as e:
            log(f"[{variant}] 제출 실패: {str(e)[:200]}")
        return "waiting"

    # 이미 제출됨 → 상태 확인
    job_id = state[key]
    location = state[loc_key]
    retries = state.get(retry_key, 0)
    gpu_idx = state.get(gpu_key, 0)

    try:
        job_state, error = check_job(job_id, location)
    except Exception as e:
        log(f"[{variant}] 상태 확인 실패: {str(e)[:150]}")
        return "waiting"

    label = STATE_LABELS.get(job_state, f"UNKNOWN({job_state})")
    log(f"[{variant}] 상태: {label} (location={location}, retry={retries})")

    if job_state == 4:  # SUCCEEDED
        log(f"[{variant}] ✅ 학습 완료!")
        state[f"{variant}_done"] = True
        save_state(state)
        return "succeeded"

    if job_state == 5:  # FAILED
        log(f"[{variant}] ❌ 실패: {error[:300]}")

        if retries >= 3:
            log(f"[{variant}] 재시도 한도 초과 (3회). 중단.")
            state[f"{variant}_final_fail"] = True
            save_state(state)
            return "failed"

        # 재시도 전략 결정
        retries += 1
        state[retry_key] = retries

        if is_infra_error(error):
            # 인프라 에러 → 다른 리전 시도
            region_idx = retries % len(REGIONS)
            location = REGIONS[region_idx]
            log(f"[{variant}] 인프라 에러 → 리전 변경: {location}")
        else:
            # 코드/의존성 에러 → 같은 리전, 다른 GPU 시도
            gpu_idx = (gpu_idx + 1) % len(GPU_CONFIGS)
            state[gpu_key] = gpu_idx
            log(f"[{variant}] 코드 에러 → GPU 변경: {GPU_CONFIGS[gpu_idx]['accelerator_type']}")

        gpu_config = GPU_CONFIGS[gpu_idx]
        try:
            new_job_id = submit_variant(variant, location, gpu_config)
            state[key] = new_job_id
            state[loc_key] = location
            save_state(state)
            log(f"[{variant}] 재제출 완료: {new_job_id}")
        except Exception as e:
            log(f"[{variant}] 재제출 실패: {str(e)[:200]}")

        return "waiting"

    # PENDING / RUNNING / QUEUED → 대기
    return "waiting"


def main():
    log("=" * 60)
    log("EXAONE 균형 학습 자동 파이프라인 시작 (15분 간격)")
    log("흐름: strict 학습 → moderate 학습 → 완료 보고")
    log("=" * 60)

    state = load_state()

    # 기존 strict job이 이미 있으면 이어서
    if "strict_job_id" not in state:
        state["strict_job_id"] = "projects/306496779169/locations/us-east1/customJobs/1184883826589958144"
        state["strict_location"] = "us-east1"
        state["strict_retry_count"] = 0
        state["strict_gpu_idx"] = 0
        save_state(state)

    while True:
        log("-" * 40)

        # Phase 1: strict
        if not state.get("strict_done") and not state.get("strict_final_fail"):
            result = handle_variant("strict", state)
            if result == "waiting":
                log(f"다음 확인: 15분 후")
                time.sleep(CHECK_INTERVAL)
                continue

        # strict 끝남 (성공 or 최종 실패)
        if state.get("strict_final_fail"):
            log("=" * 60)
            log("⛔ strict 최종 실패. 파이프라인 중단.")
            log("에러 로그를 확인하고 수동으로 대응 필요.")
            log("=" * 60)
            break

        # Phase 2: moderate (strict 성공 후)
        if state.get("strict_done") and not state.get("moderate_done") and not state.get("moderate_final_fail"):
            # moderate를 strict과 같은 성공 설정으로 제출
            if "moderate_job_id" not in state:
                log("strict 성공! → moderate 자동 제출")
                # strict이 성공한 리전/GPU 설정 재사용
                state["moderate_location_hint"] = state.get("strict_location", "us-east1")
                state["moderate_gpu_hint"] = state.get("strict_gpu_idx", 0)

            result = handle_variant("moderate", state)
            if result == "waiting":
                log(f"다음 확인: 15분 후")
                time.sleep(CHECK_INTERVAL)
                continue

        if state.get("moderate_final_fail"):
            log("=" * 60)
            log("⛔ moderate 최종 실패. strict은 성공.")
            log("strict 모델로 벤치마크 진행 가능.")
            log("=" * 60)
            break

        # Phase 3: 둘 다 완료
        if state.get("strict_done") and state.get("moderate_done"):
            log("=" * 60)
            log("🎉 strict + moderate 모두 학습 완료!")
            log("")
            log("다음 단계: 벤치마크")
            log("  1. Golden 227건으로 strict vs moderate 비교")
            log("  2. python3 scripts/benchmark_vertex_models.py 실행")
            log("")
            log("모델 위치:")
            log(f"  strict:   {VARIANTS['strict']['output_dir']}")
            log(f"  moderate: {VARIANTS['moderate']['output_dir']}")
            log("=" * 60)
            break

        # 예상치 못한 상태
        log("알 수 없는 상태. 중단.")
        break


if __name__ == "__main__":
    main()
