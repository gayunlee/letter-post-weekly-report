"""
Vertex AI 파인튜닝 모델 Golden Set 벤치마크

사용법:
  # Gemini FT 벤치마크
  python3 scripts/benchmark_vertex_models.py --model gemini

  # EXAONE LoRA 벤치마크
  python3 scripts/benchmark_vertex_models.py --model exaone

  # Solar LoRA 벤치마크
  python3 scripts/benchmark_vertex_models.py --model solar

  # 전체 비교
  python3 scripts/benchmark_vertex_models.py --model all
"""

import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.expanduser("~/.config/gcloud/legacy_credentials/gayunlee11@us-all.co.kr/adc.json"),
)

from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "data/gold_dataset/v6_golden_set.json"
VERSION_LOG_PATH = BASE_DIR / "benchmarks/version_log.json"

V3C_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "일상·감사", "기타"]

TOPIC_ALIASES = {
    "운영": "운영 피드백", "운영피드백": "운영 피드백",
    "서비스": "서비스 피드백", "서비스피드백": "서비스 피드백",
    "콘텐츠": "콘텐츠·투자", "투자": "콘텐츠·투자",
    "콘텐츠투자": "콘텐츠·투자", "콘텐츠/투자": "콘텐츠·투자",
    "일상": "일상·감사", "감사": "일상·감사",
    "일상감사": "일상·감사", "일상/감사": "일상·감사",
}


def normalize_topic(topic: str) -> str:
    t = topic.strip().replace(" ", "")
    return TOPIC_ALIASES.get(t, topic.strip())


def load_golden():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─── Gemini FT ───

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류기입니다.

## 분류 우선순위 (위에서 아래로 판별, 먼저 해당되면 확정)

### 1순위: 운영 피드백
운영팀/사업팀이 사람 대 사람으로 처리할 요청·이슈.
예: 세미나 문의, 환불 요청, 멤버십 가입/해지 문의, 배송, 가격 정책 질문, 구독 관련 민원

### 2순위: 서비스 피드백
개발팀이 시스템을 수정해야 하는 기술적 이슈·요청.
예: 앱 크래시, 결제 오류(시스템), 로그인 실패, 기능 요청, 사칭 제보, 링크 오류

### 3순위: 콘텐츠·투자 ⭐ 가장 넓은 범위
투자/콘텐츠/마스터/시장/종목과 **조금이라도** 관련된 모든 글.
아래 신호가 **하나라도** 있으면 무조건 콘텐츠·투자:
- 종목명, 섹터, ETF, 지수, 코인 언급
- 수익/손실/매수/매도/포트폴리오/리밸런싱
- 마스터의 분석/강의/뷰/관점에 대한 반응 (칭찬이든 비판이든)
- 시장 상황, 거시경제, 금리, 환율 언급
- "강의 잘 들었습니다", "덕분에 공부했습니다" 등 학습 언급
- 멤버십 불만이지만 콘텐츠 품질이 이유인 경우

### 4순위: 일상·감사
위 1~3순위에 **전혀** 해당하지 않는 순수 인사·감사·안부·응원·격려·잡담.
투자/콘텐츠 신호가 **0개**일 때만 해당.
예: "감사합니다 명절 잘 보내세요", "힘내세요", 날씨 이야기, MBTI, 자기소개

### 5순위: 기타
무의미 노이즈, 분류 불가. 예: ".", "1", "?", 자음만, 테스트 글

## 핵심 규칙
- 감사/응원 + 투자 신호 → **콘텐츠·투자** (3순위 우선)
- 감사/응원만, 투자 신호 0개 → **일상·감사** (4순위)
- 애매하면 → **콘텐츠·투자** (투자 교육 커뮤니티이므로 콘텐츠·투자가 기본값)

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

다음 텍스트를 위 기준에 따라 분류하고, 카테고리명만 답하세요.
카테고리: 운영 피드백, 서비스 피드백, 콘텐츠·투자, 일상·감사, 기타"""


def classify_gemini(text: str, model_name: str, model_obj=None):
    """Gemini 파인튜닝 모델로 분류"""
    if model_obj is None:
        from vertexai.generative_models import GenerativeModel
        model_obj = GenerativeModel(
            model_name,
            system_instruction=SYSTEM_PROMPT,
        )
    response = model_obj.generate_content(text[:500])
    raw = response.text.strip()
    return normalize_topic(raw)


def benchmark_gemini(golden: list, model_name: str) -> dict:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai.init(project="us-service-data", location="us-central1")

    model_obj = GenerativeModel(
        model_name,
        system_instruction=SYSTEM_PROMPT,
    )

    correct = 0
    results = []
    errors = 0

    for i, item in enumerate(golden):
        text = item.get("text", "")
        true_label = item["v3c_topic"]
        try:
            pred = classify_gemini(text, model_name, model_obj)
            is_correct = pred == true_label
            if is_correct:
                correct += 1
            results.append({"true": true_label, "pred": pred, "correct": is_correct})
        except Exception as e:
            errors += 1
            results.append({"true": true_label, "pred": "기타", "correct": False, "error": str(e)[:100]})

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(golden)}  현재 정확도: {correct/(i+1)*100:.1f}%")

    return {"results": results, "correct": correct, "errors": errors}


# ─── EXAONE / Solar (GCS에서 LoRA 모델 로드) ───

def benchmark_lora(golden: list, model_name: str, gcs_output: str) -> dict:
    """GCS의 LoRA 모델을 다운로드하여 로컬에서 벤치마크"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # GCS에서 다운로드
    local_model_dir = f"/tmp/vertex_model_{model_name}"
    if not os.path.exists(local_model_dir):
        print(f"    GCS에서 모델 다운로드: {gcs_output}")
        os.makedirs(local_model_dir, exist_ok=True)
        os.system(f"gcloud storage cp -r {gcs_output}/* {local_model_dir}/")

    print(f"    모델 로드: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, local_model_dir)
    model.eval()

    INSTRUCTION = "다음 텍스트를 분류하세요. 카테고리: 운영 피드백, 서비스 피드백, 콘텐츠·투자, 일상·감사, 기타. 카테고리명만 답하세요."

    correct = 0
    results = []

    for i, item in enumerate(golden):
        text = item.get("text", "")[:500]
        true_label = item["v3c_topic"]

        prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{text}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # 첫 줄만 추출
        pred_raw = response.split("\n")[0].strip()
        pred = normalize_topic(pred_raw)
        if pred not in V3C_TOPICS:
            pred = "기타"

        is_correct = pred == true_label
        if is_correct:
            correct += 1
        results.append({"true": true_label, "pred": pred, "correct": is_correct})

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(golden)}  현재 정확도: {correct/(i+1)*100:.1f}%")

    return {"results": results, "correct": correct, "errors": 0}


# ─── 공통 분석 ───

def analyze_results(model_name: str, golden: list, bench: dict):
    total = len(golden)
    correct = bench["correct"]
    errors = bench.get("errors", 0)
    accuracy = correct / total * 100

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"  정확도: {correct}/{total} ({accuracy:.1f}%)")
    if errors:
        print(f"  에러: {errors}건")
    print(f"{'=' * 60}")

    # 카테고리별 P/R/F1
    from sklearn.metrics import classification_report
    y_true = [r["true"] for r in bench["results"]]
    y_pred = [r["pred"] for r in bench["results"]]
    print(classification_report(y_true, y_pred, labels=V3C_TOPICS, zero_division=0))

    # Confusion matrix 요약 (오분류 상위)
    misclassified = [(r["true"], r["pred"]) for r in bench["results"] if not r["correct"]]
    if misclassified:
        print("  주요 오분류:")
        for (t, p), cnt in Counter(misclassified).most_common(5):
            print(f"    {t} → {p}: {cnt}건")

    return {
        "model": model_name,
        "accuracy": round(accuracy, 1),
        "correct": correct,
        "total": total,
        "errors": errors,
    }


def save_version_log(entry: dict):
    VERSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if VERSION_LOG_PATH.exists():
        with open(VERSION_LOG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "entries" in data:
            entries = data["entries"]
        elif isinstance(data, list):
            entries = data
            data = {"schema_version": 1, "entries": entries}
        else:
            entries = []
            data = {"schema_version": 1, "entries": entries}
    else:
        entries = []
        data = {"schema_version": 1, "entries": entries}
    entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    entries.append(entry)
    with open(VERSION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  version_log 저장: {VERSION_LOG_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemini", "exaone", "solar", "all"], required=True)
    parser.add_argument("--gemini-endpoint", default=None,
                        help="Gemini FT 엔드포인트 (check_vertex_jobs.py에서 확인)")
    args = parser.parse_args()

    golden = load_golden()
    print(f"Golden set: {len(golden)}건\n")

    models_to_run = [args.model] if args.model != "all" else ["gemini", "exaone", "solar"]
    all_results = []

    for model_key in models_to_run:
        if model_key == "gemini":
            if not args.gemini_endpoint:
                # tuning job에서 엔드포인트 자동 가져오기
                import vertexai
                from vertexai.tuning import sft
                vertexai.init(project="us-service-data", location="us-central1")
                job = sft.SupervisedTuningJob(
                    "projects/306496779169/locations/us-central1/tuningJobs/3338612202419519488"
                )
                endpoint = getattr(job, "tuned_model_endpoint_name", None)
                if not endpoint:
                    print("[Gemini] 아직 학습 완료 안 됨. --gemini-endpoint로 직접 지정하세요.")
                    continue
                print(f"[Gemini] 엔드포인트: {endpoint}")
                bench = benchmark_gemini(golden, endpoint)
            else:
                bench = benchmark_gemini(golden, args.gemini_endpoint)
            result = analyze_results("Gemini 2.5 Flash FT", golden, bench)

        elif model_key == "exaone":
            bench = benchmark_lora(
                golden,
                "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                "gs://us-service-data-vertex-ft-us/models/exaone",
            )
            result = analyze_results("EXAONE 3.5 7.8B LoRA", golden, bench)

        elif model_key == "solar":
            bench = benchmark_lora(
                golden,
                "upstage/SOLAR-10.7B-Instruct-v1.0",
                "gs://us-service-data-vertex-ft-us/models/solar",
            )
            result = analyze_results("Solar 10.7B LoRA", golden, bench)

        save_version_log(result)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  최종 비교")
        print(f"{'=' * 60}")
        for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
            marker = " ✅" if r["accuracy"] >= 90 else ""
            print(f"  {r['model']}: {r['accuracy']}%{marker}")
        print()
        best = max(all_results, key=lambda x: x["accuracy"])
        if best["accuracy"] >= 90:
            print(f"  🎯 {best['model']}이(가) 90% 달성! 주간 파이프라인 전환 후보.")
        else:
            print(f"  ⚠️  전 모델 90% 미달. 2차 스케일업 검토 필요.")


if __name__ == "__main__":
    main()
