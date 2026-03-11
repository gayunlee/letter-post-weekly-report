"""
Vertex AI 파인튜닝용 학습 데이터 포맷 변환

입력: data/training_data/v3/topic_v3c_full_clean/{train,val,test}.json
출력: data/training_data/v3/vertex_ai/
  - gemini_ft_{train,val}.jsonl        (Gemini supervised tuning)
  - sft_{train,val}.jsonl              (EXAONE/Solar instruction tuning)
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data/training_data/v3/topic_v3c_full_clean"
OUT_DIR = BASE_DIR / "data/training_data/v3/vertex_ai"

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

USER_TEMPLATE = "다음 텍스트를 분류하세요:\n\n{text}"


def load_split(name: str) -> list[dict]:
    path = CLEAN_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_gemini_ft(item: dict) -> dict:
    """Gemini supervised tuning 포맷 (Vertex AI)
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning
    """
    return {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": item["text"][:500]}]
            },
            {
                "role": "model",
                "parts": [{"text": item["category"]}]
            }
        ]
    }


def to_sft(item: dict) -> dict:
    """EXAONE / Solar instruction tuning 포맷
    Standard SFT format for HuggingFace-based models
    """
    return {
        "instruction": SYSTEM_PROMPT,
        "input": item["text"][:500],
        "output": item["category"]
    }


def write_jsonl(data: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(data)}건")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # train + val for training, test는 벤치마크용 (golden set 사용하므로 별도 변환 불필요)
    train = load_split("train")
    val = load_split("val")

    print(f"입력: train {len(train)}건, val {len(val)}건")
    print(f"출력: {OUT_DIR}\n")

    # 카테고리 분포
    from collections import Counter
    dist = Counter(item["category"] for item in train)
    print("train 카테고리 분포:")
    for cat, cnt in dist.most_common():
        print(f"  {cat}: {cnt}건 ({cnt/len(train)*100:.1f}%)")
    print()

    # Gemini FT
    print("[Gemini supervised tuning]")
    write_jsonl([to_gemini_ft(item) for item in train], OUT_DIR / "gemini_ft_train.jsonl")
    write_jsonl([to_gemini_ft(item) for item in val], OUT_DIR / "gemini_ft_val.jsonl")

    # SFT (EXAONE / Solar)
    print("\n[EXAONE/Solar SFT]")
    write_jsonl([to_sft(item) for item in train], OUT_DIR / "sft_train.jsonl")
    write_jsonl([to_sft(item) for item in val], OUT_DIR / "sft_val.jsonl")

    # 샘플 출력
    print("\n--- Gemini FT 샘플 ---")
    print(json.dumps(to_gemini_ft(train[0]), ensure_ascii=False, indent=2)[:400])
    print("\n--- SFT 샘플 ---")
    print(json.dumps(to_sft(train[0]), ensure_ascii=False, indent=2)[:400])


if __name__ == "__main__":
    main()
