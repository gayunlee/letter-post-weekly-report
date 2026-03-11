"""
균형 학습 데이터 생성 — Vertex AI 파인튜닝용

labeled_all_v3c_full.json에서 golden set 제외 후 두 가지 균형 데이터셋 생성:
  1. balanced_strict: 각 카테고리 ~최소 클래스 건수 (1:1 균형)
  2. balanced_moderate: 소수 클래스 전량 + 콘텐츠·투자 다운샘플링

사용법:
  python3 scripts/prepare_balanced_vertex_data.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LABELED_ALL = BASE_DIR / "data/training_data/v3/labeled_all_v3c_full.json"
GOLDEN_DIR = BASE_DIR / "data/gold_dataset"
OUT_DIR = BASE_DIR / "data/training_data/v3/vertex_ai"

V3C_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "일상·감사", "기타"]

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

SEED = 42


def load_golden_texts() -> set:
    """Golden set 텍스트를 로드하여 제외용 set 생성"""
    texts = set()
    for p in GOLDEN_DIR.glob("*_golden_set.json"):
        with open(p, encoding="utf-8") as f:
            for item in json.load(f):
                texts.add(item.get("text", "").strip())
    print(f"Golden set 텍스트: {len(texts)}건 (제외 대상)")
    return texts


def load_and_clean() -> list[dict]:
    """labeled_all 로드 + golden 제외"""
    with open(LABELED_ALL, encoding="utf-8") as f:
        data = json.load(f)
    print(f"labeled_all: {len(data)}건")

    golden_texts = load_golden_texts()
    clean = [item for item in data if item.get("text", "").strip() not in golden_texts]
    excluded = len(data) - len(clean)
    print(f"Golden 제외: {excluded}건 → 클린: {len(clean)}건")
    return clean


def group_by_category(data: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for item in data:
        cat = item.get("v3c_topic", "")
        if cat in V3C_TOPICS:
            groups[cat].append(item)
    return groups


def make_balanced_strict(groups: dict[str, list[dict]]) -> list[dict]:
    """1:1 균형 — 가장 적은 카테고리에 맞춤"""
    min_count = min(len(v) for v in groups.values())
    result = []
    for cat in V3C_TOPICS:
        sampled = random.sample(groups[cat], min_count)
        result.extend(sampled)
    random.shuffle(result)
    return result


def make_balanced_moderate(groups: dict[str, list[dict]], majority_cap: int = 2000) -> list[dict]:
    """적당히 균형 — 소수 클래스 전량 + 다수 클래스 다운샘플링"""
    result = []
    for cat in V3C_TOPICS:
        items = groups[cat]
        if len(items) > majority_cap:
            sampled = random.sample(items, majority_cap)
            result.extend(sampled)
        else:
            result.extend(items)
    random.shuffle(result)
    return result


def split_train_val(data: list[dict], val_ratio: float = 0.1):
    """train/val split"""
    random.shuffle(data)
    val_size = max(1, int(len(data) * val_ratio))
    return data[val_size:], data[:val_size]


def to_sft(item: dict) -> dict:
    return {
        "instruction": SYSTEM_PROMPT,
        "input": item["text"][:500],
        "output": item["v3c_topic"]
    }


def write_jsonl(data: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(data)}건")


def print_dist(data: list[dict], label: str):
    dist = Counter(item.get("v3c_topic", item.get("output", "")) for item in data)
    print(f"\n  [{label}] 총 {len(data)}건")
    for cat in V3C_TOPICS:
        cnt = dist.get(cat, 0)
        pct = cnt / len(data) * 100 if data else 0
        print(f"    {cat}: {cnt}건 ({pct:.1f}%)")


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean = load_and_clean()
    groups = group_by_category(clean)

    print("\n클린 데이터 카테고리별:")
    for cat in V3C_TOPICS:
        print(f"  {cat}: {len(groups[cat])}건")

    # --- 1. balanced_strict (1:1) ---
    strict = make_balanced_strict(groups)
    strict_train, strict_val = split_train_val(strict)
    print_dist(strict_train, "strict train")
    print_dist(strict_val, "strict val")

    print("\n[balanced_strict SFT]")
    write_jsonl([to_sft(i) for i in strict_train], OUT_DIR / "sft_balanced_strict_train.jsonl")
    write_jsonl([to_sft(i) for i in strict_val], OUT_DIR / "sft_balanced_strict_val.jsonl")

    # --- 2. balanced_moderate ---
    moderate = make_balanced_moderate(groups)
    mod_train, mod_val = split_train_val(moderate)
    print_dist(mod_train, "moderate train")
    print_dist(mod_val, "moderate val")

    print("\n[balanced_moderate SFT]")
    write_jsonl([to_sft(i) for i in mod_train], OUT_DIR / "sft_balanced_moderate_train.jsonl")
    write_jsonl([to_sft(i) for i in mod_val], OUT_DIR / "sft_balanced_moderate_val.jsonl")

    # --- 비교 요약 ---
    print(f"\n{'='*60}")
    print("  데이터셋 비교")
    print(f"{'='*60}")
    print(f"  기존 불균형:      train ~10,641 / val ~1,416")
    print(f"  balanced_strict:  train {len(strict_train)} / val {len(strict_val)}")
    print(f"  balanced_moderate: train {len(mod_train)} / val {len(mod_val)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
