"""
MLX EXAONE LoRA 모델 — Golden Set 벤치마크

사용법:
  python3 scripts/benchmark_mlx_golden.py --adapter-path models/mlx/exaone_balanced_strict
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_PATH = BASE_DIR / "data/gold_dataset/v6_golden_set.json"
HF_MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

CATEGORIES = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "일상·감사", "기타"]

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류기입니다.

## 분류 우선순위 (위에서 아래로 판별, 먼저 해당되면 확정)

### 1순위: 운영 피드백
운영팀/사업팀이 사람 대 사람으로 처리할 요청·이슈.
예: 세미나 문의, 환불 요청, 멤버십 가입/해지 문의, 배송, 가격 정책 질문, 구독 관련 민원

### 2순위: 서비스 피드백
개발팀이 시스템을 수정해야 하는 기술적 이슈·요청.
예: 앱 크래시, 결제 오류(시스템), 로그인 실패, 기능 요청, 사칭 제보, 링크 오류

### 3순위: 콘텐츠·투자 ⭐ 가장 넓은 범위
투자/콘텐츠/마스터/시장/종목과 조금이라도 관련된 모든 글.
아래 신호가 하나라도 있으면 무조건 콘텐츠·투자:
- 종목명, 섹터, ETF, 지수, 코인 언급
- 수익/손실/매수/매도/포트폴리오/리밸런싱
- 마스터의 분석/강의/뷰/관점에 대한 반응
- 시장 상황, 거시경제, 금리, 환율 언급
- "강의 잘 들었습니다", "덕분에 공부했습니다" 등 학습 언급
- 멤버십 불만이지만 콘텐츠 품질이 이유인 경우

### 4순위: 일상·감사
위 1~3순위에 전혀 해당하지 않는 순수 인사·감사·안부·응원·격려·잡담.
투자/콘텐츠 신호가 0개일 때만 해당.

### 5순위: 기타
무의미 노이즈, 분류 불가. 예: ".", "1", "?", 자음만, 테스트 글

## 핵심 규칙
- 감사/응원 + 투자 신호 → 콘텐츠·투자 (3순위 우선)
- 감사/응원만, 투자 신호 0개 → 일상·감사 (4순위)
- 애매하면 → 콘텐츠·투자

다음 텍스트를 위 기준에 따라 분류하고, 카테고리명만 답하세요.
카테고리: 운영 피드백, 서비스 피드백, 콘텐츠·투자, 일상·감사, 기타"""


def parse_category(response: str) -> str:
    """모델 응답에서 카테고리 추출"""
    response = response.strip()
    for cat in CATEGORIES:
        if cat in response:
            return cat
    # 부분 매칭
    if "운영" in response:
        return "운영 피드백"
    if "서비스" in response:
        return "서비스 피드백"
    if "콘텐츠" in response or "투자" in response:
        return "콘텐츠·투자"
    if "일상" in response or "감사" in response:
        return "일상·감사"
    if "기타" in response:
        return "기타"
    return response[:20]  # unknown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--max-items", type=int, default=0, help="0=all")
    args = parser.parse_args()

    # Load golden set
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)
    print(f"Golden set: {len(golden)}건")

    if args.max_items > 0:
        golden = golden[:args.max_items]
        print(f"  (제한: {args.max_items}건)")

    # Load model
    print(f"모델 로딩: {HF_MODEL}")
    print(f"어댑터: {args.adapter_path}")
    from mlx_lm import load, generate

    model, tokenizer = load(HF_MODEL, adapter_path=args.adapter_path)
    print("모델 로딩 완료\n")

    # Run inference
    correct = 0
    total = 0
    results = []
    confusion = {}  # (true, pred) -> count

    for i, item in enumerate(golden):
        true_label = item.get("v3c_topic")
        if not true_label:
            continue

        text = item["text"][:500]

        # Build chat prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"다음 텍스트를 분류하세요:\n\n{text}"},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"[|system|]{SYSTEM_PROMPT}[|endofturn|]\n[|user|]다음 텍스트를 분류하세요:\n\n{text}[|endofturn|]\n[|assistant|]"

        response = generate(model, tokenizer, prompt=prompt, max_tokens=20)
        pred_label = parse_category(response)

        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        total += 1

        # Confusion tracking
        key = (true_label, pred_label)
        confusion[key] = confusion.get(key, 0) + 1

        results.append({
            "_id": item["_id"],
            "true": true_label,
            "pred": pred_label,
            "correct": is_correct,
            "response": response.strip(),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(golden)}] 정확도: {correct}/{total} ({correct/total*100:.1f}%)")

    # Final results
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"  EXAONE LoRA Golden Benchmark")
    print(f"{'='*60}")
    print(f"  전체 정확도: {correct}/{total} ({accuracy*100:.1f}%)")

    # Per-category metrics
    print(f"\n  카테고리별 성능:")
    for cat in CATEGORIES:
        tp = sum(1 for r in results if r["true"] == cat and r["pred"] == cat)
        fp = sum(1 for r in results if r["true"] != cat and r["pred"] == cat)
        fn = sum(1 for r in results if r["true"] == cat and r["pred"] != cat)
        total_cat = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"    {cat:12s}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={total_cat})")

    # Confusion matrix summary (top misclassifications)
    print(f"\n  주요 오분류:")
    errors = {k: v for k, v in confusion.items() if k[0] != k[1]}
    for (true, pred), count in sorted(errors.items(), key=lambda x: -x[1])[:10]:
        print(f"    {true} → {pred}: {count}건")

    # Save results
    out_path = BASE_DIR / "benchmarks" / f"golden_benchmark_exaone_balanced_strict.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": HF_MODEL,
            "adapter": args.adapter_path,
            "golden_count": total,
            "accuracy": accuracy,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    main()
