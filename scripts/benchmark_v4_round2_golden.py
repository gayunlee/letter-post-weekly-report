"""v4 Round 2 벤치마크 — 기존 프롬프트 + 요약 전처리 + 후처리 4분류

목적: 기존 v3c 5분류 프롬프트를 유지하되, "핵심 요약 후 분류" chain-of-thought를
      추가하여 노이즈/경계 케이스 정확도가 올라가는지 확인.
      후처리로 일상·감사+기타→노이즈 매핑.

변경점 (vs 기존 Haiku 벤치마크):
  - 프롬프트에 "핵심을 1문장 요약 → 요약 기반 분류" 단계 추가
  - 후처리: 5분류 → 4분류 매핑

비교 대상:
  - 84.6% (기존 결과 재계산 baseline)
  - 82.4% (Round 1 cascade+키워드)

사용법:
    python3 scripts/benchmark_v4_round2_golden.py
"""
import sys, os, json, argparse, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

# 기존 v3c 프롬프트 + 요약 chain-of-thought
CLASSIFY_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류기입니다.

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

## 분류 절차 (반드시 따를 것)
1. 먼저 이 텍스트의 **핵심 의도를 1문장으로 요약**하세요.
2. 요약문을 기반으로 위 우선순위에 따라 분류하세요.

## 응답: JSON만 출력
{"summary": "핵심 요약 1문장", "topic": "콘텐츠·투자", "confidence": 0.92}"""

V3C_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "일상·감사", "기타"]
V4_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "노이즈"]


def map_to_v4(topic):
    if topic in ("일상·감사", "기타"):
        return "노이즈"
    return topic


def classify_single(client, model, text):
    if not text or len(text.strip()) < 3:
        return {"topic": "기타", "confidence": 1.0, "summary": ""}
    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,  # 요약 포함이라 더 넉넉하게
            system=CLASSIFY_PROMPT,
            messages=[{"role": "user", "content": text[:500]}],
            timeout=30.0,
        )
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        if raw.startswith("{{") and raw.endswith("}}"):
            raw = raw[1:-1]
        result = json.loads(raw)
        topic = result.get("topic", "기타")
        if topic not in V3C_TOPICS:
            topic = "기타"
        conf = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        summary = result.get("summary", "")
        return {"topic": topic, "confidence": conf, "summary": summary}
    except Exception as e:
        return {"topic": "기타", "confidence": 0.0, "error": str(e)[:80], "summary": ""}


def main():
    parser = argparse.ArgumentParser(description="v4 Round 2 — 요약 전처리 + 후처리 4분류")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--golden-path", default="data/gold_dataset/v6_golden_set.json")
    args = parser.parse_args()

    client = Anthropic()

    with open(args.golden_path, encoding='utf-8') as f:
        golden = json.load(f)

    print("=" * 60)
    print(f"  v4 Round 2: 요약 전처리 + 후처리 4분류")
    print(f"  모델: {args.model}")
    print("=" * 60)
    print(f"  Golden set: {len(golden)}건")

    # Golden 정답 v4 매핑
    y_true_v4 = [map_to_v4(item['v3c_topic']) for item in golden]
    y_true_v3c = [item['v3c_topic'] for item in golden]
    true_dist = Counter(y_true_v4)
    print(f"\n  v4 정답 분포:")
    for t in V4_TOPICS:
        print(f"    {t}: {true_dist.get(t, 0)}건")

    # 병렬 분류
    results = [None] * len(golden)
    errors = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {}
        for i, item in enumerate(golden):
            future = executor.submit(classify_single, client, args.model, item['text'])
            future_map[future] = i

        done_count = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            result = future.result()
            results[idx] = result
            if 'error' in result:
                errors.append((idx, result['error']))
            done_count += 1
            if done_count % 50 == 0 or done_count == len(golden):
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"    {done_count}/{len(golden)} 완료 ({rate:.1f}건/초)")

    elapsed = time.time() - start_time
    print(f"\n  총 소요: {elapsed:.1f}초 ({len(golden)/elapsed:.1f}건/초)")

    if errors:
        print(f"\n  에러: {len(errors)}건")
        for idx, err in errors[:5]:
            print(f"    [{idx}] {err}")

    # v3c 5분류 정확도 (요약 추가 효과)
    y_pred_v3c = [r['topic'] for r in results]
    correct_v3c = sum(1 for a, b in zip(y_true_v3c, y_pred_v3c) if a == b)
    acc_v3c = correct_v3c / len(golden)
    print(f"\n  v3c 5분류 정확도: {acc_v3c*100:.1f}% ({correct_v3c}/{len(golden)})")

    # v4 4분류 정확도 (후처리 매핑)
    y_pred_v4 = [map_to_v4(r['topic']) for r in results]
    correct_v4 = sum(1 for a, b in zip(y_true_v4, y_pred_v4) if a == b)
    acc_v4 = correct_v4 / len(golden)
    print(f"  v4 4분류 정확도: {acc_v4*100:.1f}% ({correct_v4}/{len(golden)})")

    # Per-category metrics (v4)
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    report = classification_report(y_true_v4, y_pred_v4, labels=V4_TOPICS, digits=4, zero_division=0)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true_v4, y_pred_v4, labels=V4_TOPICS)
    print("  Confusion Matrix:")
    header = f"  {'':>14}"
    for t in V4_TOPICS:
        header += f"  {t[:6]:>6}"
    print(header)
    for i, row in enumerate(cm):
        line = f"  {V4_TOPICS[i]:>14}"
        for val in row:
            line += f"  {val:>6}"
        print(line)

    # 오분류 상세
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true_v4, y_pred_v4)):
        if true != pred:
            misclassified.append({
                'text': golden[i]['text'][:80],
                'true': true,
                'pred': pred,
                'conf': results[i].get('confidence', 0),
                'summary': results[i].get('summary', ''),
                'v3c_original': golden[i]['v3c_topic'],
            })

    print(f"\n  오분류: {len(misclassified)}건")
    for m in misclassified:
        print(f"    [{m['true']} -> {m['pred']}] conf={m['conf']:.2f} | {m['summary'][:60]}")

    # 비교
    print(f"\n{'='*60}")
    print(f"  비교:")
    print(f"    v3c 5분류 Haiku (기존):       78.9% (227건)")
    print(f"    v4 4분류 재계산 (baseline):    84.6% (227건)")
    print(f"    v4 R1 cascade+키워드:          82.4% (227건)")
    print(f"    v4 R2 요약+후처리:             {acc_v4*100:.1f}% (227건)")
    delta = acc_v4 * 100 - 84.6
    sign = "+" if delta >= 0 else ""
    print(f"    baseline 대비:                 {sign}{delta:.1f}%p")
    print(f"")
    print(f"    (참고) v3c 5분류 요약 추가:    {acc_v3c*100:.1f}% vs 기존 78.9%")
    print(f"{'='*60}")

    # 결과 저장
    p, r, f1, support = precision_recall_fscore_support(
        y_true_v4, y_pred_v4, labels=V4_TOPICS, zero_division=0
    )
    per_cat = {}
    for i, t in enumerate(V4_TOPICS):
        per_cat[t] = {
            'precision': round(float(p[i]), 4),
            'recall': round(float(r[i]), 4),
            'f1': round(float(f1[i]), 4),
            'support': int(support[i]),
        }

    output = {
        'experiment': 'v4_round2_summary_postprocess',
        'model': args.model,
        'taxonomy': '4분류 후처리 (5분류 분류 → 일상·감사+기타→노이즈)',
        'changes': ['요약 chain-of-thought 추가', '후처리 4분류 매핑'],
        'golden_set_size': len(golden),
        'accuracy_v3c_5way': round(acc_v3c, 4),
        'accuracy_v4_4way': round(acc_v4, 4),
        'correct_v4': correct_v4,
        'baseline_comparison': {
            'v3c_5way_haiku': 0.789,
            'v4_4way_recalc': 0.846,
            'v4_round1': 0.824,
            'v4_round2': round(acc_v4, 4),
        },
        'per_category': per_cat,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'misclassified': misclassified,
        'elapsed_seconds': round(elapsed, 1),
        'errors': len(errors),
    }

    output_path = "benchmarks/golden_benchmark_v4_round2.json"
    os.makedirs('benchmarks', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output_path}")


if __name__ == "__main__":
    main()
