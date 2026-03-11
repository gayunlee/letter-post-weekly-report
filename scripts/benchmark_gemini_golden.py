"""Gemini Flash Golden Set 벤치마크 — 1건씩 개별 API 호출

사용법:
    python3 scripts/benchmark_gemini_golden.py
    python3 scripts/benchmark_gemini_golden.py --model gemini-2.0-flash
"""
import sys, os, json, argparse, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

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

## 응답: JSON만 출력
{"topic": "콘텐츠·투자", "confidence": 0.92}"""

V3C_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "일상·감사", "기타"]

# Gemini가 자기 맘대로 쓰는 카테고리명을 정규화
TOPIC_ALIASES = {
    "운영 피드백": "운영 피드백", "운영피드백": "운영 피드백", "운영": "운영 피드백",
    "서비스 피드백": "서비스 피드백", "서비스피드백": "서비스 피드백", "서비스": "서비스 피드백",
    "콘텐츠·투자": "콘텐츠·투자", "콘텐츠투자": "콘텐츠·투자", "콘텐츠": "콘텐츠·투자",
    "투자": "콘텐츠·투자", "콘텐츠/투자": "콘텐츠·투자",
    "일상·감사": "일상·감사", "일상감사": "일상·감사", "일상": "일상·감사", "감사": "일상·감사",
    "일상/감사": "일상·감사",
    "기타": "기타",
}

def normalize_topic(topic):
    if topic in V3C_TOPICS:
        return topic
    if topic in TOPIC_ALIASES:
        return TOPIC_ALIASES[topic]
    topic_lower = topic.strip()
    for alias, canonical in TOPIC_ALIASES.items():
        if alias in topic_lower:
            return canonical
    return "기타"


def classify_single(model_obj, text):
    if not text or len(text.strip()) < 3:
        return {"topic": "기타", "confidence": 1.0}
    try:
        response = model_obj.generate_content(
            text[:500],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1024,
                temperature=0.0,
            ),
        )
        raw = response.text.strip()
        # Handle ```json ... ``` wrapping
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        raw = raw.strip()
        # Handle single quotes → double quotes
        if raw.startswith("{") and "'" in raw:
            raw = raw.replace("'", '"')
        result = json.loads(raw)
        topic = normalize_topic(result.get("topic", "기타"))
        conf = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        return {"topic": topic, "confidence": conf}
    except Exception as e:
        return {"topic": "기타", "confidence": 0.0, "error": str(e)[:100], "raw": raw[:200] if 'raw' in dir() else ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--golden-path", default="data/gold_dataset/v6_golden_set.json")
    args = parser.parse_args()

    model_obj = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=CLASSIFY_PROMPT,
    )

    with open(args.golden_path, encoding='utf-8') as f:
        golden = json.load(f)

    print("=" * 60)
    print(f"  Golden Set 벤치마크: {args.model}")
    print("=" * 60)
    print(f"  Golden set: {len(golden)}건")

    true_dist = Counter(item['v3c_topic'] for item in golden)
    print(f"\n  Golden v3c 정답 분포:")
    for t in V3C_TOPICS:
        print(f"    {t}: {true_dist.get(t, 0)}건")

    results = [None] * len(golden)
    errors = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {}
        for i, item in enumerate(golden):
            future = executor.submit(classify_single, model_obj, item['text'])
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
                rate = done_count / elapsed
                print(f"    {done_count}/{len(golden)} 완료 ({rate:.1f}건/초)")

    elapsed = time.time() - start_time
    print(f"\n  총 소요: {elapsed:.1f}초 ({len(golden)/elapsed:.1f}건/초)")

    if errors:
        print(f"\n  에러: {len(errors)}건")
        for idx, err in errors[:5]:
            print(f"    [{idx}] {err}")

    y_true = [item['v3c_topic'] for item in golden]
    y_pred = [r['topic'] for r in results]

    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(golden)

    print(f"\n  정확도: {acc*100:.1f}% ({correct}/{len(golden)})")

    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    report = classification_report(y_true, y_pred, labels=V3C_TOPICS, digits=4, zero_division=0)
    print(f"\n{report}")

    cm = confusion_matrix(y_true, y_pred, labels=V3C_TOPICS)
    print("  Confusion Matrix:")
    header = f"  {'':>14}"
    for t in V3C_TOPICS:
        header += f"  {t[:6]:>6}"
    print(header)
    for i, row in enumerate(cm):
        line = f"  {V3C_TOPICS[i]:>14}"
        for val in row:
            line += f"  {val:>6}"
        print(line)

    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                'text': golden[i]['text'][:80],
                'true': true,
                'pred': pred,
                'conf': results[i].get('confidence', 0),
            })

    print(f"\n  오분류: {len(misclassified)}건")
    for m in misclassified:
        print(f"    [{m['true']} -> {m['pred']}] conf={m['conf']:.2f} | {m['text'][:60]}")

    print(f"\n{'='*60}")
    print(f"  KcBERT (v3c_expanded):      60.4%")
    print(f"  Haiku 4.5 (에이전트):       86.3%")
    print(f"  **{args.model}:  {acc*100:.1f}%**")
    print(f"{'='*60}")

    # Save
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=V3C_TOPICS, zero_division=0
    )
    per_cat = {}
    for i, t in enumerate(V3C_TOPICS):
        per_cat[t] = {
            'precision': round(float(p[i]), 4),
            'recall': round(float(r[i]), 4),
            'f1': round(float(f1[i]), 4),
            'support': int(support[i]),
        }

    output = {
        'model': args.model,
        'golden_set_size': len(golden),
        'accuracy': round(acc, 4),
        'correct': correct,
        'per_category': per_cat,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'misclassified': misclassified,
        'elapsed_seconds': round(elapsed, 1),
        'errors': len(errors),
    }

    safe_name = args.model.replace('/', '_').replace('.', '_')
    output_path = f"benchmarks/golden_benchmark_{safe_name}.json"
    os.makedirs('benchmarks', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output_path}")


if __name__ == "__main__":
    main()
