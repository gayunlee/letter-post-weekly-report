"""v4 Round 1 벤치마크 — 4분류 cascade + 한국어 키워드

목적: 분류 체계 단순화(노이즈 통합) + cascade 프롬프트 + 한국어 키워드가
      84.6% baseline(기존 결과 재계산)을 넘는지 확인.

변경점 (vs 기존 Haiku 벤치마크):
  1. 5분류 → 4분류 (일상·감사 + 기타 → 노이즈)
  2. Cascade 우선순위 프롬프트 (단계별 필터)
  3. 각 카테고리에 한국어 키워드 명시 (당근마켓 참고)

Golden set 비교: v3c_topic의 일상·감사/기타를 노이즈로 매핑하여 비교.

사용법:
    python3 scripts/benchmark_v4_round1_golden.py
"""
import sys, os, json, argparse, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

# v4 Round 1: 4분류 cascade + 한국어 키워드
CLASSIFY_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류기입니다.

다음 텍스트를 아래 **단계별 필터**를 순서대로 적용하여 분류하세요.
먼저 해당되는 단계에서 확정하고, 이후 단계는 무시합니다.

## Step 1: 노이즈 필터
아래에 해당하면 → **노이즈**
- 무의미한 글: ".", "1", "?", 자음/모음만, 테스트 글
- 순수 인사·안부·잡담: 투자/콘텐츠/서비스 언급이 **전혀** 없는 인사말
- 키워드: 감사합니다, 명절, 새해, 건강, 응원, 힘내세요, 날씨, MBTI, 자기소개
- ⚠️ "감사합니다" + 투자/콘텐츠 언급 → 노이즈가 **아님** (Step 3으로)

## Step 2: 대응 필요 여부 (액션 필요한 건)

### 2-A: 운영 피드백
운영팀/사업팀이 **사람 대 사람으로** 처리할 요청·이슈.
- 키워드: 환불, 취소, 해지, 구독, 멤버십, 가입, 탈퇴, 배송, 세미나, 수강, 가격, 요금, 결제 문의, 계좌이체, 패키지, 겨울학기, 여름학기, 상담, 연락처, 전화
- 예: "환불 가능한가요?", "멤버십 해지하고 싶습니다", "세미나 일정 문의"

### 2-B: 서비스 피드백
개발팀이 **시스템을 수정**해야 하는 기술적 이슈·기능 요청.
- 키워드: 오류, 에러, 버그, 크래시, 안됨, 로그인, 접속, 앱, 알림, 푸시, 링크, 기능 요청, 사칭, 해킹, 다운로드, 업데이트
- 예: "앱이 자꾸 튕겨요", "결제 화면에서 오류", "알림이 안 와요"
- ⚠️ "결제 오류(시스템)" → 서비스 / "결제 문의(사람)" → 운영

## Step 3: 콘텐츠·투자
Step 1~2에 해당하지 않는 나머지 **모든 글**.
투자/콘텐츠/마스터/시장과 조금이라도 관련되면 여기.
- 키워드: 종목, 주식, ETF, 코인, 매수, 매도, 수익, 손실, 포트폴리오, 리밸런싱, 강의, 분석, 마스터, 시장, 금리, 환율, 섹터, 콘텐츠, 영상, 자료
- 감사/응원 + 투자 신호 → **콘텐츠·투자** (노이즈 아님)
- 멤버십 불만이지만 콘텐츠 품질이 이유 → **콘텐츠·투자**
- 애매하면 → **콘텐츠·투자** (금융 교육 커뮤니티이므로 기본값)

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

## 응답: JSON만 출력
{"topic": "콘텐츠·투자", "confidence": 0.92}"""

V4_TOPICS = ["운영 피드백", "서비스 피드백", "콘텐츠·투자", "노이즈"]

# Golden set 매핑: v3c 5분류 → v4 4분류
def map_to_v4(v3c_topic):
    if v3c_topic in ("일상·감사", "기타"):
        return "노이즈"
    return v3c_topic


def classify_single(client, model, text):
    if not text or len(text.strip()) < 3:
        return {"topic": "노이즈", "confidence": 1.0}
    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
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
        topic = result.get("topic", "콘텐츠·투자")
        # 매핑: 모델이 혹시 일상·감사/기타로 답하면 노이즈로
        if topic in ("일상·감사", "기타"):
            topic = "노이즈"
        if topic not in V4_TOPICS:
            topic = "콘텐츠·투자"  # fallback
        conf = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        return {"topic": topic, "confidence": conf}
    except Exception as e:
        return {"topic": "콘텐츠·투자", "confidence": 0.0, "error": str(e)[:80]}


def main():
    parser = argparse.ArgumentParser(description="v4 Round 1 — 4분류 cascade + 키워드")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--golden-path", default="data/gold_dataset/v6_golden_set.json")
    args = parser.parse_args()

    client = Anthropic()

    with open(args.golden_path, encoding='utf-8') as f:
        golden = json.load(f)

    print("=" * 60)
    print(f"  v4 Round 1: 4분류 cascade + 키워드")
    print(f"  모델: {args.model}")
    print("=" * 60)
    print(f"  Golden set: {len(golden)}건")

    # Golden 정답을 v4로 매핑
    y_true = [map_to_v4(item['v3c_topic']) for item in golden]
    true_dist = Counter(y_true)
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

    # 정확도
    y_pred = [r['topic'] for r in results]
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(golden)

    print(f"\n  정확도: {acc*100:.1f}% ({correct}/{len(golden)})")

    # Per-category metrics
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    report = classification_report(y_true, y_pred, labels=V4_TOPICS, digits=4, zero_division=0)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=V4_TOPICS)
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
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            misclassified.append({
                'text': golden[i]['text'][:80],
                'true': true,
                'pred': pred,
                'conf': results[i].get('confidence', 0),
                'v3c_original': golden[i]['v3c_topic'],
            })

    print(f"\n  오분류: {len(misclassified)}건")
    for m in misclassified:
        print(f"    [{m['true']} -> {m['pred']}] conf={m['conf']:.2f} | {m['text'][:60]}")

    # 비교
    print(f"\n{'='*60}")
    print(f"  비교:")
    print(f"    v3c 5분류 Haiku (기존):       78.9% (227건)")
    print(f"    v4 4분류 재계산 (baseline):    84.6% (227건)")
    print(f"    v4 Round 1 cascade+키워드:     {acc*100:.1f}% (227건)")
    delta = acc * 100 - 84.6
    sign = "+" if delta >= 0 else ""
    print(f"    baseline 대비:                 {sign}{delta:.1f}%p")
    print(f"{'='*60}")

    # 결과 저장
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=V4_TOPICS, zero_division=0
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
        'experiment': 'v4_round1_cascade_keywords',
        'model': args.model,
        'taxonomy': '4분류 (운영/서비스/콘텐츠·투자/노이즈)',
        'changes': ['5→4분류 노이즈 통합', 'cascade 우선순위 프롬프트', '한국어 키워드 명시'],
        'golden_set_size': len(golden),
        'accuracy': round(acc, 4),
        'correct': correct,
        'baseline_comparison': {
            'v3c_5way_haiku': 0.789,
            'v4_4way_recalc': 0.846,
            'v4_round1': round(acc, 4),
        },
        'per_category': per_cat,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'misclassified': misclassified,
        'elapsed_seconds': round(elapsed, 1),
        'errors': len(errors),
    }

    output_path = "benchmarks/golden_benchmark_v4_round1.json"
    os.makedirs('benchmarks', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output_path}")


if __name__ == "__main__":
    main()
