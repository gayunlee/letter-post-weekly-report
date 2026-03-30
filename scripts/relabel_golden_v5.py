"""Golden set v7 → v5 5분류 후보 라벨링

v3c_topic + v4_3class 매핑으로 145건은 규칙 기반, 나머지 ~82건은 Opus로 판정.
Opus 호출 대상: v3c=콘텐츠·투자 → 마스터 반응 vs 시장·투자 분할 + edge cases.

사용법:
    python3 scripts/relabel_golden_v5.py
"""
import sys, os, json, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

V5_TOPICS = ["운영 대응", "서비스 피드백", "마스터 반응", "시장·투자", "일상"]

# v3c → v5 규칙 매핑 (v4_3class와 일치하는 깨끗한 경우만)
RULE_MAP = {
    # (v3c_topic, v4_3class) → v5_5class
    ("운영 피드백", "대응 필요"): "운영 대응",
    ("서비스 피드백", "대응 필요"): "서비스 피드백",
    ("기타", "노이즈"): "일상",
    ("일상·감사", "노이즈"): "일상",
}

OPUS_SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류 전문가입니다.

## 분류 과제
이 텍스트를 아래 5개 카테고리 중 하나로 분류하세요.

## 5분류 체계 (v5)

### 1순위: 운영 대응
CS/운영팀이 1:1로 처리해야 하는 건.
- 환불, 해지, 구독 변경, 멤버십 등급 변경
- 앱 오류, 로그인 실패, 링크 오류 (우리 서비스 문제)
- 세미나 문의, 배송 문의, 가격 정책 질문
- 커뮤니티 운영 정책 비판 (운영진/알바/관리자 비판)

### 2순위: 서비스 피드백
제품팀이 수집하는 개선 사안. 1:1 처리 불필요.
- 앱 UX 문제, 기능 요청 (투표기능, 게시판 기능 등)
- 콘텐츠 접근 문제 (VOD 업로드 요청 등)
- 플랫폼 개선 의견

### 3순위: 마스터 반응
마스터(콘텐츠 크리에이터)에 대한 평가/피드백. **마스터가 주어**.
- 마스터 칭찬/감사 + 콘텐츠 언급 ("강의 덕분에 성장했습니다")
- 마스터 비판/불신 ("계좌 공개해라", "매달 같은 브리핑")
- 마스터 건강/안부 걱정 + 콘텐츠 맥락
- 마스터의 분석/판단/실력에 대한 평가
- 학습/성장 경험 (마스터의 가르침 맥락)

### 4순위: 시장·투자
시장/종목/포트폴리오가 주어. 마스터가 아닌 **투자 자체**가 핵심.
- 종목명, 섹터, ETF, 지수, 코인 분석
- 수익/손실/매매/포트폴리오 보고
- 시장 전망, 거시경제, 금리, 환율
- 투자 전략 공유/질문
- **애매하면 → 시장·투자** (기본값)

### 5순위: 일상
투자/콘텐츠/서비스 신호 0개. 액션 불필요.
- 순수 인사, 안부, 일상 잡담
- 무의미 텍스트 (".", "1", 자음만)

## 핵심 경계 규칙

### 마스터 반응 vs 시장·투자
- 마스터의 **판단/분석/강의**에 대한 평가 → 마스터 반응
- **종목/시장** 데이터 자체에 대한 논의 → 시장·투자
- "삼전 분석 잘 해주셔서 수익" → 마스터 반응 (마스터 분석 = 주어)
- "삼전이 실적 잘 나왔네요" → 시장·투자 (종목 = 주어)
- "포트폴리오 복기해봤습니다" → 시장·투자 (투자 행위 = 주어)
- "선생님 덕분에 투자관 정립" → 마스터 반응 (마스터 영향 = 주어)

## 응답 형식: JSON만 출력
{"topic": "마스터 반응", "reason": "마스터 강의에 대한 감사와 학습 성장 언급", "confidence": 0.9}"""


def rule_map_label(item: dict) -> str | None:
    """규칙 매핑이 가능하면 v5 라벨 반환, 아니면 None."""
    key = (item.get("v3c_topic", ""), item.get("v4_3class", ""))
    return RULE_MAP.get(key)


def classify_with_opus(client: Anthropic, text: str) -> dict:
    """Opus로 v5 5분류 판정."""
    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=200,
            system=OPUS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text[:500]}],
            timeout=60.0,
        )
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        result = json.loads(raw)
        topic = result.get("topic", "시장·투자")
        if topic not in V5_TOPICS:
            topic = "시장·투자"
        return {
            "topic": topic,
            "reason": result.get("reason", ""),
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except Exception as e:
        return {
            "topic": "시장·투자",
            "reason": f"ERROR: {e}",
            "confidence": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


def main():
    golden_path = "data/gold_dataset/v7_golden_set.json"
    with open(golden_path, encoding='utf-8') as f:
        golden = json.load(f)

    print("=" * 60)
    print(f"  v5 5분류 후보 라벨링 — 규칙 매핑 + Opus")
    print(f"  Golden set: {len(golden)}건")
    print("=" * 60)

    # Phase 1: 규칙 매핑
    candidates = []
    need_opus = []

    for i, item in enumerate(golden):
        v5_label = rule_map_label(item)
        if v5_label:
            candidates.append({
                "index": i,
                "_id": item["_id"],
                "text": item["text"][:200],
                "v3c_topic": item.get("v3c_topic", ""),
                "v4_3class": item.get("v4_3class", ""),
                "v5_candidate": v5_label,
                "source": "rule",
                "reason": f"v3c={item.get('v3c_topic','')} + v4={item.get('v4_3class','')} → {v5_label}",
                "confidence": 1.0,
            })
        else:
            need_opus.append(i)
            candidates.append(None)  # placeholder

    rule_count = sum(1 for c in candidates if c is not None)
    print(f"\n  규칙 매핑: {rule_count}건")
    print(f"  Opus 필요: {len(need_opus)}건")

    # 규칙 매핑 분포
    rule_dist = Counter(c["v5_candidate"] for c in candidates if c is not None)
    print(f"\n  규칙 매핑 분포:")
    for t in V5_TOPICS:
        print(f"    {t}: {rule_dist.get(t, 0)}건")

    # Opus 대상 분석
    opus_src = Counter(
        f"{golden[i].get('v3c_topic','')} + {golden[i].get('v4_3class','')}"
        for i in need_opus
    )
    print(f"\n  Opus 대상 원본 분포:")
    for src, cnt in opus_src.most_common():
        print(f"    {src}: {cnt}건")

    # Phase 2: Opus 분류
    client = Anthropic()
    total_input = 0
    total_output = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {}
        for idx in need_opus:
            future = executor.submit(classify_with_opus, client, golden[idx]["text"])
            future_map[future] = idx

        done_count = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            result = future.result()
            total_input += result.pop("input_tokens", 0)
            total_output += result.pop("output_tokens", 0)

            candidates[idx] = {
                "index": idx,
                "_id": golden[idx]["_id"],
                "text": golden[idx]["text"][:200],
                "v3c_topic": golden[idx].get("v3c_topic", ""),
                "v4_3class": golden[idx].get("v4_3class", ""),
                "v5_candidate": result["topic"],
                "source": "opus",
                "reason": result["reason"],
                "confidence": result["confidence"],
            }

            done_count += 1
            if done_count % 20 == 0 or done_count == len(need_opus):
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"    Opus: {done_count}/{len(need_opus)} 완료 ({rate:.1f}건/초)")

    elapsed = time.time() - start_time
    print(f"\n  Opus 소요: {elapsed:.1f}초")

    # 전체 분포
    all_dist = Counter(c["v5_candidate"] for c in candidates)
    print(f"\n  전체 v5 후보 분포:")
    for t in V5_TOPICS:
        print(f"    {t}: {all_dist.get(t, 0)}건")
    print(f"    합계: {sum(all_dist.values())}건")

    # Opus 판정 분포
    opus_dist = Counter(c["v5_candidate"] for c in candidates if c["source"] == "opus")
    print(f"\n  Opus 판정 분포 (검수 대상):")
    for t in V5_TOPICS:
        cnt = opus_dist.get(t, 0)
        if cnt > 0:
            print(f"    {t}: {cnt}건")

    # Opus 판정 상세 출력 (검수용)
    print(f"\n{'='*60}")
    print(f"  Opus 판정 상세 ({len(need_opus)}건 — 검수 대상)")
    print(f"{'='*60}")
    for c in candidates:
        if c["source"] == "opus":
            print(f"\n  [{c['index']}] → {c['v5_candidate']} (conf={c['confidence']:.2f})")
            print(f"    v3c={c['v3c_topic']} v4={c['v4_3class']}")
            print(f"    reason: {c['reason'][:80]}")
            print(f"    text: {c['text'][:120]}")

    # 비용
    input_cost = total_input * 15.0 / 1_000_000
    output_cost = total_output * 75.0 / 1_000_000
    total_cost = input_cost + output_cost
    print(f"\n  비용: ${total_cost:.4f} (in:{total_input:,} / out:{total_output:,})")

    # 결과 저장
    output = {
        "experiment": "golden_v7_v5_relabel",
        "model": "claude-opus-4-6",
        "golden_set_size": len(golden),
        "rule_mapped": rule_count,
        "opus_classified": len(need_opus),
        "distribution": dict(all_dist),
        "opus_distribution": dict(opus_dist),
        "candidates": candidates,
        "cost": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(total_cost, 4),
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    output_path = "benchmarks/golden_v7_v5_candidates.json"
    os.makedirs('benchmarks', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {output_path}")


if __name__ == "__main__":
    main()
