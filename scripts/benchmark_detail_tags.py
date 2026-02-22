"""Detail Tags 모델별 비용/품질 벤치마크

동일 샘플에 대해 Claude Sonnet, Haiku, GPT-4o-mini를 비교합니다.
측정 항목: 파싱 성공률, 유효 태그 비율, 건당 비용, 처리 속도

사용법:
    python3 scripts/benchmark_detail_tags.py
    python3 scripts/benchmark_detail_tags.py --sample-size 30
    python3 scripts/benchmark_detail_tags.py --data-file data/classified_data_two_axis/2026-02-09.json
"""
import sys
import os
import json
import time
import random
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from src.classifier_v2.detail_tag_extractor import (
    DetailTagExtractor,
    CATEGORY_TAGS,
    ALL_VALID_TAGS,
)


# GPT-4o-mini용 추출기 (OpenAI API)
class OpenAIDetailTagExtractor:
    """GPT-4o-mini 기반 detail_tags 추출 (벤치마크 비교용)"""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CALLME_OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        self._parse_failures = 0
        self._invalid_tag_count = 0

    def extract_tags(self, text: str, topic: str, sentiment: str) -> dict:
        if not text or len(text.strip()) < 10:
            return {"category_tags": [], "free_tags": [], "summary": "", "parse_ok": True}

        tags = CATEGORY_TAGS.get(topic, [])
        tag_list = "\n".join(f"- {t}" for t in tags)

        sys_prompt = f"""당신은 금융 교육 플랫폼의 VOC 데이터 분석가입니다.
편지글/게시글을 읽고 2종류의 태그를 추출합니다.

## 태그 추출 규칙
### 1. 카테고리 태그 (category_tags) — 아래 목록에서 1~2개 선택
{tag_list}

### 2. 자유 태그 (free_tags) — 2~3개 구체적 명사구 (2~5단어)
### 3. 한 줄 요약 (summary) — 15~40자

## 응답 형식 — JSON만 출력:
{{"category_tags": ["태그1"], "free_tags": ["태그1", "태그2"], "summary": "한 줄 요약"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=300,
                temperature=0,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"[감성: {sentiment}]\n\n{text[:500]}"},
                ],
            )
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens
            self._total_calls += 1

            raw = response.choices[0].message.content.strip()
            return self._parse(raw, topic)
        except Exception as e:
            self._total_calls += 1
            self._parse_failures += 1
            return {"category_tags": [], "free_tags": [], "summary": str(e)[:50], "parse_ok": False}

    def _parse(self, raw: str, topic: str) -> dict:
        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            if text.startswith("{{") and text.endswith("}}"):
                text = text[1:-1]
            result = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            self._parse_failures += 1
            return {"category_tags": [], "free_tags": [], "summary": "파싱 실패", "parse_ok": False}

        valid_for_topic = set(CATEGORY_TAGS.get(topic, []))
        validated = [t for t in result.get("category_tags", []) if t in valid_for_topic]
        self._invalid_tag_count += len(result.get("category_tags", [])) - len(validated)

        return {
            "category_tags": validated,
            "free_tags": result.get("free_tags", [])[:3],
            "summary": result.get("summary", "")[:60],
            "parse_ok": True,
        }

    def get_cost_report(self) -> dict:
        # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
        input_cost = self._total_input_tokens / 1_000_000 * 0.15
        output_cost = self._total_output_tokens / 1_000_000 * 0.60
        total_cost = input_cost + output_cost
        return {
            "model": self.model,
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "parse_failures": self._parse_failures,
            "invalid_tags": self._invalid_tag_count,
            "parse_success_rate": (
                round((1 - self._parse_failures / self._total_calls) * 100, 1)
                if self._total_calls > 0 else 0
            ),
            "estimated_cost_usd": round(total_cost, 6),
            "cost_per_item_usd": round(total_cost / self._total_calls, 6) if self._total_calls > 0 else 0,
        }


def load_sample(data_file: str, sample_size: int) -> list:
    """분류 완료 데이터에서 샘플 로드"""
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    all_items = data.get("letters", []) + data.get("posts", [])
    # 텍스트가 있는 건만
    valid = []
    for item in all_items:
        text = item.get("message") or item.get("textBody") or ""
        if text and len(text.strip()) >= 10:
            valid.append(item)

    if len(valid) <= sample_size:
        return valid

    # Topic × Sentiment 균형 샘플링
    groups = {}
    for item in valid:
        cls = item.get("classification", {})
        key = (cls.get("topic", "미분류"), cls.get("sentiment", "미분류"))
        groups.setdefault(key, []).append(item)

    sampled = []
    per_group = max(1, sample_size // len(groups))
    for key, items in groups.items():
        n = min(per_group, len(items))
        sampled.extend(random.sample(items, n))

    # 부족분 채우기
    if len(sampled) < sample_size:
        remaining = [i for i in valid if i not in sampled]
        sampled.extend(random.sample(remaining, min(sample_size - len(sampled), len(remaining))))

    return sampled[:sample_size]


def run_benchmark(samples: list, extractor, model_name: str) -> dict:
    """단일 모델 벤치마크 실행"""
    print(f"\n{'='*60}")
    print(f"  {model_name} ({len(samples)}건)")
    print(f"{'='*60}")

    results = []
    start = time.time()

    for i, item in enumerate(samples):
        cls = item.get("classification", {})
        topic = cls.get("topic", "커뮤니티 소통")
        sentiment = cls.get("sentiment", "중립")
        text = item.get("message") or item.get("textBody") or ""

        result = extractor.extract_tags(text, topic, sentiment)
        results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"  {i+1}/{len(samples)} ({elapsed:.1f}초)")

    elapsed = time.time() - start
    cost = extractor.get_cost_report()

    # 품질 메트릭
    parse_ok = sum(1 for r in results if r.get("parse_ok", False))
    has_tags = sum(1 for r in results if r.get("category_tags"))
    has_free = sum(1 for r in results if r.get("free_tags"))
    has_summary = sum(1 for r in results if r.get("summary") and r["summary"] != "파싱 실패")

    tag_counter = Counter()
    for r in results:
        for t in r.get("category_tags", []):
            tag_counter[t] += 1

    report = {
        "model": model_name,
        "sample_size": len(samples),
        "elapsed_sec": round(elapsed, 1),
        "items_per_sec": round(len(samples) / elapsed, 2) if elapsed > 0 else 0,
        "parse_success_rate": round(parse_ok / len(samples) * 100, 1),
        "tag_coverage": round(has_tags / len(samples) * 100, 1),
        "free_tag_coverage": round(has_free / len(samples) * 100, 1),
        "summary_coverage": round(has_summary / len(samples) * 100, 1),
        "unique_tags": len(tag_counter),
        "top_tags": tag_counter.most_common(10),
        **cost,
    }

    print(f"\n  결과:")
    print(f"    시간: {report['elapsed_sec']}초 ({report['items_per_sec']}건/초)")
    print(f"    파싱 성공: {report['parse_success_rate']}%")
    print(f"    태그 커버리지: {report['tag_coverage']}%")
    print(f"    비용: ${report['estimated_cost_usd']:.4f} (건당 ${report['cost_per_item_usd']:.6f})")
    print(f"    목록 외 태그: {report['invalid_tags']}건")

    return report


def main():
    parser = argparse.ArgumentParser(description="Detail Tags 모델 벤치마크")
    parser.add_argument("--data-file", default="data/classified_data_two_axis/2026-02-09.json",
                        help="분류 완료 데이터 파일")
    parser.add_argument("--sample-size", type=int, default=30, help="샘플 크기")
    parser.add_argument("--models", nargs="+",
                        default=["haiku", "sonnet", "gpt4omini"],
                        choices=["haiku", "sonnet", "gpt4omini"],
                        help="벤치마크할 모델")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"데이터 파일을 찾을 수 없습니다: {args.data_file}")
        return

    random.seed(42)
    samples = load_sample(args.data_file, args.sample_size)
    print(f"샘플 {len(samples)}건 로드 완료")

    # Topic 분포 확인
    topic_dist = Counter(
        item.get("classification", {}).get("topic", "미분류") for item in samples
    )
    for topic, cnt in topic_dist.most_common():
        print(f"  {topic}: {cnt}건")

    # 모델별 벤치마크
    model_map = {
        "haiku": ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        "sonnet": ("claude-sonnet-4-20250514", "Claude Sonnet 4"),
        "gpt4omini": ("gpt-4o-mini", "GPT-4o-mini"),
    }

    reports = []
    for model_key in args.models:
        model_id, model_name = model_map[model_key]

        if model_key == "gpt4omini":
            extractor = OpenAIDetailTagExtractor(model=model_id)
        else:
            extractor = DetailTagExtractor(model=model_id, max_workers=1)

        report = run_benchmark(samples, extractor, model_name)
        reports.append(report)

    # 비교 요약
    print(f"\n\n{'='*70}")
    print(f"  비교 요약 ({len(samples)}건 기준)")
    print(f"{'='*70}")
    print(f"\n{'모델':20s} | {'파싱%':>6s} | {'태그%':>6s} | {'속도':>8s} | {'건당비용':>10s} | {'주간추정':>10s}")
    print("-" * 70)

    weekly_estimate = 2500  # 주간 예상 건수
    for r in reports:
        weekly_cost = r["cost_per_item_usd"] * weekly_estimate
        print(
            f"{r['model']:20s} | {r['parse_success_rate']:5.1f}% | {r['tag_coverage']:5.1f}% | "
            f"{r['items_per_sec']:6.2f}/s | ${r['cost_per_item_usd']:.6f} | ${weekly_cost:.2f}"
        )

    # 결과 저장
    output_path = "data/benchmark_detail_tags.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Counter는 JSON 직렬화 불가 → top_tags를 리스트로 변환
        for r in reports:
            r["top_tags"] = [[t, c] for t, c in r["top_tags"]]
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
