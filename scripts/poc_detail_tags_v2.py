"""POC v2: 카테고리 태그(controlled) + 자유 태그(free) 동시 추출"""
import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

client = Anthropic()

# Topic별 카테고리 태그 목록 (POC v1에서 수렴한 패턴 기반)
CATEGORY_TAGS = {
    "서비스 이슈": [
        "결제/환불/구독",
        "앱/기능 오류",
        "게시판/커뮤니티 운영",
        "온보딩/접근성",
        "배송/일정",
        "가격/프로모션 정책",
        "콘텐츠 접근 문제",
        "기타 서비스",
    ],
    "투자 이야기": [
        "포트폴리오 전략/비중",
        "개별 종목 분석",
        "시장 전망/매크로",
        "수익/손실 공유",
        "매매 타이밍",
        "섹터/테마 분석",
        "투자 학습 질문",
        "기타 투자",
    ],
    "콘텐츠 반응": [
        "콘텐츠 품질/깊이",
        "마스터 소통/태도",
        "강의/수업 피드백",
        "리포트/브리핑 피드백",
        "콘텐츠 주제 요청",
        "기타 콘텐츠",
    ],
    "커뮤니티 소통": [
        "인사/안부/감사",
        "투자 경험 공유",
        "마스터 응원/격려",
        "커뮤니티 분위기",
        "일상 공유",
        "기타 소통",
    ],
}

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC(Voice of Customer) 데이터 분석가입니다.
사용자가 보낸 편지글이나 게시글을 읽고, 2종류의 태그를 추출합니다.

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

## 태그 추출 규칙

### 1. 카테고리 태그 (category_tags)
아래 목록에서 1~2개를 선택하세요. 반드시 목록에 있는 것만 선택하세요.
{category_list}

### 2. 자유 태그 (free_tags)
2~3개의 구체적 명사구를 추출하세요 (2~5단어).
다른 팀이 검색할 때 유용한 구체적 내용이어야 합니다.
예: "게시판 폐쇄 사전 공지 부족", "ARKK 포트폴리오 비중 논란"

### 3. 한 줄 요약 (summary)
15자~40자 내외로 핵심 내용을 요약하세요.

## 응답 형식
반드시 아래 JSON만 출력하세요. 다른 텍스트 없이:
{{"category_tags": ["태그1"], "free_tags": ["태그1", "태그2"], "summary": "한 줄 요약"}}"""


def build_prompt(topic: str) -> str:
    """Topic에 맞는 카테고리 목록을 포함한 시스템 프롬프트 생성"""
    tags = CATEGORY_TAGS.get(topic, [])
    tag_list = "\n".join(f"- {t}" for t in tags)
    return SYSTEM_PROMPT.replace("{category_list}", tag_list)


def extract_tags(text: str, topic: str, sentiment: str) -> dict:
    """LLM으로 카테고리 태그 + 자유 태그 추출"""
    sys_prompt = build_prompt(topic)
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=300,
        system=sys_prompt,
        messages=[{
            "role": "user",
            "content": f"[감성: {sentiment}]\n\n{text[:500]}"
        }]
    )
    try:
        raw = response.content[0].text
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return result
    except (json.JSONDecodeError, IndexError):
        return {"category_tags": [], "free_tags": [], "summary": "파싱 실패",
                "raw": response.content[0].text}


def main():
    with open("data/classified_data_two_axis/2026-02-09.json") as f:
        data = json.load(f)

    all_items = data["letters"] + data["posts"]

    # Topic x Sentiment 별로 그룹핑
    groups = {}
    for item in all_items:
        c = item["classification"]
        key = (c["topic"], c["sentiment"])
        groups.setdefault(key, []).append(item)

    # 샘플 사이즈: 부정 더 많이, 전체적으로 좀 더 뽑기
    sample_sizes = {"부정": 10, "중립": 5, "긍정": 4}

    results = []
    total_count = 0

    for (topic, sentiment), items in sorted(groups.items()):
        n = min(sample_sizes.get(sentiment, 3), len(items))
        sampled = random.sample(items, n)

        print(f"\n{'='*70}")
        print(f"  {topic} × {sentiment}  ({len(items)}건 중 {n}건 샘플)")
        print(f"{'='*70}")

        for item in sampled:
            text = item.get("message") or item.get("textBody") or item.get("title", "")
            if not text or len(text.strip()) < 10:
                continue

            source = "편지" if "message" in item else "게시글"
            master = item.get("masterName", "?")

            tag_result = extract_tags(text, topic, sentiment)

            cat_tags = tag_result.get("category_tags", [])
            free_tags = tag_result.get("free_tags", [])
            summary = tag_result.get("summary", "")

            print(f"\n  [{source}] {master}")
            print(f"  원문: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"  카테고리: {cat_tags}")
            print(f"  자유태그: {free_tags}")
            print(f"  요약: {summary}")

            results.append({
                "id": item.get("_id", ""),
                "source": source,
                "master": master,
                "topic": topic,
                "sentiment": sentiment,
                "topic_confidence": item["classification"].get("topic_confidence", 0),
                "sentiment_confidence": item["classification"].get("sentiment_confidence", 0),
                "text_preview": text[:200],
                "category_tags": cat_tags,
                "free_tags": free_tags,
                "summary": summary,
            })
            total_count += 1

    # 결과 저장
    output_path = "data/poc_detail_tags_v2_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # === 분석 ===
    from collections import Counter

    print(f"\n\n{'='*70}")
    print(f"  분석 결과 (총 {total_count}건)")
    print(f"{'='*70}")

    # 카테고리 태그 빈도 (전체)
    cat_counter = Counter()
    for r in results:
        for tag in r["category_tags"]:
            cat_counter[tag] += 1

    print(f"\n  카테고리 태그 빈도 (전체):")
    for tag, cnt in cat_counter.most_common():
        bar = "█" * cnt
        print(f"    {tag:25s} | {cnt:>3} | {bar}")

    # Topic별 카테고리 태그 분포
    for topic in ["서비스 이슈", "투자 이야기", "콘텐츠 반응", "커뮤니티 소통"]:
        topic_cat = Counter()
        topic_free = Counter()
        for r in results:
            if r["topic"] == topic:
                for tag in r["category_tags"]:
                    topic_cat[tag] += 1
                for tag in r["free_tags"]:
                    topic_free[tag] += 1

        if topic_cat:
            print(f"\n  [{topic}] 카테고리 태그:")
            for tag, cnt in topic_cat.most_common():
                print(f"    {tag:25s} | {cnt}")

            print(f"  [{topic}] 자유 태그 상위 10:")
            for tag, cnt in topic_free.most_common(10):
                print(f"    {tag:30s} | {cnt}")

    # Sentiment별 카테고리 태그 분포
    for sentiment in ["부정", "중립", "긍정"]:
        sent_cat = Counter()
        for r in results:
            if r["sentiment"] == sentiment:
                for tag in r["category_tags"]:
                    sent_cat[tag] += 1
        if sent_cat:
            print(f"\n  [감성: {sentiment}] 카테고리 태그:")
            for tag, cnt in sent_cat.most_common(10):
                print(f"    {tag:25s} | {cnt}")

    # 목록 외 태그 검증
    all_valid = set()
    for tags in CATEGORY_TAGS.values():
        all_valid.update(tags)

    invalid_tags = Counter()
    for r in results:
        for tag in r["category_tags"]:
            if tag not in all_valid:
                invalid_tags[tag] += 1

    if invalid_tags:
        print(f"\n  ⚠️ 목록 외 카테고리 태그 (LLM이 만든 것):")
        for tag, cnt in invalid_tags.most_common():
            print(f"    {tag:25s} | {cnt}")

    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    random.seed(42)
    main()
