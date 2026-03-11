"""POC: 기존 2축 분류 데이터에 detail_tags 추출"""
import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

client = Anthropic()

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC(Voice of Customer) 데이터 분석가입니다.
사용자가 보낸 편지글이나 게시글을 읽고, 구체적인 세부 태그를 추출합니다.

규칙:
1. 2~4개의 태그를 추출하세요
2. 각 태그는 2~5단어의 구체적 명사구여야 합니다
3. 태그는 다른 팀이 검색/필터링할 때 유용해야 합니다
4. 추상적 태그(예: "불만", "긍정") 대신 구체적 내용(예: "포트폴리오 비중 질문", "게시판 폐쇄 불만")을 추출하세요
5. 이 플랫폼에서 마스터란 투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다

반드시 아래 JSON 형식으로만 응답하세요:
{"tags": ["태그1", "태그2", "태그3"], "summary": "한 줄 요약"}"""


def extract_tags(text: str, topic: str, sentiment: str) -> dict:
    """LLM으로 detail_tags 추출"""
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"[분류: {topic} / {sentiment}]\n\n{text[:500]}"
        }]
    )
    try:
        raw = response.content[0].text
        # 마크다운 코드 블록 제거
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return result
    except (json.JSONDecodeError, IndexError):
        return {"tags": [], "summary": "파싱 실패", "raw": response.content[0].text}


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

    # 각 그룹에서 샘플링 (부정은 더 많이, 긍정/중립은 적게)
    sample_sizes = {
        "부정": 8,
        "중립": 4,
        "긍정": 3,
    }

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

            print(f"\n  [{source}] {master}")
            print(f"  원문: {text[:120]}{'...' if len(text) > 120 else ''}")

            tag_result = extract_tags(text, topic, sentiment)
            print(f"  태그: {tag_result.get('tags', [])}")
            print(f"  요약: {tag_result.get('summary', '')}")

            results.append({
                "id": item.get("_id", ""),
                "source": source,
                "master": master,
                "topic": topic,
                "sentiment": sentiment,
                "text_preview": text[:200],
                "detail_tags": tag_result.get("tags", []),
                "summary": tag_result.get("summary", ""),
            })
            total_count += 1

    # 결과 저장
    output_path = "data/poc_detail_tags_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 태그 빈도 분석
    print(f"\n\n{'='*70}")
    print(f"  태그 빈도 분석 (총 {total_count}건)")
    print(f"{'='*70}")

    from collections import Counter
    all_tags = Counter()
    for r in results:
        for tag in r["detail_tags"]:
            all_tags[tag] += 1

    print("\n  전체 태그 빈도:")
    for tag, cnt in all_tags.most_common(30):
        print(f"    {tag:30s} | {cnt}회")

    # Topic별 태그 빈도
    for topic in ["투자 이야기", "서비스 이슈", "콘텐츠 반응", "커뮤니티 소통"]:
        topic_tags = Counter()
        for r in results:
            if r["topic"] == topic:
                for tag in r["detail_tags"]:
                    topic_tags[tag] += 1
        if topic_tags:
            print(f"\n  [{topic}] 태그 빈도:")
            for tag, cnt in topic_tags.most_common(10):
                print(f"    {tag:30s} | {cnt}회")

    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    random.seed(42)
    main()
