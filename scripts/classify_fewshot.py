#!/usr/bin/env python3
"""Few-shot GPT-4o-mini 분류기

테스트셋을 도메인 컨텍스트 + 예시가 포함된 프롬프트로 분류

사용법:
    python scripts/classify_fewshot.py
"""
import json
import os
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 5개 카테고리
CATEGORIES = ["서비스 이슈", "서비스 칭찬", "투자 질문", "정보/의견", "일상 소통"]

# Few-shot 프롬프트 (도메인 컨텍스트 + 예시)
SYSTEM_PROMPT = """너는 금융 콘텐츠 플랫폼 '어스(us)'의 VOC 분석가야.

[플랫폼 개요]
- 마스터: 투자 전문가, 콘텐츠 크리에이터 (예: 박두환, 서재형 등)
- 오피셜클럽: 마스터가 운영하는 유료 구독 커뮤니티
- 편지글: 구독자→마스터 1:1 메시지 (감사, 일상, 소통이 주)
- 게시글: 커뮤니티 게시판 (정보 공유, 수익 후기 등)
- 서비스: 온라인 콘텐츠(강의, 레포트) + 오프라인 세미나

[분류 카테고리]
1. 서비스 이슈: 플랫폼 운영/기능 문제 (링크오류, 결제, 배송, 신청방법, 자료지연, 답변지연)
2. 서비스 칭찬: 콘텐츠/마스터에 대한 구체적 감사 ("덕분에 배웠다", "수익났다")
3. 투자 질문: 종목, 비중, 매매 타이밍 질문
4. 정보/의견: 증시 뉴스, 분석, 수익/손실 후기, 투자 심리
5. 일상 소통: 인사, 안부, 축하, 조문, 형식적 감사

[핵심 구분 - 매우 중요!]
- 증시/투자 이야기 ≠ 서비스 이슈 (아무리 부정적이어도)
  예: "요즘 장이 너무 힘들어요" → 정보/의견 (서비스 이슈 ❌)
- 구체적 감사 = 서비스 칭찬, 형식적 감사 = 일상 소통
  예: "감사합니다" → 일상 소통, "강의 덕분에 수익났어요" → 서비스 칭찬
- 운영/기능 문의 = 서비스 이슈
  예: "레포트 언제 올라오나요?" → 서비스 이슈 (투자 질문 ❌)

[분류 예시]
1. "링크가 안 열려요" → 서비스 이슈 (기술 오류)
2. "교재가 아직 안 왔어요" → 서비스 이슈 (배송 지연)
3. "강의 덕분에 많이 배웠어요" → 서비스 칭찬 (구체적 도움 언급)
4. "레포트 보고 삼전 매수했는데 10% 수익났습니다" → 서비스 칭찬 (콘텐츠 활용 결과)
5. "삼성전자 지금 사도 될까요?" → 투자 질문 (매매 타이밍)
6. "비중 어떻게 가져가면 좋을까요?" → 투자 질문 (포트폴리오 조언)
7. "오늘 시장 분석입니다" → 정보/의견 (시장 정보)
8. "요즘 장이 너무 힘들어요" → 정보/의견 (투자 심리)
9. "계좌가 녹아요 ㅠㅠ" → 정보/의견 (투자 심리)
10. "감사합니다" → 일상 소통 (형식적 감사)
11. "새해 복 많이 받으세요" → 일상 소통 (인사)

반드시 위 5개 카테고리 중 하나만 답변해."""


def classify_single(client: OpenAI, text: str) -> str:
    """단일 텍스트 분류"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"다음 글을 분류해줘:\n\n{text[:500]}"}
        ],
        temperature=0,
        max_tokens=20
    )
    result = response.choices[0].message.content.strip()

    # 카테고리 정규화
    for cat in CATEGORIES:
        if cat in result:
            return cat

    return result


def main():
    project_root = Path(__file__).parent.parent
    test_path = project_root / "data" / "test" / "test_set_100.json"
    domain_test_path = project_root / "data" / "test" / "domain_test_20.json"
    result_path = project_root / "data" / "results" / "fewshot_results.json"

    print("=" * 60)
    print("Few-shot GPT-4o-mini 분류")
    print("=" * 60)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = {
        "model": "gpt-4o-mini",
        "method": "few-shot",
        "main_test": [],
        "domain_test": []
    }

    # 1. 메인 테스트셋 분류
    print("\n1. 메인 테스트셋 분류 (100건)")
    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)

    correct = 0
    for i, item in enumerate(test_data):
        predicted = classify_single(client, item["text"])
        expected = item["category"]
        is_correct = predicted == expected

        results["main_test"].append({
            "id": item.get("id"),
            "text": item["text"][:200],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct
        })

        if is_correct:
            correct += 1

        if (i + 1) % 20 == 0:
            print(f"  진행: {i+1}/{len(test_data)} (정확도: {correct}/{i+1})")

    main_accuracy = correct / len(test_data) * 100
    print(f"\n  메인 테스트 정확도: {correct}/{len(test_data)} ({main_accuracy:.1f}%)")

    # 2. 도메인 이해도 테스트 분류
    print("\n2. 도메인 이해도 테스트 (20건)")
    with open(domain_test_path, encoding="utf-8") as f:
        domain_data = json.load(f)

    domain_correct = 0
    for item in domain_data:
        predicted = classify_single(client, item["text"])
        expected = item["category"]
        is_correct = predicted == expected

        results["domain_test"].append({
            "id": item.get("id"),
            "text": item["text"],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "reason": item.get("reason")
        })

        if is_correct:
            domain_correct += 1

        mark = "✓" if is_correct else "✗"
        print(f"  {mark} [{expected}] → [{predicted}] {item['text'][:40]}")

    domain_accuracy = domain_correct / len(domain_data) * 100
    print(f"\n  도메인 테스트 정확도: {domain_correct}/{len(domain_data)} ({domain_accuracy:.1f}%)")

    # 3. 결과 요약
    results["summary"] = {
        "main_accuracy": main_accuracy,
        "main_correct": correct,
        "main_total": len(test_data),
        "domain_accuracy": domain_accuracy,
        "domain_correct": domain_correct,
        "domain_total": len(domain_data)
    }

    # 카테고리별 정확도
    print("\n3. 카테고리별 정확도")
    for cat in CATEGORIES:
        cat_items = [r for r in results["main_test"] if r["expected"] == cat]
        cat_correct = sum(1 for r in cat_items if r["correct"])
        cat_acc = cat_correct / len(cat_items) * 100 if cat_items else 0
        print(f"  {cat}: {cat_correct}/{len(cat_items)} ({cat_acc:.1f}%)")

    # 4. 오분류 케이스 출력
    print("\n4. 메인 테스트 오분류 케이스 (최대 10건)")
    errors = [r for r in results["main_test"] if not r["correct"]][:10]
    for err in errors:
        print(f"  [{err['expected']}] → [{err['predicted']}]")
        print(f"    {err['text'][:60]}...")

    # 저장
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {result_path}")


if __name__ == "__main__":
    main()
