#!/usr/bin/env python3
"""파인튜닝된 모델 테스트

사용법:
    python scripts/test_finetuned_model.py [model_id]
"""
import os
import sys
import json
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CATEGORIES = ["서비스 이슈", "서비스 칭찬", "투자 질문", "정보/의견", "일상 소통"]

SYSTEM_PROMPT = """너는 금융 콘텐츠 플랫폼의 VOC 분류 전문가야. 다음 5개 카테고리 중 하나로 분류해:
1. 서비스 이슈: 서비스/운영에 대한 불만, 문의, 개선요청
2. 서비스 칭찬: 콘텐츠/마스터/플랫폼에 대한 구체적 감사, 칭찬
3. 투자 질문: 종목, 비중, 매매 타이밍 등 투자 관련 질문
4. 정보/의견: 뉴스, 분석, 시장 전망, 수익/손실 후기, 투자 심리
5. 일상 소통: 인사, 안부, 축하, 조문, 가입인사"""

# 테스트 케이스
TEST_CASES = [
    ("링크가 안 열려요", "서비스 이슈"),
    ("강의 너무 좋았습니다. 덕분에 많이 배웠어요", "서비스 칭찬"),
    ("삼성전자 지금 매수해도 될까요?", "투자 질문"),
    ("오늘 10% 수익 봤어요", "정보/의견"),
    ("새해 복 많이 받으세요", "일상 소통"),
    ("교재가 아직 안 왔어요", "서비스 이슈"),
    ("감사합니다", "일상 소통"),
    ("오늘 시장 분석 공유합니다", "정보/의견"),
    ("비중을 어떻게 조절하면 좋을까요?", "투자 질문"),
    ("매번 좋은 콘텐츠 감사드립니다", "서비스 칭찬"),
]


def get_latest_finetuned_model(client):
    """가장 최근 파인튜닝 모델 ID 조회"""
    jobs = client.fine_tuning.jobs.list(limit=10)
    for job in jobs.data:
        if job.status == "succeeded" and job.fine_tuned_model:
            return job.fine_tuned_model
    return None


def classify(client, model_id, text):
    """텍스트 분류"""
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text[:500]}
        ],
        temperature=0,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 모델 ID 결정
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
    else:
        model_id = get_latest_finetuned_model(client)
        if not model_id:
            print("파인튜닝된 모델이 없습니다.")
            print("상태 확인: python scripts/check_finetune.py --list")
            return

    print("=" * 60)
    print("파인튜닝 모델 테스트")
    print("=" * 60)
    print(f"모델: {model_id}")

    # 1. 기본 테스트 케이스
    print("\n1. 기본 테스트 케이스")
    correct = 0
    for text, expected in TEST_CASES:
        result = classify(client, model_id, text)
        match = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"  {match} [{expected}] → [{result}] {text[:40]}")

    print(f"\n정확도: {correct}/{len(TEST_CASES)} ({correct/len(TEST_CASES)*100:.1f}%)")

    # 2. 검증 데이터로 테스트
    val_path = Path(__file__).parent.parent / "data" / "finetune" / "validation.jsonl"
    if val_path.exists():
        print("\n2. 검증 데이터 테스트")
        with open(val_path, encoding="utf-8") as f:
            val_data = [json.loads(line) for line in f]

        correct = 0
        predictions = []
        for item in val_data[:50]:  # 50건만 테스트
            text = item["messages"][1]["content"]
            expected = item["messages"][2]["content"]
            result = classify(client, model_id, text)
            predictions.append((expected, result))
            if result == expected:
                correct += 1

        print(f"  정확도: {correct}/50 ({correct/50*100:.1f}%)")

        # 혼동 행렬 간단히
        print("\n  오분류 케이스:")
        for expected, result in predictions:
            if expected != result:
                print(f"    {expected} → {result}")


if __name__ == "__main__":
    main()
