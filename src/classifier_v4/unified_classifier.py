"""UnifiedClassifier — Single LLM call로 topic + sentiment + tags + summary 추출

3분류 체계: 대응 필요 / 콘텐츠·투자 / 노이즈
Summary chain-of-thought: "요약 후 분류" 방식 (R2 실험에서 +4.0%p)
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 VOC 데이터 분류기입니다.

## 분류 절차
1. 먼저 이 텍스트의 **핵심 의도를 1문장으로 요약**하세요.
2. 요약문을 기반으로 아래 우선순위에 따라 분류하세요.

## 분류 우선순위

### 1순위: 대응 필요
담당자가 이 글에 **1:1로 회신하거나 시스템 처리**를 해야 하는 건만 해당.
- 환불 요청, 해지 요청/통보, 구독·멤버십 등급 변경 문의
- 앱 오류, 로그인 실패, 링크 오류, 사칭 제보 (우리 서비스 문제)
- 세미나 문의, 배송 문의, 가격 정책 질문 (개별 답변 필요)
- 커뮤니티 운영 정책 비판: 운영진/알바/관리자 활동 비판, 규칙·정책 비판
- ⚠️ 환불/해지/구독 관련 불만은 콘텐츠 맥락이 섞여 있어도 → 대응 필요
- ⚠️ 외부 플랫폼(코인원, 바이낸스, 증권사 앱 등) 기술 문제는 우리 서비스가 아님 → 대응 필요 아님
- ⚠️ 콘텐츠 주제/형식 제안, 서비스 개선 의견은 → 콘텐츠·투자 (1:1 처리건이 아닌 피드백)
- ⚠️ 투자 관점·콘텐츠 품질에 대한 비판/불만은 → 콘텐츠·투자 (환불/해지/운영 요구가 없으면)

### 2순위: 콘텐츠·투자 ⭐ 가장 넓은 범위
투자/콘텐츠/마스터/시장/종목과 **조금이라도** 관련된 모든 글.
아래 신호가 **하나라도** 있으면 무조건 콘텐츠·투자:
- 종목명, 섹터, ETF, 지수, 코인 언급
- 수익/손실/매수/매도/포트폴리오/리밸런싱
- 마스터의 분석/강의/브리핑/콘텐츠에 대한 반응 (칭찬이든 비판이든)
- 시장 상황, 거시경제, 금리, 환율, 투자 이벤트(주주총회 등)
- 학습/공부 경험 언급 ("공부했습니다", "훈련", "배움", "성장")
- 투자 심리/멘탈 관련 언급
- 멤버십 불만이지만 콘텐츠 품질이 이유이고, 해지/환불 요구가 명시적이지 않은 경우
- 감사/응원 + 위 신호 중 하나 → 콘텐츠·투자
- **애매하면 → 콘텐츠·투자** (투자 교육 커뮤니티이므로 기본값)

### 3순위: 노이즈
위 1~2에 **전혀** 해당하지 않는 것. 투자/콘텐츠 신호가 **0개**일 때만:
- 순수 인사·잡담·응원: 투자/학습 내용 없이 마스터 인격에 감사만 표현
- 무의미 텍스트: ".", "1", 자음만, 밈, 한 줄 감탄
- ⚠️ "선생님 감사합니다 건강하세요" (투자/학습 신호 0개) → 노이즈
- ⚠️ "감사합니다 강의 잘 듣겠습니다" (콘텐츠명만, 내용 반응 없음) → 노이즈

## 경계 케이스 예시

### 콘텐츠·투자 vs 노이즈
- "두환쌤 방송 눈물이 납니다 감사드립니다" → 노이즈 (마스터 인격 감사, 투자/학습 신호 없음)
- "두환쌤 방송에서 말한 삼전 분석 덕분에 수익났습니다" → 콘텐츠·투자 (종목+수익)
- "아침 브리핑 들으면 멘탈 잡는데 도움이 됩니다" → 콘텐츠·투자 (콘텐츠 반응+투자 심리)
- "거시경제 분석이 최고입니다 감사합니다" → 콘텐츠·투자 (콘텐츠 내용 칭찬)

### ⚠️ 투자 비판/손실 감정 vs 대응 필요 (핵심 경계)
부정적이더라도 투자/콘텐츠에 대한 비판·감정이면 → 콘텐츠·투자 (sentiment=부정).
대응 필요는 **운영팀이 실제로 처리해야 할 요청**만 해당.
- "전체자산으로 투자하라더니 이게 뭐냐 책임져라" → 콘텐츠·투자 (투자 조언 비판, 감정 표출)
- "계좌 공개해서 본인도 물려있다는 걸 증명해라" → 콘텐츠·투자 (마스터 신뢰 비판)
- "내 피같은 돈 날렸는데 진짜 화가 난다" → 콘텐츠·투자 (투자 손실 감정)
- "매달 구독료가 아깝고 콘텐츠가 반복된다" → 콘텐츠·투자 (콘텐츠 품질 비판, 해지 요구 없음)
- "1기부터 있었는데 너무하네요 저도 이만 갑니다" → 콘텐츠·투자 (이탈 의사 표현이지 해지 요청 아님)
- "제미나이 응답이 느려졌네요" → 콘텐츠·투자 (외부 플랫폼 이슈, 우리 서비스 아님)

### 제안/피드백 vs 대응 필요 (핵심: 1:1 처리가 필요한가?)
제안/피드백/의견은 대응 필요가 아님 → 콘텐츠·투자 + 태그로 분류.
- "종이책으로 만들어주세요 / 클라우드 특집 다뤄주세요" → 콘텐츠·투자 (콘텐츠 제안, 1:1 처리건 아님)
- "라이브에서 게시판 질문도 답변해주세요" → 콘텐츠·투자 (방송 형식 제안)
- "세미나 영상 빨리 올려주세요" → 콘텐츠·투자 (서비스 피드백, 개별 회신 불필요)
- "사원들도 들을 수 있게 해주면 좋겠다" → 콘텐츠·투자 (운영 의견)

### 대응 필요 (1:1 회신/처리가 필요한 것만)
- "핑계 그만대고 구독자 돈 물어내" → 대응 필요 (환불 요구)
- "해지 신청합니다" → 대응 필요 (해지 요청)
- "팀장에서 인턴으로 변경 가능한가요?" → 대응 필요 (구독 등급 변경 문의)
- "커뮤 운영 방식이 이상하다 댓글 알바 풀지 마" → 대응 필요 (운영 정책 비판)
- "앱이 안 열려요" → 대응 필요 (서비스 오류)
- "세미나 신청은 어디서 하나요?" → 대응 필요 (운영 문의, 개별 답변 필요)

## 감정 분류
- 긍정: 감사, 칭찬, 만족, 수익 기쁨
- 부정: 불만, 실망, 손실 아쉬움, 항의
- 중립: 질문, 정보 전달, 단순 인사

## 태그 추출
텍스트에서 핵심 키워드 2~4개를 추출하세요.
피드백/제안 성격의 글에는 아래 태그를 포함하세요:
- 콘텐츠 제안: 콘텐츠 주제/형식/방송 방식 요청
- 서비스 피드백: 서비스 개선 의견, 기능 제안
- 운영 의견: 운영 방식에 대한 의견 (정책 비판이 아닌 제안)

## 마스터란?
투자 교육 커뮤니티를 운영하는 금융 콘텐츠 크리에이터입니다.

## 응답: JSON만 출력
{"summary": "핵심 요약", "topic": "콘텐츠·투자", "sentiment": "긍정", "tags": ["태그1", "태그2"], "confidence": 0.92}"""

V4_TOPICS = ["대응 필요", "콘텐츠·투자", "노이즈"]
V4_SENTIMENTS = ["긍정", "부정", "중립"]


def _parse_json_response(raw: str) -> dict:
    """LLM 응답에서 JSON 추출. 코드펜스/이중중괄호 처리."""
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    if raw.startswith("{{") and raw.endswith("}}"):
        raw = raw[1:-1]
    return json.loads(raw)


class UnifiedClassifier:
    """Single LLM call로 topic + sentiment + tags + summary 추출"""

    def __init__(self, model="claude-haiku-4-5-20251001", max_workers=10):
        self.client = Anthropic()
        self.model = model
        self.max_workers = max_workers
        self._input_tokens = 0
        self._output_tokens = 0
        self._errors = 0
        self._total = 0

    def classify_single(self, text: str) -> dict:
        """단일 건 분류 → {topic, sentiment, tags, summary, confidence}"""
        if not text or len(text.strip()) < 2:
            return {
                "topic": "노이즈",
                "sentiment": "중립",
                "tags": [],
                "summary": "",
                "confidence": 1.0,
            }

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=350,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text[:500]}],
                timeout=30.0,
            )

            self._input_tokens += response.usage.input_tokens
            self._output_tokens += response.usage.output_tokens

            result = _parse_json_response(response.content[0].text.strip())

            topic = result.get("topic", "콘텐츠·투자")
            if topic not in V4_TOPICS:
                topic = "콘텐츠·투자"

            sentiment = result.get("sentiment", "중립")
            if sentiment not in V4_SENTIMENTS:
                sentiment = "중립"

            tags = result.get("tags", [])
            if not isinstance(tags, list):
                tags = []

            return {
                "topic": topic,
                "sentiment": sentiment,
                "tags": tags[:4],
                "summary": str(result.get("summary", ""))[:200],
                "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            }

        except Exception as e:
            self._errors += 1
            return {
                "topic": "콘텐츠·투자",
                "sentiment": "중립",
                "tags": [],
                "summary": "",
                "confidence": 0.0,
                "error": str(e)[:80],
            }

    def classify_batch(self, items, content_field="message"):
        """ThreadPoolExecutor 병렬 분류. 각 item에 classification 필드 부착."""
        results = [None] * len(items)
        start_time = time.time()
        self._total = len(items)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {}
            for i, item in enumerate(items):
                text = item.get(content_field, "") if isinstance(item, dict) else str(item)
                future = executor.submit(self.classify_single, text)
                future_map[future] = i

            done_count = 0
            for future in as_completed(future_map):
                idx = future_map[future]
                classification = future.result()
                if isinstance(items[idx], dict):
                    items[idx]["classification"] = classification
                results[idx] = classification
                done_count += 1
                if done_count % 50 == 0 or done_count == len(items):
                    elapsed = time.time() - start_time
                    rate = done_count / elapsed if elapsed > 0 else 0
                    print(f"    {done_count}/{len(items)} 완료 ({rate:.1f}건/초)")

        return items

    def get_cost_report(self) -> dict:
        """비용 추적 리포트"""
        input_cost = self._input_tokens * 0.80 / 1_000_000
        output_cost = self._output_tokens * 4.00 / 1_000_000
        return {
            "model": self.model,
            "total_items": self._total,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cost_usd": round(input_cost + output_cost, 4),
            "errors": self._errors,
        }
