"""ChannelClassifier — 채널톡 CS 문의 다중 태그 분류 (Haiku API)

2축 분류 중 Axis 2 (topics) 담당.
Axis 1 (route)는 규칙 기반으로 channel_preprocessor.detect_route()에서 처리.

5분류 (다중 태그): 결제·환불 / 구독·멤버십 / 콘텐츠·수강 / 기술·오류 / 기타
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CHANNEL_SYSTEM_PROMPT = """당신은 금융 교육 플랫폼의 CS 문의 분류기입니다.
채널톡으로 접수된 고객 상담을 분류합니다.

## 핵심 원칙

채널톡 CS에 문의를 남겼다는 것은 **명확한 의도 + 액션 대기** 상태입니다.
"~하고 싶다", "~할 수 있나요"는 명시적 요청으로 봅니다.

## 분류 규칙

아래 5개 주제 중 **관련된 주제를 모두** topics에 포함하세요.
하나만 해당되면 하나만, 여러 개 해당되면 여러 개 넣으세요.

### 결제·환불
결제, 환불, 계좌이체, 카드 결제/변경, 영수증 요청, 의도치 않은 자동결제.
- "환불해주세요", "결제했는데 반영이 안 돼요", "영수증 발급"
- "결제코드 보내주세요" → 결제 진행 목적
- "날짜 지난 걸 모르고 결제됐다" → 의도치 않은 결제

### 구독·멤버십
구독 신청, 해지, 등급 변경, 갱신, 멤버십 관련. 아직 이용하지 않는 강의에 대한 신청 의사도 여기.
- "구독 해지하고 싶어요", "상품 변경", "멤버십 취소"
- "다음 결제일에 결제 안 되게 해주세요" → 해지
- "함께하고 싶습니다", "강좌 듣고 싶어요" → 가입/신청 의향
- 구독 상품 간 차이/비교 문의

### 콘텐츠·수강
**이미 이용 중인** 강의 접근, 수강 방법, 자료/교재 요청, 이용 방법. 오프라인 행사/세미나/특강도 여기.
- "강의 어디서 보나요", "수강 방법 알려주세요"
- "교재 배송 현황", "어플 깔아야 하나요"
- 오프라인 강의 장소/시간, 세미나 참석, QR코드, 동반자 참석
- 마스터 콘텐츠 누락 불만
- 구독 범위/포함 콘텐츠 문의

### 기술·오류
**확실히 시스템 문제**인 경우. 불확실하면 콘텐츠·수강으로.
- 로그인 안 됨, 앱 크래시, 서버 오류, "멤버가 아니라고 뜸" (구독 중인데)
- 결제 완료 + 접근 차단 (시스템 원인 명확)
- 앱/플랫폼 기술 기능 건의 ("프린트 가능하게 해주세요")

### 기타
위 4개에 해당하지 않는 문의.
- 단순 인사, 투자 질문, 번호 변경
- 플랫폼 기능 추가/개선 건의 (검색, 자막 등). 단 기술적 기능 건의는 기술·오류
- 워크플로우 버튼만 누르고 실제 문의 내용 없음

## 다중 주제 처리

여러 주제가 섞여 있으면 **관련된 주제를 모두 topics에 넣으세요**.
- "구독 해지 + 환불해주세요" → topics: ["결제·환불", "구독·멤버십"]
- "로그인 안 됨 + 환불" → topics: ["기술·오류", "결제·환불"]
- "강의 접근 안 됨 + 해지하려는데" → topics: ["콘텐츠·수강", "구독·멤버십"]
하나만 해당되면 하나만 넣으면 됩니다.

## 워크플로우 힌트

사용자 메시지에 [워크플로우: ...] 접두사가 있으면 참고하세요.
단, 실제 텍스트 내용이 버튼과 다르면 텍스트 기준으로 분류하세요.

## tags 작성 규칙

tags에는 문의에 등장하는 세부 키워드를 포함하세요.
예: tags: ["환불", "구독해지", "영수증"]

## 응답: JSON만 출력

{"topics": ["결제·환불", "구독·멤버십"], "summary": "핵심 요약 1문장", "tags": ["환불", "구독해지"], "confidence": 0.92}"""

CHANNEL_TOPICS = ["결제·환불", "구독·멤버십", "콘텐츠·수강", "기술·오류", "기타"]


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


class ChannelClassifier:
    """채널톡 CS 문의 다중 태그 분류 (Haiku API)"""

    def __init__(self, model="claude-haiku-4-5-20251001", max_workers=10):
        self.client = Anthropic()
        self.model = model
        self.max_workers = max_workers
        self._input_tokens = 0
        self._output_tokens = 0
        self._errors = 0
        self._total = 0

    def classify_single(
        self, text: str, workflow_buttons: Optional[List[str]] = None
    ) -> dict:
        """단일 건 분류 → {topics, summary, tags, confidence}"""
        if not text or len(text.strip()) < 2:
            return {
                "topics": ["기타"],
                "summary": "",
                "tags": [],
                "confidence": 1.0,
            }

        # 워크플로우 버튼 접두사 추가
        prompt_text = text[:500]
        if workflow_buttons:
            button_str = ", ".join(workflow_buttons)
            prompt_text = f"[워크플로우: {button_str}]\n{prompt_text}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=250,
                system=CHANNEL_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt_text}],
                timeout=30.0,
            )

            self._input_tokens += response.usage.input_tokens
            self._output_tokens += response.usage.output_tokens

            result = _parse_json_response(response.content[0].text.strip())

            # topics 배열 파싱 (하위 호환: topic 단일값도 처리)
            topics = result.get("topics", result.get("topic", ["기타"]))
            if isinstance(topics, str):
                topics = [topics]
            topics = [t for t in topics if t in CHANNEL_TOPICS]
            if not topics:
                topics = ["기타"]

            tags = result.get("tags", [])
            if not isinstance(tags, list):
                tags = []

            return {
                "topics": topics,
                "summary": str(result.get("summary", ""))[:200],
                "tags": tags[:6],
                "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
            }

        except Exception as e:
            self._errors += 1
            if self._errors == 1:
                print(f"    [첫 에러] {type(e).__name__}: {str(e)[:120]}")
            return {
                "topics": ["기타"],
                "summary": "",
                "tags": [],
                "confidence": 0.0,
                "error": str(e)[:80],
            }

    def classify_batch(
        self, items: List[Dict[str, Any]], content_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """ThreadPoolExecutor 병렬 분류. 각 item에 classification 필드 부착."""
        results = [None] * len(items)
        start_time = time.time()
        self._total = len(items)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {}
            for i, item in enumerate(items):
                text = item.get(content_field, "")
                buttons = item.get("workflow_buttons", [])
                future = executor.submit(self.classify_single, text, buttons)
                future_map[future] = i

            done_count = 0
            for future in as_completed(future_map):
                idx = future_map[future]
                classification = future.result()
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
