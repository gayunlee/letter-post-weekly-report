"""Few-shot learning 기반 콘텐츠 분류 시스템"""
import os
from typing import List, Dict, Any
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ContentClassifier:
    """Claude API를 사용한 콘텐츠 분류기"""

    # 분류 카테고리 정의
    CATEGORIES = {
        "감사·후기": "마스터에 대한 감사, 긍정적 피드백, 투자 성과 후기",
        "질문·토론": "포트폴리오, 종목, 투자 전략, 비중, 세팅 방법에 대한 질문 및 토론",
        "정보성 글": "투자 경험 공유, 종목 분석, 뉴스/정보 공유, 수익률 공유",
        "서비스 피드백": "플랫폼/서비스 기능 문의, 강의자료/링크 문제, 배송/일정 문의",
        "불편사항": "불만 표현, 답답함, 소외감, 부정적 감정, 서비스에 대한 불만",
        "일상·공감": "새해인사, 안부, 축하, 가입인사, 일상 이야기, 공감 표현"
    }

    # Few-shot 예제
    FEW_SHOT_EXAMPLES = """
예제 1:
내용: "쌤과 인연이 되어 양때목장의 양이 된지 2개월 남짓 됬네요. 두환쌤 덕분에 투자의 눈을 떠가는 1인입니다."
분류: 감사·후기
이유: 마스터에 대한 감사와 긍정적인 피드백을 표현하고 있습니다.

예제 2:
내용: "26년도 포트폴리오 구성할때 샘이 생각하시는 방향으로 가고 싶은데, 삼성전자나 하이닉스를 스터디 목록에 편입하지 않으시는 이유가 궁금합니다."
분류: 질문·토론
이유: 포트폴리오 구성 및 종목 선택에 대한 질문을 하고 있습니다.

예제 3:
내용: "제 직업은 반도체 설계 엔지니어입니다. 엔비디아나 브로드컴 같은 글로벌 팹리스 기업에 대해 분석해봤습니다."
분류: 정보성 글
이유: 전문적인 지식을 바탕으로 투자 정보를 공유하고 있습니다.

예제 4:
내용: "19회차 라이프 강의 자료 링크가 첨부 파일로 연결되지 않습니다. 확인 부탁드립니다."
분류: 서비스 피드백
이유: 플랫폼의 기능적 문제점을 제기하고 있습니다.

예제 5:
내용: "오로지 희망님 기쁜 소식 축하드립니다! 너무 너무 축하드려요. 담쌤 수면 시간 충분히 늘리셔요."
분류: 일상·공감
이유: 축하와 안부를 전하는 일상적인 공감 표현입니다.

예제 6:
내용: "일주일에 한번도 안 올라오는 컨텐츠에 무엇을 기대할까요..."
분류: 불편사항
이유: 서비스에 대한 불만과 답답함을 표현하고 있습니다.

예제 7:
내용: "새해 복 많이 받으세요! 건강하고 행복한 한 해 되세요."
분류: 일상·공감
이유: 새해 인사와 안부를 전하는 표현입니다.
"""

    def __init__(self, api_key: str = None):
        """
        ContentClassifier 초기화

        Args:
            api_key: Anthropic API 키 (None일 경우 환경변수에서 로드)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")

        self.client = Anthropic(api_key=self.api_key)

    def classify_content(self, content: str) -> Dict[str, Any]:
        """
        단일 콘텐츠 분류

        Args:
            content: 분류할 콘텐츠 텍스트

        Returns:
            {"category": str, "confidence": str, "reason": str}
        """
        # 카테고리 설명 생성
        category_desc = "\n".join([
            f"- {cat}: {desc}"
            for cat, desc in self.CATEGORIES.items()
        ])

        # 분류 프롬프트 구성
        prompt = f"""다음은 금융 콘텐츠 크리에이터 플랫폼의 사용자가 작성한 글입니다.
이 글을 아래 카테고리 중 하나로 분류해주세요.

[분류 카테고리]
{category_desc}

[Few-shot 예제]
{self.FEW_SHOT_EXAMPLES}

[분류할 내용]
{content[:500]}

위 내용을 가장 적합한 카테고리 하나로 분류하고, 다음 형식으로 답변해주세요:

분류: [카테고리명]
확신도: [높음/중간/낮음]
이유: [1-2문장으로 분류 이유 설명]"""

        try:
            # Claude API 호출
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # 응답 파싱
            response_text = message.content[0].text.strip()

            # 응답에서 정보 추출
            lines = response_text.split('\n')
            category = None
            confidence = None
            reason = None

            for line in lines:
                if line.startswith('분류:'):
                    category = line.replace('분류:', '').strip()
                elif line.startswith('확신도:'):
                    confidence = line.replace('확신도:', '').strip()
                elif line.startswith('이유:'):
                    reason = line.replace('이유:', '').strip()

            return {
                "category": category or "분류 불가",
                "confidence": confidence or "중간",
                "reason": reason or "응답 파싱 실패",
                "raw_response": response_text
            }

        except Exception as e:
            return {
                "category": "분류 오류",
                "confidence": "없음",
                "reason": f"API 호출 오류: {str(e)}",
                "error": str(e)
            }

    def classify_batch(
        self,
        contents: List[Dict[str, Any]],
        content_field: str = "message"
    ) -> List[Dict[str, Any]]:
        """
        여러 콘텐츠를 일괄 분류

        Args:
            contents: 분류할 콘텐츠 리스트
            content_field: 콘텐츠 텍스트가 포함된 필드명

        Returns:
            분류 결과가 추가된 콘텐츠 리스트
        """
        results = []

        for i, item in enumerate(contents):
            content_text = item.get(content_field, "")

            if not content_text or len(content_text.strip()) == 0:
                classification = {
                    "category": "내용 없음",
                    "confidence": "없음",
                    "reason": "분류할 내용이 없습니다."
                }
            else:
                classification = self.classify_content(content_text)

            # 원본 데이터에 분류 결과 추가
            result = item.copy()
            result["classification"] = classification
            results.append(result)

            # 진행 상황 출력 (10건마다)
            if (i + 1) % 10 == 0:
                print(f"  진행: {i + 1}/{len(contents)} 완료")

        return results
