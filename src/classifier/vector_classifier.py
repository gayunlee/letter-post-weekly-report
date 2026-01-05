"""벡터 유사도 기반 콘텐츠 분류 시스템"""
from typing import List, Dict, Any
from ..vectorstore.chroma_store import ChromaVectorStore


class VectorContentClassifier:
    """벡터 유사도 기반 콘텐츠 분류기 (빠른 분류)"""

    # Few-shot 학습 예제 (분류 가이드)
    TRAINING_EXAMPLES = [
        # 감사·후기
        {
            "content": "쌤과 인연이 되어 양때목장의 양이 된지 2개월 남짓 됬네요. 두환쌤 덕분에 투자의 눈을 떠가는 1인입니다. 정말 감사합니다.",
            "category": "감사·후기"
        },
        {
            "content": "올해도 며칠안남았는데 마무리 잘하셨음하고 새해 복 많이 받으세요. 늘 감사하고 응원합니다.",
            "category": "감사·후기"
        },
        {
            "content": "전무님 올 한해 임직원분들과 함께 신세 많이 졌습니다. 감사합니다.",
            "category": "감사·후기"
        },
        # 질문·토론
        {
            "content": "26년도 포트폴리오 구성할때 샘이 생각하시는 방향으로 가고 싶은데, 삼성전자나 하이닉스를 스터디 목록에 편입하지 않으시는 이유가 궁금합니다.",
            "category": "질문·토론"
        },
        {
            "content": "매주 공유해주시는 포트폴리오 종목들 비중도 같이 알려주실수있나요?",
            "category": "질문·토론"
        },
        {
            "content": "잠재 GDP의 과거 데이터까지 모두 확인하는 방법을 알고 싶습니다.",
            "category": "질문·토론"
        },
        # 정보성 글
        {
            "content": "제 직업은 반도체 설계 엔지니어입니다. 엔비디아나 브로드컴 같은 글로벌 팹리스 기업에 대해 분석해봤습니다. AI 데이터센터 수요가 폭발적으로 증가하고 있습니다.",
            "category": "정보성 글"
        },
        {
            "content": "1-2주간 포트폴리오에서 가장큰 변화는 반도체의 비중을 많이 늘렸습니다. 16%에서 28%까지 올렸네요. 26년도 삼성전자, 하이닉스가 좋은 실적을 보일 것 같습니다.",
            "category": "정보성 글"
        },
        {
            "content": "금이 우리 포트보다 많이 올랐다. 은이 우리 포트보다 많이 올랐다. 투자 전략을 공유합니다.",
            "category": "정보성 글"
        },
        # 서비스 피드백 (중립적인 문의/요청)
        {
            "content": "19회차 라이프 강의 자료 링크가 첨부 파일로 연결되지 않습니다. 확인 부탁드립니다.",
            "category": "서비스 피드백"
        },
        {
            "content": "굿모닝 담샘 글이 올라오고 나서 바로 들어가 보면 녹음 파일이 안보이던데 녹음 파일 올라오는 데는 시간이 조금 더 걸리는 건가요?",
            "category": "서비스 피드백"
        },
        {
            "content": "컴퍼런스 일정 안내를 받지 못했습니다. 언제, 어디서 하는지 어떻게 알 수 있을까요?",
            "category": "서비스 피드백"
        },
        # 불편사항 (불만, 답답함, 개선 요청)
        {
            "content": "앱이 자꾸 튕겨요. 강의 보다가 끊기면 너무 답답합니다. 언제 고쳐지나요?",
            "category": "불편사항"
        },
        {
            "content": "결제를 했는데 강의가 안 열려요. 문의해도 답변이 없어서 정말 불편합니다.",
            "category": "불편사항"
        },
        {
            "content": "매번 같은 질문인데 왜 답변을 안 해주시는 건가요? 소통이 안 되는 느낌입니다.",
            "category": "불편사항"
        },
        {
            "content": "구독료 대비 콘텐츠가 너무 적은 것 같아요. 좀 더 자주 업데이트 해주셨으면 합니다.",
            "category": "불편사항"
        },
        {
            "content": "알림이 너무 많이 와서 불편해요. 알림 설정 기능이 있었으면 좋겠습니다.",
            "category": "불편사항"
        },
        # 일상·공감
        {
            "content": "오로지 희망님 기쁜 소식 축하드립니다! 너무 너무 축하드려요. 담쌤 수면 시간 충분히 늘리셔요.",
            "category": "일상·공감"
        },
        {
            "content": "새해 복 많이 받으세요! 모두 건강하시고 행복한 한 해 되시길 바랍니다.",
            "category": "일상·공감"
        },
        {
            "content": "오투님 이거 입히는것이 제 소원이었는데 겨우 구했습니다. 기쁩니다!",
            "category": "일상·공감"
        }
    ]

    def __init__(self, collection_name: str = "classification_guide"):
        """
        VectorContentClassifier 초기화

        Args:
            collection_name: ChromaDB 컬렉션 이름
        """
        self.collection_name = collection_name
        self.store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )

        # 학습 예제가 저장되어 있는지 확인
        if self.store.collection.count() == 0:
            print("분류 가이드 데이터 초기화 중...")
            self._initialize_training_data()
            print(f"✓ {len(self.TRAINING_EXAMPLES)}개 예제 저장 완료")

    def _initialize_training_data(self):
        """Few-shot 학습 예제를 벡터 스토어에 저장"""
        for i, example in enumerate(self.TRAINING_EXAMPLES):
            self.store.add_content(
                content_id=f"example_{i}",
                text=example["content"],
                metadata={"category": example["category"]}
            )

    def classify_content(self, content: str) -> Dict[str, Any]:
        """
        단일 콘텐츠를 벡터 유사도로 분류

        Args:
            content: 분류할 콘텐츠 텍스트

        Returns:
            {"category": str, "confidence": float, "similar_example": str}
        """
        if not content or len(content.strip()) == 0:
            return {
                "category": "내용 없음",
                "confidence": 0.0,
                "similar_example": ""
            }

        # 가장 유사한 예제 검색
        similar = self.store.search_similar(
            query_text=content[:500],  # 처음 500자만 사용
            n_results=1
        )

        if similar and len(similar) > 0:
            top_match = similar[0]
            category = top_match["metadata"].get("category", "미분류")

            # 거리를 confidence로 변환 (거리가 가까울수록 높은 confidence)
            # distance는 0에 가까울수록 유사함
            distance = top_match.get("distance", 1.0)
            confidence = max(0.0, min(1.0, 1.0 - distance))

            return {
                "category": category,
                "confidence": confidence,
                "similar_example": top_match["text"][:100]
            }
        else:
            return {
                "category": "미분류",
                "confidence": 0.0,
                "similar_example": ""
            }

    def classify_batch(
        self,
        contents: List[Dict[str, Any]],
        content_field: str = "message"
    ) -> List[Dict[str, Any]]:
        """
        여러 콘텐츠를 일괄 분류 (벡터 유사도 기반)

        Args:
            contents: 분류할 콘텐츠 리스트
            content_field: 콘텐츠 텍스트가 포함된 필드명

        Returns:
            분류 결과가 추가된 콘텐츠 리스트
        """
        results = []

        for i, item in enumerate(contents):
            content_text = item.get(content_field, "")

            classification = self.classify_content(content_text)

            # 원본 데이터에 분류 결과 추가
            result = item.copy()
            result["classification"] = classification
            results.append(result)

            # 진행 상황 출력 (100건마다)
            if (i + 1) % 100 == 0:
                print(f"  진행: {i + 1}/{len(contents)} 완료")

        return results
