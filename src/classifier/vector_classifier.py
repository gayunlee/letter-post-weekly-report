"""벡터 유사도 기반 콘텐츠 분류 시스템 (하이브리드)"""
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from ..vectorstore.chroma_store import ChromaVectorStore
from .content_classifier import ContentClassifier

# 라벨링 데이터의 신규 카테고리 → 기존 카테고리 매핑
NEW_TO_OLD_CATEGORY = {
    "긍정 피드백": "감사·후기",
    "부정 피드백": "불편사항",
    "질문/문의": "질문·토론",
    "정보 공유": "정보성 글",
    "일상 소통": "일상·공감",
}


class VectorContentClassifier:
    """벡터 유사도 기반 콘텐츠 분류기 (k-NN 투표 방식)"""

    # 기본 학습 예제 (라벨링 데이터가 없을 때 fallback)
    TRAINING_EXAMPLES = [
        # ========== 감사·후기 (12개) ==========
        {"content": "감사합니다", "category": "감사·후기"},
        {"content": "정보 감사합니다", "category": "감사·후기"},
        {"content": "답변주셔서 고맙습니다", "category": "감사·후기"},
        {"content": "쌤 덕분에 투자의 눈을 떠가는 1인입니다. 정말 감사합니다.", "category": "감사·후기"},
        {"content": "항상 좋은 정보 감사드립니다. 덕분에 많이 배우고 있습니다.", "category": "감사·후기"},
        {"content": "쌤 믿고 기다리니 쭉쭉 올라갔어요. 신입분들 조급하게 생각마시고 기다리세요.", "category": "감사·후기"},
        {"content": "처음 두산우를 고점에 들어가서 몇달 고생했는데 믿고 기다리니 쭉쭉올라갔어요.", "category": "감사·후기"},
        {"content": "몰빵 종목 털어버리고 새로운 신학기로 가서 좋았어요.", "category": "감사·후기"},
        {"content": "쌤의 엄청난 노력으로 제가 돈 벌고 있음에 고맙고 감사합니다.", "category": "감사·후기"},
        {"content": "월간전망 강의듣고 집으로가는길입니다. 늘 감사드립니다.", "category": "감사·후기"},
        {"content": "오프라인 강의 참석했는데 정말 유익했습니다.", "category": "감사·후기"},
        {"content": "두환쌤 덕에 쉽게 주식합니다. 주식은 계속내리다가 잠깐 오르니깐요.", "category": "감사·후기"},

        # ========== 질문·토론 (12개) ==========
        {"content": "삼성전자를 스터디 목록에 편입하지 않으시는 이유가 궁금합니다.", "category": "질문·토론"},
        {"content": "포트폴리오 종목들 비중도 같이 알려주실수있나요?", "category": "질문·토론"},
        {"content": "비트코인 rsi는 일봉 기준인가요?", "category": "질문·토론"},
        {"content": "조정장이 올수도 있다고 하셨는데요 이게 4분기보다 조정이 클까요?", "category": "질문·토론"},
        {"content": "크립토는 편출 이슈 이후에 진입해도 늦지 않을까요?", "category": "질문·토론"},
        {"content": "개나리반하고 겨울학기하고 어떻게 차이 있나요?", "category": "질문·토론"},
        {"content": "sk이노베이션은 에너지와 이차전지 혼합인가요?", "category": "질문·토론"},
        {"content": "월세 보증금 비중반영은 어떻게 해야하나요?", "category": "질문·토론"},
        {"content": "이수페타시스는 스터디종목에서 제외된걸까요?", "category": "질문·토론"},
        {"content": "퇴직금 운영방식이 DB형과 DC형이 있다고해요. 어떤게 나을까요?", "category": "질문·토론"},
        {"content": "IRP는 어떻게 세팅해야할까요?", "category": "질문·토론"},
        {"content": "템퍼스 AI 주식을 갖고 있는데 계속 가지고 있어야 할지 팔아야 할지 모르겠어요.", "category": "질문·토론"},

        # ========== 정보성 글 (12개) ==========
        {"content": "AI 데이터센터 수요가 폭발적으로 증가하고 있습니다.", "category": "정보성 글"},
        {"content": "포트폴리오에서 반도체 비중을 16%에서 28%까지 올렸습니다.", "category": "정보성 글"},
        {"content": "지난밤 미국 원전주 포함 에너지주가 폭등했습니다.", "category": "정보성 글"},
        {"content": "정리된 종목표 챕터에 있습니다 참고요~", "category": "정보성 글"},
        {"content": "■ 마스터님은 유튜브와 어스플러스 외에 어떤 채팅방도 운영하지 않습니다.", "category": "정보성 글"},
        {"content": "속보 떴네요. 뉴스 링크 공유합니다.", "category": "정보성 글"},
        {"content": "작년 코스피 상승, 삼전·SK하닉 빼면 반토막", "category": "정보성 글"},
        {"content": "오늘 코스피 삼전,닉스 빼면 지수가 마이너스. 하락종목수가 더 많아요.", "category": "정보성 글"},
        {"content": "[지노믹트리] 식약처 승인이 연장되더니 결국 취소됐습니다.", "category": "정보성 글"},
        {"content": "현재 PER 34 저평가다. 보수적으로 PER 36으로 계산해보면 18% 추가 상승 가능합니다.", "category": "정보성 글"},
        {"content": "올해 1년간 결산입니다. 한국자산 +166.8%, 해외자산 +23.9% 수익률입니다.", "category": "정보성 글"},
        {"content": "제 의견을 드리자면 두 종목간에 매수매도하여 평단을 낮추는건 크게 의미가 없을듯합니다.", "category": "정보성 글"},

        # ========== 서비스 피드백 (10개) ==========
        {"content": "강의 자료 링크가 연결되지 않습니다. 확인 부탁드립니다.", "category": "서비스 피드백"},
        {"content": "녹음 파일이 안보이던데 시간이 더 걸리는 건가요?", "category": "서비스 피드백"},
        {"content": "컴퍼런스 일정 안내를 받지 못했습니다. 언제 하는지 알 수 있을까요?", "category": "서비스 피드백"},
        {"content": "오프라인 강의 초대는 어떻게 받나요?", "category": "서비스 피드백"},
        {"content": "유튜브에 사칭하는 광고가 있어 제보합니다.", "category": "서비스 피드백"},
        {"content": "TV모니터 미러링 기능 추가해 주시기를 바래봅니다.", "category": "서비스 피드백"},
        {"content": "책 도착했는데 1권만 배송됐어요. 나머지 2권은 언제 배송될까요?", "category": "서비스 피드백"},
        {"content": "오프라인초대 다 마감된건가요? 연락을 못받은거면 탈락 된건가요?", "category": "서비스 피드백"},
        {"content": "홍매화반 교재 언제 보내주세요?", "category": "서비스 피드백"},
        {"content": "상품 변경 칸이 없어서요~ 변경이 가능할까요?", "category": "서비스 피드백"},

        # ========== 불편사항 (10개) ==========
        {"content": "앱이 자꾸 튕겨요. 너무 답답합니다.", "category": "불편사항"},
        {"content": "결제를 했는데 강의가 안 열려요. 정말 불편합니다.", "category": "불편사항"},
        {"content": "매번 같은 질문인데 왜 답변을 안 해주시는 건가요? 소통이 안 되는 느낌입니다.", "category": "불편사항"},
        {"content": "구독료 대비 콘텐츠가 너무 적은 것 같아요.", "category": "불편사항"},
        {"content": "여전히 소외감이 느껴집니다. 시장은 좋은데 우리종목 대부분 마이너스입니다.", "category": "불편사항"},
        {"content": "종목이 너무많습니다. 원금 1억이 가치없는 휴지가 된것같습니다.", "category": "불편사항"},
        {"content": "비중 평단을 제시하지 않아서 정말 답답합니다.", "category": "불편사항"},
        {"content": "좋은기업만 알려주시지 비중을 안알려주셔서 40만원에 산분도 90만원에 산분도 있어요.", "category": "불편사항"},
        {"content": "일주일에 한번도 안 올라오는 컨텐츠에 무엇을 기대할까요...", "category": "불편사항"},
        {"content": "문의를 드렸는데 답이 없네요. 실망입니다.", "category": "불편사항"},

        # ========== 일상·공감 (10개) ==========
        {"content": "새해 복 많이 받으세요!", "category": "일상·공감"},
        {"content": "축하드립니다! 너무 너무 축하드려요.", "category": "일상·공감"},
        {"content": "주말은 잘 쉬시길 바랍니다.", "category": "일상·공감"},
        {"content": "안녕하세요 가입인사드립니다. 동행하게 되어 기쁜마음입니다.", "category": "일상·공감"},
        {"content": "1등 매니져 따라하기 3기 가입인사드립니다.", "category": "일상·공감"},
        {"content": "오늘 아침 브리핑 시원합니다~~ 오늘도 홧팅입니다!", "category": "일상·공감"},
        {"content": "원전 월요일 가겠죠? 기다림의 미학입니다.", "category": "일상·공감"},
        {"content": "오래 기다린만큼 많이 기대됩니다. 열심히 따라해보려구요.", "category": "일상·공감"},
        {"content": "두환쌤 인기 때문인지 이상한 사람들이 많이 생기네요. 마음이 아픕니다.", "category": "일상·공감"},
        {"content": "요즘 부담감이 심해보여 한마디 남깁니다. 너무 큰 책임감으로 스스로를 옥죄지마세요.", "category": "일상·공감"},
    ]

    def __init__(
        self,
        collection_name: str = "classification_guide",
        use_llm_fallback: bool = False,
        confidence_threshold: float = 0.3,
        embedding_model: str = None,
        k_neighbors: int = 5,
    ):
        """
        Args:
            collection_name: ChromaDB 컬렉션 이름
            use_llm_fallback: confidence가 낮을 때 LLM으로 재분류 여부
            confidence_threshold: LLM fallback을 위한 confidence 임계값
            embedding_model: 임베딩 모델명 (None이면 기본 한국어 모델)
            k_neighbors: k-NN 투표에 사용할 이웃 수
        """
        self.collection_name = collection_name
        self.use_llm_fallback = use_llm_fallback
        self.confidence_threshold = confidence_threshold
        self.k_neighbors = k_neighbors
        self.llm_classifier = None
        self.llm_fallback_count = 0

        self.store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory="./chroma_db",
            embedding_model=embedding_model,
        )

        # LLM fallback 활성화시 분류기 초기화
        if use_llm_fallback:
            try:
                self.llm_classifier = ContentClassifier()
            except Exception as e:
                print(f"⚠️ LLM 분류기 초기화 실패: {e}")
                self.use_llm_fallback = False

        # 학습 예제가 저장되어 있는지 확인
        if self.store.collection.count() == 0:
            print("분류 가이드 데이터 초기화 중...")
            self._initialize_training_data()

    def _initialize_training_data(self):
        """학습 예제를 벡터 스토어에 저장 (라벨링 데이터 우선, 없으면 기본 예제)"""
        count = 0

        # 1) 라벨링 데이터 로드 시도
        labeling_file = Path(__file__).parent.parent.parent / "data" / "labeling" / "refined_labeled.json"
        if labeling_file.exists():
            with open(labeling_file, encoding="utf-8") as f:
                labeled_data = json.load(f)

            for item in labeled_data:
                text = item.get("text", "").strip()
                new_label = item.get("new_label", "")
                if not text or not new_label:
                    continue

                # 신규 카테고리 → 기존 카테고리 매핑
                old_category = NEW_TO_OLD_CATEGORY.get(new_label)
                if not old_category:
                    continue

                self.store.add_content(
                    content_id=f"labeled_{item.get('id', count)}",
                    text=text[:500],
                    metadata={"category": old_category},
                )
                count += 1

            print(f"  라벨링 데이터 {count}건 로드")

        # 2) 기본 예제도 추가 (라벨링 데이터와 중복될 수 있지만 다양성 보장)
        for i, example in enumerate(self.TRAINING_EXAMPLES):
            self.store.add_content(
                content_id=f"example_{i}",
                text=example["content"],
                metadata={"category": example["category"]},
            )
            count += 1

        print(f"  총 {count}건 학습 예제 저장 완료")

    def classify_content(self, content: str) -> Dict[str, Any]:
        """
        단일 콘텐츠를 k-NN 투표로 분류

        cosine distance 사용: distance=0이면 동일, distance=1이면 직교
        confidence = 1 - distance (cosine similarity)

        Args:
            content: 분류할 콘텐츠 텍스트

        Returns:
            {"category": str, "confidence": float, "method": str}
        """
        if not content or len(content.strip()) == 0:
            return {
                "category": "내용 없음",
                "confidence": 0.0,
                "method": "empty",
            }

        # k개의 유사한 예제 검색
        similar = self.store.search_similar(
            query_text=content[:500],
            n_results=self.k_neighbors,
        )

        if not similar:
            return {
                "category": "미분류",
                "confidence": 0.0,
                "method": "none",
            }

        # k-NN 가중 투표: similarity(=1-distance)를 가중치로 사용
        vote_weights = Counter()
        for match in similar:
            category = match["metadata"].get("category", "미분류")
            distance = match.get("distance", 1.0)
            similarity = max(0.0, 1.0 - distance)
            vote_weights[category] += similarity

        # 최다 득표 카테고리
        if not vote_weights:
            return {"category": "미분류", "confidence": 0.0, "method": "none"}

        best_category = vote_weights.most_common(1)[0][0]
        total_weight = sum(vote_weights.values())
        confidence = vote_weights[best_category] / total_weight if total_weight > 0 else 0.0

        # LLM fallback: confidence가 낮으면 LLM으로 재분류
        if self.use_llm_fallback and self.llm_classifier and confidence < self.confidence_threshold:
            try:
                llm_result = self.llm_classifier.classify_content(content)
                self.llm_fallback_count += 1
                return {
                    "category": llm_result.get("category", best_category),
                    "confidence": 0.8 if llm_result.get("confidence") == "높음" else 0.6,
                    "method": "llm",
                    "reason": llm_result.get("reason", ""),
                }
            except Exception:
                pass

        return {
            "category": best_category,
            "confidence": round(confidence, 4),
            "method": "vector",
        }

    def classify_batch(
        self,
        contents: List[Dict[str, Any]],
        content_field: str = "message",
    ) -> List[Dict[str, Any]]:
        """
        여러 콘텐츠를 일괄 분류 (k-NN 투표 + 선택적 LLM fallback)

        Args:
            contents: 분류할 콘텐츠 리스트
            content_field: 콘텐츠 텍스트가 포함된 필드명

        Returns:
            분류 결과가 추가된 콘텐츠 리스트
        """
        results = []
        self.llm_fallback_count = 0

        for i, item in enumerate(contents):
            content_text = item.get(content_field, "")

            classification = self.classify_content(content_text)

            result = item.copy()
            result["classification"] = classification
            results.append(result)

            if (i + 1) % 100 == 0:
                llm_info = f" (LLM: {self.llm_fallback_count}건)" if self.use_llm_fallback else ""
                print(f"  진행: {i + 1}/{len(contents)} 완료{llm_info}")

        if self.use_llm_fallback and self.llm_fallback_count > 0:
            print(f"  → LLM fallback 사용: {self.llm_fallback_count}건")

        return results
