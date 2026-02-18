"""벡터 유사도 기반 콘텐츠 분류 시스템 (하이브리드)"""
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from ..vectorstore.chroma_store import ChromaVectorStore
from .content_classifier import ContentClassifier


def preprocess_text(text: str) -> str:
    """분류 전 텍스트 전처리: 노이즈 제거 및 정규화"""
    if not text:
        return text

    # URL 제거
    text = re.sub(r'https?://\S+', '', text)

    # 반복 문자 정규화 (4회 이상 → 2회): ㅋㅋㅋㅋㅋ → ㅋㅋ, !!!!→ !!
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # 과도한 개행/공백 정규화
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 특수 포맷 문자 제거 (■, ★, ●, ◆ 등)
    text = re.sub(r'[■★●◆◇▶▷►▲△▽▼○◎□☆♥♡♣♠♦]', '', text)

    return text.strip()

# 라벨링 데이터의 신규 카테고리 → 기존 카테고리 매핑
NEW_TO_OLD_CATEGORY = {
    "긍정 피드백": "감사·후기",
    "부정 피드백": "불편사항",
    "질문/문의": "질문·토론",
    "정보 공유": "정보성 글",
    "일상 소통": "일상·공감",
}

OLD_TO_NEW_CATEGORY = {v: k for k, v in NEW_TO_OLD_CATEGORY.items()}
# 서비스 피드백은 불편사항과 같은 "부정 피드백"으로 매핑
OLD_TO_NEW_CATEGORY["서비스 피드백"] = "부정 피드백"


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
        k_neighbors: int = 3,
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
                    text=preprocess_text(text)[:500],
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

        # 전처리 적용
        content = preprocess_text(content)

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

    def _classify_from_neighbors(self, similar: List[Dict[str, Any]]) -> Dict[str, Any]:
        """k-NN 이웃 목록으로부터 분류 결과를 생성"""
        if not similar:
            return {"category": "미분류", "confidence": 0.0, "method": "none"}

        vote_weights = Counter()
        for match in similar:
            category = match["metadata"].get("category", "미분류")
            distance = match.get("distance", 1.0)
            similarity = max(0.0, 1.0 - distance)
            vote_weights[category] += similarity

        if not vote_weights:
            return {"category": "미분류", "confidence": 0.0, "method": "none"}

        best_category = vote_weights.most_common(1)[0][0]
        total_weight = sum(vote_weights.values())
        confidence = vote_weights[best_category] / total_weight if total_weight > 0 else 0.0

        return {
            "category": best_category,
            "confidence": round(confidence, 4),
            "method": "vector",
        }

    def classify_batch(
        self,
        contents: List[Dict[str, Any]],
        content_field: str = "message",
        batch_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        여러 콘텐츠를 배치 쿼리로 일괄 분류 (k-NN 투표 + 선택적 LLM fallback)

        ChromaDB 배치 쿼리로 임베딩 생성을 묶어서 처리하여 속도를 대폭 향상.

        Args:
            contents: 분류할 콘텐츠 리스트
            content_field: 콘텐츠 텍스트가 포함된 필드명
            batch_size: 한 번에 ChromaDB에 쿼리할 크기

        Returns:
            분류 결과가 추가된 콘텐츠 리스트
        """
        results = [None] * len(contents)
        self.llm_fallback_count = 0

        # 빈 콘텐츠와 유효 콘텐츠를 분리
        valid_indices = []
        valid_texts = []
        for i, item in enumerate(contents):
            text = item.get(content_field, "")
            if not text or len(text.strip()) == 0:
                result = item.copy()
                result["classification"] = {
                    "category": "내용 없음",
                    "confidence": 0.0,
                    "method": "empty",
                }
                results[i] = result
            else:
                valid_indices.append(i)
                valid_texts.append(preprocess_text(text)[:500])

        # 배치 단위로 ChromaDB 쿼리
        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]

            # 배치 쿼리: 한 번의 호출로 여러 텍스트의 유사도 검색
            batch_neighbors = self.store.search_similar_batch(
                query_texts=batch_texts,
                n_results=self.k_neighbors,
            )

            for j, (idx, neighbors) in enumerate(zip(batch_indices, batch_neighbors)):
                classification = self._classify_from_neighbors(neighbors)

                # LLM fallback
                if (self.use_llm_fallback and self.llm_classifier
                        and classification["confidence"] < self.confidence_threshold):
                    try:
                        content_text = contents[idx].get(content_field, "")
                        llm_result = self.llm_classifier.classify_content(content_text)
                        self.llm_fallback_count += 1
                        classification = {
                            "category": llm_result.get("category", classification["category"]),
                            "confidence": 0.8 if llm_result.get("confidence") == "높음" else 0.6,
                            "method": "llm",
                            "reason": llm_result.get("reason", ""),
                        }
                    except Exception:
                        pass

                result = contents[idx].copy()
                result["classification"] = classification
                results[idx] = result

            processed = min(batch_end, len(valid_texts))
            llm_info = f" (LLM: {self.llm_fallback_count}건)" if self.use_llm_fallback else ""
            print(f"  진행: {processed}/{len(valid_texts)} 완료{llm_info}")

        if self.use_llm_fallback and self.llm_fallback_count > 0:
            print(f"  → LLM fallback 사용: {self.llm_fallback_count}건")

        return results


class EnsembleClassifier:
    """벡터 분류기 + 파인튜닝 분류기 앙상블

    두 모델의 결과를 confidence 가중으로 결합합니다.
    카테고리는 새 5개 카테고리 체계로 통일합니다.
    """

    def __init__(
        self,
        vector_weight: float = 0.5,
        finetuned_weight: float = 0.5,
        **vector_kwargs,
    ):
        """
        Args:
            vector_weight: 벡터 분류기 기본 가중치
            finetuned_weight: 파인튜닝 분류기 기본 가중치
            **vector_kwargs: VectorContentClassifier에 전달할 인자
        """
        from ..classifier_v2.finetuned_classifier import FinetunedClassifier

        self.vector_weight = vector_weight
        self.finetuned_weight = finetuned_weight

        print("앙상블 분류기 초기화...")
        print("  [1/2] 벡터 분류기 로드")
        self.vector_clf = VectorContentClassifier(**vector_kwargs)
        print("  [2/2] 파인튜닝 분류기 로드")
        self.finetuned_clf = FinetunedClassifier()

    def _to_new_category(self, category: str) -> str:
        """기존 카테고리를 새 카테고리로 변환 (이미 새 카테고리면 그대로)"""
        return OLD_TO_NEW_CATEGORY.get(category, category)

    def classify_content(self, content: str) -> Dict[str, Any]:
        """단일 콘텐츠를 앙상블로 분류"""
        if not content or len(content.strip()) == 0:
            return {"category": "내용 없음", "confidence": 0.0, "method": "empty"}

        # 두 분류기 실행
        vec_result = self.vector_clf.classify_content(content)
        ft_result = self.finetuned_clf.classify_content(content)

        # 벡터 결과를 새 카테고리로 변환
        vec_cat = self._to_new_category(vec_result["category"])
        ft_cat = self._to_new_category(ft_result["category"])
        vec_conf = vec_result["confidence"]
        ft_conf = ft_result["confidence"]

        # 동의: confidence 부스트
        if vec_cat == ft_cat:
            return {
                "category": vec_cat,
                "confidence": round(min(1.0, (vec_conf + ft_conf) / 2 + 0.1), 4),
                "method": "ensemble_agree",
            }

        # 불일치: 벡터 분류기를 기본 신뢰 (정확도가 훨씬 높으므로)
        return {
            "category": vec_cat,
            "confidence": round(vec_conf * 0.9, 4),
            "method": "ensemble_vector",
        }

    def classify_batch(
        self,
        contents: List[Dict[str, Any]],
        content_field: str = "message",
        batch_size: int = 200,
    ) -> List[Dict[str, Any]]:
        """여러 콘텐츠를 앙상블로 일괄 분류"""
        print("  앙상블: 벡터 분류기 실행...")
        vec_results = self.vector_clf.classify_batch(contents, content_field, batch_size)
        print("  앙상블: 파인튜닝 분류기 실행...")
        ft_results = self.finetuned_clf.classify_batch(contents, content_field)

        results = []
        agree_count = 0
        for vec_item, ft_item in zip(vec_results, ft_results):
            vec_cls = vec_item["classification"]
            ft_cls = ft_item["classification"]

            # 빈 콘텐츠
            if vec_cls["method"] == "empty":
                results.append(vec_item)
                continue

            vec_cat = self._to_new_category(vec_cls["category"])
            ft_cat = self._to_new_category(ft_cls["category"])
            vec_conf = vec_cls["confidence"]
            ft_conf = ft_cls["confidence"]

            if vec_cat == ft_cat:
                agree_count += 1
                classification = {
                    "category": vec_cat,
                    "confidence": round(min(1.0, (vec_conf + ft_conf) / 2 + 0.1), 4),
                    "method": "ensemble_agree",
                }
            else:
                # 불일치: 벡터 분류기를 기본 신뢰
                classification = {
                    "category": vec_cat,
                    "confidence": round(vec_conf * 0.9, 4),
                    "method": "ensemble_vector",
                }

            result = vec_item.copy()
            result["classification"] = classification
            results.append(result)

        total = len([r for r in results if r["classification"]["method"] != "empty"])
        print(f"  앙상블 결과: 동의 {agree_count}/{total} ({agree_count/total*100:.1f}%)")

        return results
