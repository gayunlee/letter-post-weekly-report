"""ChromaDB를 사용한 벡터 스토어"""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()


class ChromaVectorStore:
    """ChromaDB를 사용한 콘텐츠 벡터 저장소"""

    # 한국어 임베딩 모델 (한국어 특화)
    EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

    def __init__(
        self,
        collection_name: str = "weekly_contents",
        persist_directory: str = "./chroma_db"
    ):
        """
        ChromaVectorStore 초기화

        Args:
            collection_name: 컬렉션 이름
            persist_directory: 데이터 저장 디렉토리
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )

        # 한국어 임베딩 함수 설정
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBEDDING_MODEL
        )

        # 컬렉션 가져오기 또는 생성 (한국어 임베딩 적용)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "주간 콘텐츠 벡터 저장소"}
        )

    def _generate_embedding(self, text: str) -> List[float]:
        """
        텍스트 임베딩 생성

        참고: 현재 Anthropic은 직접 임베딩 API를 제공하지 않으므로,
        간단한 방법으로 텍스트를 토큰화하여 임베딩을 생성합니다.
        실제 프로덕션에서는 OpenAI embedding API나 다른 임베딩 모델을 사용하는 것이 좋습니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터
        """
        # ChromaDB의 기본 임베딩 함수 사용 (sentence-transformers)
        # 별도의 임베딩 생성 없이 ChromaDB가 자동으로 처리하도록 함
        return None

    def add_content(
        self,
        content_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        콘텐츠 추가

        Args:
            content_id: 콘텐츠 고유 ID
            text: 콘텐츠 텍스트
            metadata: 메타데이터 (카테고리, 날짜 등)
        """
        self.collection.add(
            ids=[content_id],
            documents=[text],
            metadatas=[metadata] if metadata else None
        )

    def add_contents_batch(
        self,
        contents: List[Dict[str, Any]],
        id_field: str = "_id",
        text_field: str = "message"
    ) -> int:
        """
        여러 콘텐츠를 일괄 추가

        Args:
            contents: 콘텐츠 리스트
            id_field: ID 필드명
            text_field: 텍스트 필드명

        Returns:
            추가된 콘텐츠 수
        """
        ids = []
        documents = []
        metadatas = []

        for content in contents:
            content_id = str(content.get(id_field))
            text = content.get(text_field, "")

            if not content_id or not text:
                continue

            # 메타데이터 구성
            metadata = {
                "category": content.get("classification", {}).get("category", "미분류"),
                "createdAt": content.get("createdAt", ""),
                "masterId": content.get("masterId", ""),
                "type": "letter" if "message" in content else "post"
            }

            ids.append(content_id)
            documents.append(text[:1000])  # 최대 1000자까지만 저장
            metadatas.append(metadata)

        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

        return len(ids)

    def search_similar(
        self,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        유사한 콘텐츠 검색

        Args:
            query_text: 검색 쿼리 텍스트
            n_results: 반환할 결과 수
            filter_metadata: 메타데이터 필터

        Returns:
            유사 콘텐츠 리스트
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )

        # 결과 포맷팅
        similar_contents = []
        if results and results['ids']:
            for i in range(len(results['ids'][0])):
                similar_contents.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results.get('distances') else None
                })

        return similar_contents

    def get_by_category(
        self,
        category: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        카테고리별 콘텐츠 조회

        Args:
            category: 카테고리명
            limit: 최대 결과 수

        Returns:
            해당 카테고리의 콘텐츠 리스트
        """
        results = self.collection.get(
            where={"category": category},
            limit=limit
        )

        contents = []
        if results and results['ids']:
            for i in range(len(results['ids'])):
                contents.append({
                    "id": results['ids'][i],
                    "text": results['documents'][i],
                    "metadata": results['metadatas'][i] if results['metadatas'] else {}
                })

        return contents

    def get_stats(self) -> Dict[str, Any]:
        """
        저장소 통계 조회

        Returns:
            통계 정보
        """
        total_count = self.collection.count()

        # 카테고리별 통계
        category_stats = {}
        for category in ["감사·후기", "질문·토론", "정보성 글", "문의사항", "불편사항", "제보/건의", "일상·공감"]:
            results = self.collection.get(
                where={"category": category}
            )
            category_stats[category] = len(results['ids']) if results['ids'] else 0

        return {
            "total_count": total_count,
            "category_stats": category_stats,
            "collection_name": self.collection_name
        }

    def reset(self) -> None:
        """컬렉션 초기화 (모든 데이터 삭제)"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "주간 콘텐츠 벡터 저장소"}
        )
