"""ChromaDB를 사용한 벡터 스토어"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 기본 임베딩 모델: 한국어 특화 sentence-transformers
DEFAULT_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"


class ChromaVectorStore:
    """ChromaDB를 사용한 콘텐츠 벡터 저장소"""

    def __init__(
        self,
        collection_name: str = "weekly_contents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = None,
    ):
        """
        ChromaVectorStore 초기화

        Args:
            collection_name: 컬렉션 이름
            persist_directory: 데이터 저장 디렉토리
            embedding_model: sentence-transformers 모델명 (None이면 기본 한국어 모델 사용)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

        # 임베딩 함수 초기화
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )

        # 컬렉션 가져오기 또는 생성 (cosine distance 사용)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

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

    def search_similar_batch(
        self,
        query_texts: List[str],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        여러 쿼리를 한 번에 배치 검색 (임베딩을 배치로 처리하여 속도 향상)

        Args:
            query_texts: 검색 쿼리 텍스트 리스트
            n_results: 각 쿼리당 반환할 결과 수
            filter_metadata: 메타데이터 필터

        Returns:
            각 쿼리에 대한 유사 콘텐츠 리스트의 리스트
        """
        if not query_texts:
            return []

        results = self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=filter_metadata,
        )

        batch_results = []
        if results and results["ids"]:
            for q_idx in range(len(results["ids"])):
                similar_contents = []
                for i in range(len(results["ids"][q_idx])):
                    similar_contents.append({
                        "id": results["ids"][q_idx][i],
                        "text": results["documents"][q_idx][i],
                        "metadata": results["metadatas"][q_idx][i] if results["metadatas"] else {},
                        "distance": results["distances"][q_idx][i] if results.get("distances") else None,
                    })
                batch_results.append(similar_contents)
        else:
            batch_results = [[] for _ in query_texts]

        return batch_results

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
        for category in ["감사·후기", "질문·토론", "정보성 글", "서비스 피드백", "일상·공감"]:
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
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
