"""분류 가이드 벡터 컬렉션 리셋 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vectorstore.chroma_store import ChromaVectorStore
from src.classifier.vector_classifier import VectorContentClassifier


def main():
    print("=" * 60)
    print("분류 가이드 벡터 컬렉션 리셋")
    print("=" * 60)

    # 1. 기존 컬렉션 리셋
    print("\n[1단계] 기존 classification_guide 컬렉션 리셋")
    store = ChromaVectorStore(
        collection_name="classification_guide",
        persist_directory="./chroma_db"
    )

    old_count = store.collection.count()
    print(f"  기존 예제 수: {old_count}")

    store.reset()
    print("  ✓ 컬렉션 리셋 완료")

    # 2. 새로운 예제로 재초기화
    print("\n[2단계] 새로운 분류 가이드 초기화")
    classifier = VectorContentClassifier()

    new_count = classifier.store.collection.count()
    print(f"  ✓ 새로운 예제 수: {new_count}")

    # 3. 카테고리별 예제 수 확인
    print("\n[3단계] 카테고리별 예제 수 확인")
    categories = {}
    for example in VectorContentClassifier.TRAINING_EXAMPLES:
        cat = example["category"]
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}개")

    print("\n" + "=" * 60)
    print("✅ 분류 가이드 리셋 완료!")
    print("   이제 '불편사항' 카테고리가 포함됩니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
