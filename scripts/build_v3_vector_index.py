"""v3 벡터 인덱스 구축 스크립트

Opus 라벨링 데이터(labeled_all.json)를 벡터화하여
data/vectorstore_v3/ (ChromaDB)에 저장.

사용법:
    python3 scripts/build_v3_vector_index.py
    python3 scripts/build_v3_vector_index.py --data data/training_data/v3/labeled_all.json
    python3 scripts/build_v3_vector_index.py --k 7
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.classifier_v3.vector_v3_classifier import VectorV3Classifier


def main():
    parser = argparse.ArgumentParser(description="v3 벡터 인덱스 구축")
    parser.add_argument(
        "--data",
        default="data/training_data/v3/labeled_all.json",
        help="Opus 라벨링 데이터 경로",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/vectorstore_v3",
        help="ChromaDB 저장 디렉토리",
    )
    parser.add_argument("--k", type=int, default=7, help="KNN k값")
    args = parser.parse_args()

    print("=" * 60)
    print("  v3 벡터 인덱스 구축")
    print("=" * 60)
    print(f"  입력: {args.data}")
    print(f"  출력: {args.persist_dir}")
    print(f"  k: {args.k}")

    clf = VectorV3Classifier(persist_dir=args.persist_dir, k=args.k)
    stats = clf.build_index(args.data)

    print(f"\n{'='*60}")
    print(f"  총 임베딩: {stats['total']}건")
    print(f"  인덱스 경로: {args.persist_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
