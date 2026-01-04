"""BigQuery 데이터베이스 탐색 스크립트"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient
import json


def main():
    """BigQuery 데이터셋과 테이블 구조 탐색"""

    print("BigQuery 연결 중...")
    client = BigQueryClient()
    print(f"✓ 프로젝트 ID: {client.project_id}\n")

    # 데이터셋 목록 조회
    print("=" * 60)
    print("📂 데이터셋 목록")
    print("=" * 60)
    datasets = client.list_datasets()

    if not datasets:
        print("❌ 데이터셋이 없습니다.")
        return

    for idx, dataset_id in enumerate(datasets, 1):
        print(f"{idx}. {dataset_id}")

    print()

    # 각 데이터셋의 테이블 정보 출력
    for dataset_id in datasets:
        print("=" * 60)
        print(f"📊 데이터셋: {dataset_id}")
        print("=" * 60)

        tables = client.list_tables(dataset_id)

        if not tables:
            print("  ❌ 테이블이 없습니다.\n")
            continue

        print(f"  테이블 수: {len(tables)}\n")

        for table_id in tables:
            print(f"  📋 테이블: {table_id}")
            print("  " + "-" * 56)

            # 스키마 정보
            schema = client.get_table_schema(dataset_id, table_id)
            print("  스키마:")
            for field in schema:
                print(f"    - {field['name']}: {field['type']} ({field['mode']})")

            # 샘플 데이터
            print("\n  샘플 데이터 (최대 3건):")
            try:
                samples = client.query_sample(dataset_id, table_id, limit=3)
                if samples:
                    for i, sample in enumerate(samples, 1):
                        print(f"\n  [{i}번째 행]")
                        for key, value in sample.items():
                            # 긴 텍스트는 잘라서 표시
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:100] + "..."
                            print(f"    {key}: {value}")
                else:
                    print("    (데이터 없음)")
            except Exception as e:
                print(f"    ❌ 오류: {str(e)}")

            print()

    print("=" * 60)
    print("✓ 탐색 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
