"""BigQuery에서 마스터 정보 탐색"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigquery.client import BigQueryClient


def main():
    client = BigQueryClient()

    # us_plus 데이터셋의 테이블 목록 확인
    print("=" * 60)
    print("us_plus 데이터셋 테이블 목록:")
    print("=" * 60)

    query = f"""
    SELECT table_name
    FROM `{client.project_id}.us_plus.INFORMATION_SCHEMA.TABLES`
    ORDER BY table_name
    """

    tables = client.execute_query(query)
    for table in tables:
        print(f"  - {table['table_name']}")

    print("\n" + "=" * 60)
    print("masters 테이블 스키마 확인:")
    print("=" * 60)

    # masters 테이블이 있다면 스키마 확인
    query = f"""
    SELECT column_name, data_type
    FROM `{client.project_id}.us_plus.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'masters'
    ORDER BY ordinal_position
    """

    try:
        columns = client.execute_query(query)
        for col in columns:
            print(f"  - {col['column_name']}: {col['data_type']}")

        # 샘플 데이터 확인
        print("\n" + "=" * 60)
        print("masters 테이블 샘플 데이터:")
        print("=" * 60)

        query = f"""
        SELECT *
        FROM `{client.project_id}.us_plus.masters`
        LIMIT 5
        """

        masters = client.execute_query(query)
        for master in masters:
            print(f"\n마스터:")
            for key, value in master.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
