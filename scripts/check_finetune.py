#!/usr/bin/env python3
"""파인튜닝 작업 상태 확인

사용법:
    python scripts/check_finetune.py [job_id]
    python scripts/check_finetune.py --list  # 모든 작업 목록
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        # 모든 파인튜닝 작업 목록
        print("=" * 60)
        print("파인튜닝 작업 목록")
        print("=" * 60)

        jobs = client.fine_tuning.jobs.list(limit=10)
        for job in jobs.data:
            print(f"\n작업 ID: {job.id}")
            print(f"  모델: {job.model}")
            print(f"  상태: {job.status}")
            if job.fine_tuned_model:
                print(f"  파인튜닝 모델: {job.fine_tuned_model}")
            if job.error:
                print(f"  오류: {job.error}")
    else:
        # 특정 작업 상태 확인
        job_id = sys.argv[1]
        print(f"작업 ID: {job_id}")

        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"\n상태: {job.status}")
        print(f"모델: {job.model}")

        if job.fine_tuned_model:
            print(f"\n✓ 파인튜닝 완료!")
            print(f"파인튜닝 모델: {job.fine_tuned_model}")

        if job.error:
            print(f"\n✗ 오류 발생:")
            print(f"  {job.error}")

        # 이벤트 로그
        print("\n최근 이벤트:")
        events = client.fine_tuning.jobs.list_events(job_id, limit=10)
        for event in events.data:
            print(f"  [{event.created_at}] {event.message}")


if __name__ == "__main__":
    main()
