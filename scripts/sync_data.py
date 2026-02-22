"""Vultr Object Storage 데이터 동기화

로컬 data/ 폴더를 Vultr Object Storage(S3 호환)와 동기화합니다.

사용법:
    python3 scripts/sync_data.py push          # 로컬 → 원격 (업로드)
    python3 scripts/sync_data.py pull          # 원격 → 로컬 (다운로드)
    python3 scripts/sync_data.py push --dry    # 변경 사항만 확인
    python3 scripts/sync_data.py status        # 원격 파일 목록
"""
import os
import sys
import argparse
import hashlib
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

# 동기화 대상 디렉토리
SYNC_DIRS = [
    "data/classified_data_two_axis",
    "data/sub_themes",
    "data/stats",
    "data/review",
    "reports",
]

# 동기화 제외 패턴
EXCLUDE_EXTENSIONS = {".tmp", ".swp", ".DS_Store"}


def get_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("VULTR_S3_ENDPOINT"),
        aws_access_key_id=os.getenv("VULTR_S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("VULTR_S3_SECRET_KEY"),
        region_name="sgp1",
    )


def get_bucket():
    return os.getenv("VULTR_S3_BUCKET", "voc-data")


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def list_remote(client, bucket: str, prefix: str = "") -> dict:
    """원격 파일 목록 + ETag"""
    remote = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            etag = obj["ETag"].strip('"')
            remote[obj["Key"]] = {
                "size": obj["Size"],
                "etag": etag,
                "modified": obj["LastModified"],
            }
    return remote


def list_local(base_dir: str) -> dict:
    """로컬 파일 목록 + MD5"""
    local = {}
    for sync_dir in SYNC_DIRS:
        dir_path = Path(base_dir) / sync_dir
        if not dir_path.exists():
            continue
        for file_path in dir_path.rglob("*"):
            if file_path.is_dir():
                continue
            if file_path.suffix in EXCLUDE_EXTENSIONS:
                continue
            rel = str(file_path.relative_to(base_dir))
            local[rel] = {
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "md5": md5_file(str(file_path)),
            }
    return local


def cmd_push(args):
    """로컬 → 원격 업로드"""
    client = get_client()
    bucket = get_bucket()
    base_dir = args.base_dir

    print("로컬 파일 스캔 중...")
    local = list_local(base_dir)
    print(f"  로컬: {len(local)}개 파일")

    print("원격 파일 확인 중...")
    remote = list_remote(client, bucket)
    print(f"  원격: {len(remote)}개 파일")

    # 변경/신규 파일 찾기
    to_upload = []
    for key, info in local.items():
        if key not in remote:
            to_upload.append((key, info, "새 파일"))
        elif remote[key]["etag"] != info["md5"]:
            to_upload.append((key, info, "변경됨"))

    if not to_upload:
        print("\n  동기화 완료 — 변경 없음")
        return

    print(f"\n  업로드 대상: {len(to_upload)}개")
    for key, info, reason in to_upload:
        size_kb = info["size"] / 1024
        print(f"    [{reason}] {key} ({size_kb:.1f}KB)")

    if args.dry:
        print("\n  (dry run — 실제 업로드 안 함)")
        return

    uploaded = 0
    for key, info, reason in to_upload:
        try:
            client.upload_file(info["path"], bucket, key)
            uploaded += 1
            print(f"  ↑ {key}")
        except Exception as e:
            print(f"  ✗ {key}: {e}")

    print(f"\n  {uploaded}/{len(to_upload)}개 업로드 완료")


def cmd_pull(args):
    """원격 → 로컬 다운로드"""
    client = get_client()
    bucket = get_bucket()
    base_dir = args.base_dir

    print("원격 파일 확인 중...")
    remote = list_remote(client, bucket)
    print(f"  원격: {len(remote)}개 파일")

    print("로컬 파일 스캔 중...")
    local = list_local(base_dir)
    print(f"  로컬: {len(local)}개 파일")

    # 원격에만 있거나 변경된 파일
    to_download = []
    for key, info in remote.items():
        if key not in local:
            to_download.append((key, info, "새 파일"))
        elif local[key]["md5"] != info["etag"]:
            to_download.append((key, info, "변경됨"))

    if not to_download:
        print("\n  동기화 완료 — 변경 없음")
        return

    print(f"\n  다운로드 대상: {len(to_download)}개")
    for key, info, reason in to_download:
        size_kb = info["size"] / 1024
        print(f"    [{reason}] {key} ({size_kb:.1f}KB)")

    if args.dry:
        print("\n  (dry run — 실제 다운로드 안 함)")
        return

    downloaded = 0
    for key, info, reason in to_download:
        local_path = Path(base_dir) / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(bucket, key, str(local_path))
            downloaded += 1
            print(f"  ↓ {key}")
        except Exception as e:
            print(f"  ✗ {key}: {e}")

    print(f"\n  {downloaded}/{len(to_download)}개 다운로드 완료")


def cmd_status(args):
    """원격 파일 목록"""
    client = get_client()
    bucket = get_bucket()

    remote = list_remote(client, bucket)
    if not remote:
        print("원격 버킷이 비어있습니다.")
        return

    total_size = 0
    for key in sorted(remote.keys()):
        info = remote[key]
        size_kb = info["size"] / 1024
        total_size += info["size"]
        print(f"  {key:60s} {size_kb:>8.1f}KB")

    print(f"\n  총 {len(remote)}개 파일, {total_size/1024/1024:.1f}MB")


def main():
    parser = argparse.ArgumentParser(description="Vultr Object Storage 동기화")
    parser.add_argument("command", choices=["push", "pull", "status"])
    parser.add_argument("--dry", action="store_true", help="변경 확인만 (실제 전송 안 함)")
    parser.add_argument("--base-dir", default=".", help="프로젝트 루트 디렉토리")
    args = parser.parse_args()

    if args.command == "push":
        cmd_push(args)
    elif args.command == "pull":
        cmd_pull(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
