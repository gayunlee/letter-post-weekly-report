# Data Sync Specification

## Purpose

분류 데이터와 리포트를 Vultr Object Storage(S3 호환)로 동기화하여, 여러 기기에서 접근 가능하게 한다.

## Requirements

### Requirement: 데이터 업로드 (Push)

로컬 데이터를 원격 Object Storage에 업로드한다.

#### Scenario: 변경 파일만 업로드
- GIVEN 로컬에 분류 데이터/리포트가 존재
- WHEN `make push` 실행
- THEN MD5 해시 비교로 변경/신규 파일만 업로드됨
- AND 동기화 대상: `data/classified_data_two_axis/`, `data/sub_themes/`, `data/stats/`, `data/review/`, `reports/`

#### Scenario: Dry run
- GIVEN 업로드 전 확인이 필요
- WHEN `python scripts/sync_data.py push --dry` 실행
- THEN 업로드 대상 파일 목록만 출력되고 실제 전송 안 됨

---

### Requirement: 데이터 다운로드 (Pull)

원격 Object Storage에서 로컬로 데이터를 다운로드한다.

#### Scenario: 새 기기 세팅
- GIVEN 레포를 클론하고 환경변수가 설정됨
- WHEN `make setup` 실행
- THEN boto3/python-dotenv 설치 + 전체 데이터 다운로드됨

#### Scenario: 데이터 최신화
- GIVEN 이미 데이터가 존재하는 환경
- WHEN `make pull` 실행
- THEN `git pull` + 변경된 파일만 다운로드됨

---

### Requirement: 환경변수

Object Storage 접근에 필요한 환경변수.

| 변수 | 필수 | 설명 |
|------|------|------|
| `VULTR_S3_ENDPOINT` | O | Object Storage 엔드포인트 |
| `VULTR_S3_ACCESS_KEY` | O | Access Key |
| `VULTR_S3_SECRET_KEY` | O | Secret Key |
| `VULTR_S3_BUCKET` | - | 버킷명 (기본: voc-data) |

`.env` 파일 또는 시스템 환경변수(`~/.bashrc`) 모두 지원.
