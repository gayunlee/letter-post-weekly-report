# 데이터 인벤토리

전체 데이터 자산의 현황과 상태를 추적한다.

## 도메인별 현황

| 도메인 | 소스 | 규모 | 기간 | 라벨링 상태 | manifest |
|--------|------|------|------|------------|----------|
| 편지/게시글 | `classified_data_two_axis/` | 13,815건 (10주) | 2025-12 ~ 2026-02 | 완료 (13,995건) | `data/pipeline/letter_post.json` |
| 채널톡 | `channel_io.messages` | 431,589 메시지 | 2024-08 ~ 2025-12 | 미시작 | `data/pipeline/channel_talk.json` (예정) |

## 편지/게시글 상세

### 원시 데이터
- **경로**: `data/classified_data_two_axis/`
- **소스**: BigQuery `us-service-data` → 주간 편지글/게시글
- **기간**: 10주 (2025-12 ~ 2026-02)
- **건수**: 13,815건

### 라벨링 결과
- **파일**: `data/training_data/v3/labeled_all_v3c_full.json`
- **건수**: 13,995건 (5,962 신규 LLM + 8,033 구 데이터 변환)
- **분류 체계**: v3c 5분류 (운영 피드백, 서비스 피드백, 콘텐츠·투자, 일상·감사, 기타)
- **라벨러**: Claude Opus 4.6

### Golden Set
- **파일**: `data/gold_dataset/v6_golden_set.json`
- **건수**: 227건
- **검수 완료**: 진행 중 (102건 검수 → 227건 전수 검수 목표)
- **검수 도구**: `scripts/_golden_v3c_manual.py`

### 학습 데이터 (Clean Split)
- **디렉토리**: `data/training_data/v3/topic_v3c_full_clean/`
- **상태**: 재생성 필요 (golden set 전수 검수 후)

### 모델
- **기존 최고**: v3c_full — Test 89.6%, Golden 85.3% (102건 기준)
- **경로**: `models/v3/topic_v3c_full/`

## 채널톡 상세

### 원시 데이터
- **소스**: BigQuery `channel_io.messages`
- **기간**: 2024-08 ~ 2025-12
- **건수**: 431,589 메시지
- **전처리**: `src/bigquery/channel_preprocessor.py` — chatId 그룹핑 + 사용자 텍스트 연결
- **추출 스크립트**: `scripts/export_channel_for_labeling.py`

### 주요 특성 (기존 탐색 결과)
- personType: bot 68%, user 20%, manager 12%
- 대화당 평균 22건 메시지, 사용자 메시지 4.5건
- 주요 키워드: 환불, 취소, 구독, 해지, 멤버십, 결제, 패키지

### 라벨링/Golden/모델
- 상태: 미시작 (Phase 2에서 분류 체계 설계 후 Phase 3에서 진행)
