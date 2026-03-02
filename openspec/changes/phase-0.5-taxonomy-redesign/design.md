# Design: Phase 0.5 — Topic 분류 체계 재설계

## 접근 방법

### 2단계 파이프라인

```
[1단계] 서버 — Opus 4.6 API → 5,962건 라벨링 + 102건 golden set  ← 완료
    ↓  (최상 품질 훈련 데이터 + 벤치마크 기준)
[2단계] 로컬 — KcBERT 파인튜닝 (GPU) + 벤치마크
    ↓  (golden set 기준, v2 69.3% 대비 비교)
[3단계] 서버 — make pull → classify_v3.py (주간 분류, $0)
```

### 왜 이 구조인가

**라벨링에 Opus 4.6을 쓰는 이유:**
- 파인튜닝 모델 품질 = 훈련 데이터 품질. 최상의 모델로 라벨링해야 의미 있음
- Golden set도 Opus 4.6으로 만들었으므로 동일 기준 유지
- Claude Code Max 플랜 사용 중 → 비용 부담 적음
- EXAONE/Colab 시도 → transformers 호환성 문제로 실패. API가 안정적

**파인튜닝 대상이 KcBERT인 이유:**
- klue/roberta-base (110M params) → 추론 속도 빠름, CPU에서도 실행 가능
- 기존 v2 파인튜닝 인프라 재활용
- 주간 파이프라인에서 반복 실행 → LLM 비용 $0

**로컬에서 파인튜닝하는 이유:**
- 서버 RAM 3.8GB → KcBERT 파인튜닝 시 OOM (batch_size=4로도 Kill)
- 로컬 GPU 활용 → 수 분 내 완료

### 환경별 역할

| 환경 | 스펙 | 역할 |
|------|------|------|
| 서버 | AMD EPYC, RAM 3.8GB, GPU 없음 | API 라벨링, 데이터 준비, 주간 분류 추론 |
| 로컬 | GPU 있는 개발 머신 | KcBERT 파인튜닝, 벤치마크 |
| S3 (Vultr) | 오브젝트 스토리지 | 데이터·모델 동기화 (`make push/pull`) |

### Opus 라벨링 결과 (완료)

| Topic | 건수 | 비율 |
|-------|------|------|
| 투자 담론 | 2,753 | 46.2% |
| 콘텐츠 반응 | 2,006 | 33.6% |
| 기타 | 793 | 13.3% |
| 운영 피드백 | 310 | 5.2% |
| 서비스 피드백 | 100 | 1.7% |

- 평균 신뢰도: 0.865, 에러: 22건 (0.4%)
- 필터 후 훈련 대상: 4,980건 (신뢰도<0.7 403건 + 짧은텍스트 579건 제외)

## 코드 구조

```
src/classifier_v3/
├── __init__.py              # V3TopicClassifier export
├── taxonomy.py              # v3 5분류 체계 정의 (5개 Topic + 29개 카테고리 태그)
├── v3_topic_classifier.py   # v3 추론 분류기 (Topic 5분류 + Sentiment, CPU 추론)
├── detail_tag_extractor.py  # v3용 태그 추출기
├── four_axis_classifier.py  # 기존 4축 래퍼 (미사용, 참고용)
├── urgency_rules.py         # 기존 긴급도 규칙 (미사용, 참고용)
└── department_router.py     # 기존 부서 라우팅 (미사용, 참고용)

src/bigquery/
├── channel_queries.py       # 채널톡 BigQuery 쿼리 서비스
└── channel_preprocessor.py  # chatId 그룹핑 + 사용자 텍스트 연결

scripts/
├── validate_v3_taxonomy.py      # LLM 일관성 검증 (API 의존)
├── label_v3_taxonomy.py         # 전체 데이터 LLM 라벨링 (API 의존)
├── prepare_v3_training_data.py  # 훈련 데이터 split
├── train_v3_topic.py            # KcBERT 파인튜닝
├── benchmark_v3_golden.py       # Golden set 벤치마크 (v2 대비 비교)
├── classify_v3.py               # v3 주간 분류 (BigQuery → 분류 → JSON)
├── explore_channel_io.py        # 채널톡 데이터 탐색
└── export_channel_for_labeling.py  # 채널톡 라벨링용 JSON 추출

models/v3/topic/final_model/     # 로컬 파인튜닝 결과 (make push/pull로 동기화)

data/
├── gold_dataset/
│   ├── v3_golden_set.json       # 초기 4분류 golden set (52건, 참고용)
│   └── v5_golden_set.json       # 5분류 golden set (102건, 현행)
├── classified_data_v3/          # v3 분류 결과 캐시 (주간별 JSON)
├── channel_io/                  # 채널톡 데이터
│   └── channel_items_for_labeling.json
└── training_data/v3/
    ├── labeled_all.json         # Opus 4.6 라벨링 결과 (5,962건)
    └── topic/
        ├── train.json           # 3,740건
        ├── val.json             # 496건
        ├── test.json            # 744건
        └── category_mapping.json
```

### 로컬 파인튜닝 + 서버 배포 흐름

```
로컬:
  git pull                          # 코드 + OpenSpec
  make pull                         # S3에서 훈련 데이터 다운로드
  python3 scripts/train_v3_topic.py # KcBERT 파인튜닝 (GPU)
  python3 scripts/benchmark_v3_golden.py  # 벤치마크
  make push                         # 모델 S3 업로드

서버:
  make pull                         # 모델 다운로드
  python3 scripts/classify_v3.py    # 주간 분류 실행 ($0)
```

### 채널톡 데이터 흐름

```
python3 scripts/explore_channel_io.py --start 2025-11-01 --end 2025-11-08
python3 scripts/export_channel_for_labeling.py --start ... --end ...
→ 분류 체계 확정 후 Opus 라벨링 → 파인튜닝 (v3와 동일 파이프라인)
```

## 활용도 검증 (완료)

v3 분류 결과가 기존 보고서/분석 시나리오를 충분히 지원하는지 확인.

**검증 과정:**
1. 초기 4분류(여론 통합)로 golden set 52건 생성
2. 4가지 구체적 분석 질문으로 시뮬레이션 → **"여론" 너무 넓음 발견** (4개 중 3개 실패)
3. 주간 데이터 2,542건 분석 → **"커뮤니티 소통" catch-all 오염 발견**
4. 5분류 체계로 재설계 → 102건 golden set 재생성
5. 동일 분석 질문으로 재검증 → **모든 질문에 Topic 필터만으로 대응 가능**

**핵심 검증 결과:**
- "마스터별 콘텐츠 감정 비중?" → Topic="콘텐츠 반응"으로 필터 가능 ✓
- "증시 여론 및 감정?" → Topic="투자 담론"으로 필터 가능 ✓
- "서비스 개선 요구사항?" → Topic="서비스 피드백"으로 필터 가능 ✓
- "콘텐츠 불편사항?" → Topic="콘텐츠 반응" + Sentiment="부정"으로 필터 가능 ✓
