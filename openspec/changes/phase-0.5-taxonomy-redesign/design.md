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

### 5분류 → 4분류 전환

5분류 벤치마크에서 콘텐츠 반응↔투자 담론 혼동이 오분류의 61%를 차지.
합산 시 Test 83.1% → 93.4% (+10.3%p). 세부 분석은 detail_tags + sentiment로 대체.

| 4분류 Topic | 원본 | Train | Val | Test |
|------------|------|-------|-----|------|
| 콘텐츠·투자 | 콘텐츠 반응 + 투자 담론 | 3,249 | 432 | 648 |
| 운영 피드백 | 운영 피드백 | 200 | 26 | 39 |
| 기타 | 기타 | 229 | 30 | 45 |
| 서비스 피드백 | 서비스 피드백 | 62 | 8 | 12 |

상세: `decisions/2026-03-02-v3-4cat-merge.md`

### 소수 카테고리 보정 파이프라인

4분류 KcBERT(Test 93.3%, Golden 80.4%)의 잔존 오분류 구조:
- **기타↔콘텐츠·투자 혼동 ~56%**: LLM 보정으로 해결 불가 (분류 체계 자체의 경계 모호성)
- **소수 카테고리→다수 클래스 흡수 ~20%**: LLM 보정 대상
- **기타 ~24%**

```
KcBERT 4분류 → [보정] LLM 재검증 (top2가 소수 카테고리인 건만) → 캐시 저장
```

**폐기한 접근:**
- ~~tag 키워드 매칭~~: 에러 메시지 등 예상치 못한 텍스트에 취약하여 정확도 폭락 (77.5%→22.5%)
- ~~softmax margin 기반 선별~~: KcBERT가 극도로 overconfident (margin 전부 0.93+), 선별 불가

**현행 LLM 재검증 (~$0.1~0.2/주)**
- 조건: topic=콘텐츠·투자 AND topic_top2 ∈ {운영/서비스 피드백} (margin 조건 제거)
- Haiku에게 binary 질문: "콘텐츠·투자인가, [운영/서비스 피드백]인가?"
- confidence ≥ 0.6일 때만 보정
- detail_tags 추출과 독립적으로 동작 (classification의 top2만 사용)
- 벤치마크 미실행 상태 (API 장애)

**Golden set 사람 검수 (완료)**
- Opus 4.6 라벨링 오류 7/16건 발견 (기타 카테고리 집중)
- 수정 후 Golden 정확도: 77.5% → 80.4% (+2.9%p, 모델 변경 없이)
- 교훈: 같은 모델(Opus)로 재검수는 무의미 → 사람 검수 필수
- `v5_golden_set.json`에 `v4_topic` 필드 추가

## 코드 구조

```
src/classifier_v3/
├── __init__.py              # V3TopicClassifier export
├── taxonomy.py              # v3 분류 체계 정의 (4분류 Topic + 28개 카테고리 태그)
├── v3_topic_classifier.py   # v3 추론 분류기 (4분류 Topic + Sentiment, margin/top2 포함)
├── vector_v3_classifier.py  # 벡터 KNN 분류기 (k=7, 유사도 가중 투표)
├── detail_tag_extractor.py  # v3용 태그 추출기 (콘텐츠·투자 태그 합산 지원)
├── topic_corrector.py       # 소수 카테고리 보정 (LLM binary 재검증)
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
├── build_v3_vector_index.py     # 벡터 인덱스 구축 (labeled_all.json → ChromaDB)
├── benchmark_v3_vector.py       # KcBERT / 벡터 KNN / 하이브리드 3종 비교 벤치마크
├── benchmark_v3_golden.py       # Golden set 벤치마크 (v2 대비 비교)
├── benchmark_correction.py      # 보정 파이프라인 벤치마크 (golden set 기준)
├── classify_v3.py               # v3 주간 분류 (BigQuery → 분류 → 보정 → JSON)
├── explore_channel_io.py        # 채널톡 데이터 탐색
└── export_channel_for_labeling.py  # 채널톡 라벨링용 JSON 추출

models/v3/
├── topic/final_model/           # 5분류 KcBERT (비교 기준, 보존)
└── topic_4cat/final_model/      # 4분류 KcBERT (현행)

data/
├── vectorstore_v3/              # 벡터 KNN 인덱스 (ChromaDB, S3 동기화)
├── gold_dataset/
│   ├── v3_golden_set.json       # 초기 4분류 golden set (52건, 참고용)
│   └── v5_golden_set.json       # 5분류 golden set (102건, 4분류 평가에도 사용)
├── classified_data_v3/          # v3 분류 결과 캐시 (주간별 JSON)
├── channel_io/                  # 채널톡 데이터
│   └── channel_items_for_labeling.json
└── training_data/v3/
    ├── labeled_all.json         # Opus 4.6 라벨링 결과 (5,962건)
    ├── topic/                   # 5분류 split (비교 기준)
    │   ├── train.json / val.json / test.json
    │   └── category_mapping.json
    └── topic_4cat/              # 4분류 split (현행)
        ├── train.json / val.json / test.json
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
