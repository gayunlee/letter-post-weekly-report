# Tasks: Phase 0.5 — Topic 분류 체계 재설계

## 완료

- [x] v3 Topic 4분류 정의 (운영 피드백/서비스 피드백/여론/기타)
- [x] 활용도 시뮬레이션 → "여론" 너무 넓음 발견
- [x] "커뮤니티 소통" catch-all 오염 분석 (841건, 61% 오분류)
- [x] **v3 5분류 확정** (운영 피드백/서비스 피드백/콘텐츠 반응/투자 담론/기타)
- [x] 경계 판별 규칙 작성 (대상 기반: 마스터/콘텐츠 vs 시장/투자)
- [x] 29종 카테고리 태그 5분류 기준 재배치
- [x] `src/classifier_v3/taxonomy.py` 5분류 반영
- [x] `src/classifier_v3/detail_tag_extractor.py` 신규 생성
- [x] `scripts/validate_v3_taxonomy.py` 생성 (LLM API 검증용)
- [x] `scripts/label_v3_taxonomy.py` 생성 (전체 데이터 LLM 라벨링)
- [x] `scripts/prepare_v3_training_data.py` 생성 (train/val/test split)
- [x] `scripts/train_v3_topic.py` 생성 (KcBERT 파인튜닝)
- [x] `scripts/benchmark_v3_golden.py` 생성 (golden set 벤치마크)
- [x] LLM 라벨링 모델별 비용·효과 비교 (Gemini/GPT-4o-mini/HF/Colab)
- [x] 실행 환경 판정 (이 머신 vs Colab 역할 분담)
- [x] Golden set v3 4분류 — Opus 4.6으로 52건 분류
- [x] Golden set v3 5분류 — 102건 (52건 재분류 + 50건 추가 샘플)
  - 결과: 콘텐츠 반응 34, 투자 담론 28, 기타 23, 운영 피드백 14, 서비스 피드백 3
  - 평균 신뢰도: ~0.87, border cases 10건 (9.8%)
  - 저장: `data/gold_dataset/v5_golden_set.json`
- [x] OpenSpec 업데이트 (5분류 반영, analysis-scenarios.md, decision 기록)
- [x] 스크립트 5분류 반영 (validate/label/prepare/train)

## 완료 (서버 — Opus 라벨링 + 훈련 데이터)

**계획 변경**: EXAONE/Colab → Opus 4.6 API/서버
- 원래: EXAONE-3.5-7.8B on Colab T4 ($0) → transformers 호환성 문제로 실패
- 변경: **Opus 4.6 API로 라벨링** (최상 품질) → KcBERT 파인튜닝 → 주간 분류 $0
- 근거: 파인튜닝 모델 품질 = 라벨링 품질. Golden set도 Opus로 만들었으므로 동일 기준 유지

- [x] Colab 노트북 생성 — Colab 환경 문제로 미사용
- [x] **Opus 4.6 라벨링 완료** (5,962건)
  - 투자 담론 2,753 (46.2%) / 콘텐츠 반응 2,006 (33.6%) / 기타 793 (13.3%) / 운영 피드백 310 (5.2%) / 서비스 피드백 100 (1.7%)
  - 평균 신뢰도: 0.865, 에러 22건 (0.4%)
  - 저장: `data/training_data/v3/labeled_all.json`
- [x] 훈련 데이터 준비 (`prepare_v3_training_data.py`)
  - 필터 후 4,980건 (신뢰도<0.7: 403건 제외, 짧은 텍스트: 579건 제외)
  - Train 3,740 / Val 496 / Test 744
  - 저장: `data/training_data/v3/topic/` (S3 동기화 완료)

## 로컬 이어작업 (파인튜닝 + 벤치마크)

서버 RAM 3.8GB로 파인튜닝 OOM → **로컬에서 이어 진행**.
훈련 데이터는 S3에 업로드 완료.

```bash
# 1. 코드 + 데이터 받기
git pull
make pull   # S3에서 data/training_data/v3/ 다운로드

# 2. 의존성 설치
pip install torch transformers scikit-learn accelerate

# 3. KcBERT 파인튜닝
python3 scripts/train_v3_topic.py --epochs 5 --batch-size 16
#   → 결과: models/v3/topic/final_model/

# 4. Golden set 벤치마크 (v2 69.3% 대비)
python3 scripts/benchmark_v3_golden.py
#   → 결과: models/v3/topic/golden_benchmark.json

# 5. 결과 서버에 올리기
make push   # models/v3/ 업로드

# 6. 서버에서 주간 분류 실행
#    (서버) make pull && python3 scripts/classify_v3.py --start YYYY-MM-DD --end YYYY-MM-DD
```

- [ ] KcBERT 파인튜닝 (로컬)
- [ ] 벤치마크: v2 69.3% 대비 v3 정확도 비교 (golden set 102건)
- [ ] 결과 모델 S3 업로드 (`make push`)

## 완료 (서버 — v3 추론 파이프라인 + 채널톡 데이터)

### v3 추론 파이프라인
- [x] `scripts/sync_data.py` SYNC_DIRS에 `models/v3`, `data/classified_data_v3`, `data/channel_io`, `data/training_data` 추가
- [x] `src/classifier_v3/v3_topic_classifier.py` 생성 — v3 Topic(5분류) + v2 Sentiment 분류기
- [x] `scripts/classify_v3.py` 생성 — BigQuery → v3 분류 → JSON 캐시
- [x] Makefile `extract-v3-model`, `classify-v3` 타겟 추가
- [x] `src/classifier_v3/__init__.py` V3TopicClassifier export 추가

### 채널톡 CS 데이터 탐색·추출
- [x] `src/bigquery/channel_queries.py` 생성 — 채널톡 BigQuery 쿼리 서비스
- [x] `src/bigquery/channel_preprocessor.py` 생성 — chatId 그룹핑 + 사용자 텍스트 연결
- [x] `scripts/explore_channel_io.py` 생성 — 데이터 탐색 (스키마, personType 분포, 키워드)
- [x] `scripts/export_channel_for_labeling.py` 생성 — 라벨링용 JSON 추출
- [x] 데이터 탐색 완료 — 431K건, 2024-08~2025-12, CS 문의 데이터 확인
- [x] 월별 추이 분석 — 2025-07 급증 (85K), 이후 53K~85K/월 유지
- [x] 분류 목적 정의 — 서비스·콘텐츠 개선, CS 개발 지원, 수강 데이터 조인, 시계열 통계
- [ ] **분류 체계 확정** — 1차 안: 결제·환불 / 구독·멤버십 / 콘텐츠·수강 / 서비스·기술 / 기타
  - 미결: 경계 규칙, 수강 데이터 조인 키, bot 응답 참고 여부, 마스터 매핑
  - 상세: `channel-talk-plan.md` 참조

## 완료 (4분류 전환)

**결정**: 콘텐츠 반응 + 투자 담론 → "콘텐츠·투자" 합산 (5분류 → 4분류)
- 근거: Test 83.1% → 93.4% (+10.3%p), 오분류 61% 해소
- 세부 구분은 detail_tags 28종 + sentiment로 슬라이싱
- 상세: `decisions/2026-03-02-v3-4cat-merge.md`

- [x] 4분류 학습 데이터 생성 (`data/training_data/v3/topic_4cat/`)
- [x] 벡터 KNN 분류기 구현 (`src/classifier_v3/vector_v3_classifier.py`)
- [x] 5분류 벤치마크 완료 (KcBERT 83.1% / 벡터 83.5% / 하이브리드 84.0%)
- [x] 4분류 KcBERT 파인튜닝 (`models/v3/topic_4cat/`) — Test 93.3%, Golden 77.5%
- [x] `v3_topic_classifier.py` 기본 경로 4분류 전환 + margin/top2 노출
- [x] `detail_tag_extractor.py` "콘텐츠·투자" 태그 합산 처리
- [x] `classify_v3.py` 4분류 파이프라인 반영

## 진행 중 (소수 카테고리 보정 파이프라인)

잔존 오분류의 실제 구조 (Golden 102건 기준, 사람 검수 후):
- **기타↔콘텐츠·투자 혼동**: 전체 오분류의 ~56% → LLM 보정 대상 아님
- **소수 카테고리 흡수**: ~20% → LLM 보정 대상
- 기타: ~24%

### 시도한 접근과 결과

1. ~~tag 키워드 매칭 보정 ($0)~~ — **폐기**
   - detail_tags/summary에서 운영/서비스 키워드 감지 시 topic 보정
   - 결과: 77.5% → **22.5%로 폭락** (API 에러 메시지 "오류"가 SERVICE_KEYWORDS에 매칭)
   - 교훈: 규칙 기반 후처리는 예상치 못한 입력에 취약

2. ~~LLM-only, margin 기반 선별~~ — **폐기**
   - softmax margin < 0.3인 건만 Haiku binary 검증
   - 결과: **후보 0건**. KcBERT softmax margin이 전부 0.93+ (극도로 overconfident)
   - 교훈: 파인튜닝 BERT의 calibration 문제는 구조적. Temperature Scaling 없이 raw softmax는 신뢰 불가

3. **LLM-only, top2 기반 선별** — 현행
   - 조건: `topic == "콘텐츠·투자" AND top2 ∈ {운영/서비스 피드백}`
   - 후보 56/102건 식별, confidence ≥ 0.6일 때만 보정
   - API 장애로 **벤치마크 미실행**

### Golden set 사람 검수

Opus 4.6이 라벨링한 golden set에서 "기타" 카테고리 오라벨링 발견.
같은 Opus로 재검수하면 동일 판단 반복 → 사람 검수 실시.

- 16건 검토, **7건 수정** (기타→콘텐츠·투자 5건, 기타→운영 피드백 2건)
- **Golden 정확도: 77.5% → 80.4% (+2.9%p)** — 모델 변경 없이, 라벨만 수정
- `v5_golden_set.json`에 `v4_topic` 필드 추가 (4분류 평가용)

### 현재 상태

- [x] `src/classifier_v3/topic_corrector.py` 생성 — LLM binary 보정 (top2 기반, 키워드 매칭 폐기)
- [x] `scripts/benchmark_correction.py` 생성 — golden set 보정 효과 벤치마크
- [x] `scripts/classify_v3.py` 보정 단계 삽입 (`--skip-correction` 플래그)
- [x] Golden set 사람 검수 완료 (7건 수정, 77.5% → 80.4%)
- [ ] **벤치마크 실행** — API 복구 후 `python3 scripts/benchmark_correction.py`
- [ ] 결과에 따라 보정 임계값 튜닝
- [ ] 결과 S3 업로드 (`make push`)

## 대기 (보정 파이프라인 검증 후)

- [ ] `python3 scripts/classify_v3.py` 주간 분류 실행
- [ ] 채널톡 golden set 생성 + Opus 라벨링
- [ ] 채널톡 KcBERT 파인튜닝

## 후속 (Phase 1 이후)

- [ ] v3 리포트 파이프라인 구축 (기존 generate_two_axis_report.py와 별도)
- [ ] 분석 시나리오 문서 v3 확정판 업데이트
