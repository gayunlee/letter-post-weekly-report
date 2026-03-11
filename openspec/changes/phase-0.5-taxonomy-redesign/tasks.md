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

- [x] KcBERT 파인튜닝 (로컬) — 5분류, 4분류, 클린 4분류 3회 완료
- [x] 벤치마크: v2 69.3% 대비 v3 정확도 비교 (golden set 102건) — 클린 83.3%
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

## 완료 (소수 카테고리 보정 + 데이터 누수 수정)

### 시도한 접근과 결과

1. ~~tag 키워드 매칭 보정 ($0)~~ — **폐기**
   - detail_tags/summary에서 운영/서비스 키워드 감지 시 topic 보정
   - 결과: 77.5% → **22.5%로 폭락** (API 에러 메시지 "오류"가 SERVICE_KEYWORDS에 매칭)
   - 교훈: 규칙 기반 후처리는 예상치 못한 입력에 취약

2. ~~LLM-only, margin 기반 선별~~ — **폐기**
   - softmax margin < 0.3인 건만 Haiku binary 검증
   - 결과: **후보 0건**. KcBERT softmax margin이 전부 0.93+ (극도로 overconfident)
   - 교훈: 파인튜닝 BERT의 calibration 문제는 구조적

3. **LLM-only, top2 기반 선별** — 현행
   - 조건: `topic == "콘텐츠·투자" AND top2 ∈ {운영/서비스 피드백}`
   - 시뮬레이션: 83.3% → ~86.3% (+3.0%p, 3건 개선)
   - API 크레딧 소진으로 실측 미완

### Golden set 사람 검수 (2라운드)

- 1차: 16건 검토, 7건 수정 → 정확도 77.5% → 80.4%
- 2차: 추가 6건 수정 (운영↔콘텐츠·투자 경계) → 총 13건 수정
- 교훈: Opus 4.6의 비결정성 — 같은 텍스트 102건을 두 세션에서 라벨링 시 24건(23.5%) 불일치

### 데이터 누수 발견 + 클린 재학습

- **발견**: Golden set 102건이 labeled_all.json과 100% 중복. 66건이 Train에 포함.
- **대응**: Golden 제외 클린 split 생성 (`topic_4cat_clean/`) → KcBERT 재학습
- **결과**: Golden 정확도 86.3%(오염) → **83.3%(클린, 최초 유효 벤치마크)**

### 완료 사항

- [x] `src/classifier_v3/topic_corrector.py` 생성 — LLM binary 보정
- [x] `scripts/benchmark_correction.py` 생성 — golden set 보정 벤치마크
- [x] `scripts/classify_v3.py` 보정 단계 삽입 (`--skip-correction` 플래그)
- [x] Golden set 사람 검수 2라운드 완료 (총 13건 수정)
- [x] 데이터 누수 발견 + 클린 split 생성 (`data/training_data/v3/topic_4cat_clean/`)
- [x] 클린 KcBERT 파인튜닝 (`models/v3/topic_4cat_clean/`) — Test 91.4%, Golden 83.3%
- [x] 벤치마크 완료: 클린 Golden 83.3%, 시뮬 보정 후 ~86.3%

### 미완료 (KcBERT 보정 — 보류)

> KcBERT 60.4% 한계로 LLM 디스틸레이션 전략으로 전환. 아래 항목은 보류.
> 상세: `decisions/2026-03-09-model-distillation-strategy.md`

- [ ] ~~LLM 보정 실측~~ — KcBERT 자체가 한계, 보정으로 해결 불가
- [ ] ~~결과 S3 업로드~~ — 보류
- [ ] ~~`v3_topic_classifier.py` 기본 모델 경로 전환~~ — 새 모델 채택 후 결정

## 완료 (학습/평가 데이터 완전 분리)

**발견**: v6 golden set 227건 중 74건이 학습 데이터에 포함 (v5 102건 leak check만 존재, v6 125건 미검사)

- [x] `prepare_v3_training_data.py` — golden set 제외 로직 추가 (`--golden-dir`, `gold_dataset/*_golden_set.json` glob)
- [x] `prepare_v3_training_data.py` — `--topic-field` 인자 추가 (v3c_topic 등 유연 대응)
- [x] `train_v3_topic.py` — leak check를 `gold_dataset/` 전체로 확대 (v5 하드코딩 제거)
- [x] `design.md` — 학습/평가 데이터 분리 원칙 섹션 추가
- [x] 클린 데이터 준비 (`topic_v3c_full_clean/` — golden 227건 제외, 12,073건)
- [x] 클린 모델 재학습 — KcBERT golden 60.4% (한계 확인)

## 완료 (소수 카테고리 데이터 보강)

- [x] BigQuery에서 과거 데이터 43,000건 추출 (2024-11 ~ 2025-11)
- [x] KcBERT로 0원 1차 필터링 → 소수 카테고리 후보 ~2,000건 추출
- [x] Opus 재라벨링 → 서비스/운영/기타 데이터 보강
- [x] labeled_all_v3c_full.json (13,995건) 생성
- [x] 재학습 → golden 59.0% → 60.4% (+1.4%p만 개선 = KcBERT 한계 확인)

## 완료 (모델 벤치마크 비교)

상세: `decisions/2026-03-09-model-distillation-strategy.md`

- [x] Haiku 4.5 에이전트 벤치마크: 86.3% (참고치, 배치 방식)
- [x] gpt-5o-codex-mini 에이전트 벤치마크: 46.3% (코드 특화 모델, 부적합)
- [x] `scripts/benchmark_haiku_golden.py` 생성 — Haiku 개별 API 호출
- [x] `scripts/benchmark_gemini_golden.py` 생성 — Gemini Flash 개별 API 호출
- [x] `scripts/benchmark_gpt4o_mini_golden.py` 생성 — GPT-4o-mini 개별 API 호출
- [x] Gemini 2.5 Flash API 벤치마크: **67.4%** (153/227)
- [x] GPT-4o-mini API 벤치마크: **75.3%** (171/227)
- [x] Haiku 4.5 API 벤치마크: **78.9%** (179/227)
- [x] KcBERT 한계 결론 — 모델 용량(110M) 문제, 데이터 문제 아님
- [x] 디스틸레이션 전략 의사결정 문서 작성

## 진행 중 — LLM 디스틸레이션 (Phase 0.5b)

**목표**: Opus 라벨로 저렴한 모델 파인튜닝 → Golden 90%+
**핵심 발견**: 불균형 데이터(콘텐츠·투자 87.9%) 파인튜닝은 zero-shot보다 나빠질 수 있음
**환경**: Vertex AI (1차) + OpenAI (2차 백업)
**상세**: `decisions/2026-03-09-model-distillation-strategy.md`

### 데이터 준비 ✅
- [x] Golden 227건 제외 clean split — `topic_v3c_full_clean/` (train 10,641 / val 1,416)
- [x] Gemini FT용 JSONL — `vertex_ai/gemini_ft_{train,val}.jsonl`
- [x] EXAONE/Solar SFT용 JSONL (불균형) — `vertex_ai/sft_{train,val}.jsonl`
- [x] `scripts/prepare_vertex_training_data.py` — 포맷 변환 스크립트

### Vertex AI 환경 세팅 ✅
- [x] GCP 프로젝트: `us-service-data`, 리전: `us-central1`
- [x] ADC 인증: `~/.config/gcloud/legacy_credentials/gayunlee11@us-all.co.kr/adc.json`
- [x] GCS 버킷: `gs://us-service-data-vertex-ft` (Gemini), `gs://us-service-data-vertex-ft-us` (EXAONE/Solar)
- [x] 스크립트: `check_vertex_jobs.py`, `monitor_vertex_jobs.py`, `benchmark_vertex_models.py`

### Step 1: 불균형 데이터 파인튜닝 (완료/진행 중)

1차 실험 — 불균형 데이터 그대로 3모델 동시 제출

- [x] Gemini 2.5 Flash FT 제출 → **SUCCEEDED**
- [x] Gemini FT 벤치마크: **63.4%** (144/227) — zero-shot Haiku 78.9%보다 낮음
  - 원인: 콘텐츠·투자 recall 0.95가 다른 카테고리 흡수, 기타 recall 0.10
  - **불균형 파인튜닝이 zero-shot보다 나빠지는 것 확인**
- [ ] EXAONE 3.5 7.8B LoRA — PENDING (GPU 대기)
- [ ] Solar 10.7B LoRA — PENDING (GPU 대기)

### Step 2: 균형 데이터 생성 ✅

**배경**: KcBERT(110M)에서 균형 샘플링이 효과 없었지만(66.5%), 그건 모델 용량 한계.
7.8B+ 모델은 맥락 추론 가능 → 균형 데이터 효과 기대.

- [x] `scripts/prepare_balanced_vertex_data.py` 생성
- [x] 균형 데이터 2종 생성:

| 데이터셋 | train | val | 최대 비율 | 설명 |
|---------|-------|-----|----------|------|
| 기존 불균형 | 10,641 | 1,416 | 77.5% | 콘텐츠·투자 쏠림 |
| **balanced_strict** | 1,503 | 167 | 20.4% | 각 ~334건 (1:1 균형) |
| **balanced_moderate** | 5,295 | 588 | 34.0% | 소수 전량 + 콘텐츠·투자 2,000 |

### Step 3: EXAONE 균형 파인튜닝 (진행 예정)

**전략**: EXAONE 7.8B로 먼저 균형 효과 검증 → 효과 있으면 나머지 모델로 확대

- [ ] EXAONE balanced_strict 파인튜닝 (Vertex AI)
- [ ] EXAONE balanced_moderate 파인튜닝 (Vertex AI)
- [ ] 3종 비교 벤치마크 (Golden 227건):
  - EXAONE 불균형 (Step 1에서 진행 중)
  - EXAONE balanced_strict
  - EXAONE balanced_moderate
- [ ] 균형 효과 판정:
  - 효과 있음 (best ≥ 80%) → Step 4로
  - 효과 없음 → 분류 체계 재설계 or 접근법 변경

### Step 4: 최적 데이터로 모델 비교 (Step 3 결과 후)

Step 3에서 가장 좋은 데이터셋으로 나머지 모델도 파인튜닝:

- [ ] Gemini 2.5 Flash FT (균형 데이터)
- [ ] Solar 10.7B LoRA (균형 데이터)
- [ ] 3모델 비교 벤치마크 + version_log 기록

### Step 5: 의사결정

- [ ] 90%+ 달성 모델 채택 → 주간 파이프라인 전환
- [ ] 전부 90% 미달 → 스케일업 (EXAONE 32B / GPT-4.1-mini FT) 또는 분류 체계 재설계

## 대기 (디스틸레이션 완료 후)

- [ ] 채택 모델로 `classify_v3.py` 주간 분류 실행
- [ ] 채널톡 분류 체계 확정 + golden set 생성
- [ ] 채널톡 파이프라인 (Phase 2)

## 후속 (Phase 1 이후)

- [ ] v3 리포트 파이프라인 구축 (기존 generate_two_axis_report.py와 별도)
- [ ] 분석 시나리오 문서 v3 확정판 업데이트
