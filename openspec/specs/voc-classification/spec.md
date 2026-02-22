# VOC Classification Specification

## Purpose

금융 콘텐츠 플랫폼(오피셜클럽)의 편지글/게시글/채널톡 VOC 데이터를 다층 분류하여, 주간 인사이트 리포트 자동 생성 및 다른 팀의 자유 질의 응답을 가능하게 하는 시스템.

## Requirements

### Requirement: 2축 분류 (Topic × Sentiment)

파인튜닝된 KcBERT 모델로 편지글/게시글을 Topic 4개 × Sentiment 3개로 분류한다.

- **Topic**: 콘텐츠 반응, 투자 이야기, 서비스 이슈, 커뮤니티 소통
- **Sentiment**: 긍정, 부정, 중립
- **모델**: `models/two_axis/topic/final_model`, `models/two_axis/sentiment/final_model`

#### Scenario: 주간 데이터 일괄 분류
- GIVEN 1주치 편지글/게시글이 BigQuery에서 조회됨
- WHEN 2축 분류 파이프라인 실행
- THEN 각 항목에 topic, sentiment, topic_confidence, sentiment_confidence 필드가 부여됨
- AND 결과가 `data/classified_data_two_axis/{date}.json`에 캐싱됨
- AND 이미 분류된 데이터는 건너뜀

#### Scenario: 전주 대비 통계
- GIVEN 대상 주간과 전주 분류 데이터가 존재
- WHEN 통계 분석 실행
- THEN Topic × Sentiment 교차 집계, 마스터별 분포, 전주 대비 증감이 산출됨

---

### Requirement: Detail Tags 추출 (카테고리 태그 + 자유 태그)

2축 분류 후, LLM(Claude Haiku)으로 28개 controlled vocabulary 카테고리 태그 + 자유 태그 + 요약을 추출한다.

- **모델**: claude-haiku-4-5-20251001 ($3.12/주, 2,542건 기준)
- **카테고리 태그**: Topic별 6~8개, 총 28개
- **품질 기준**: 목록 외 태그 비율 < 5%, 태그 커버리지 > 90%

#### Scenario: 카테고리 태그 추출
- GIVEN 2축 분류가 완료된 항목
- WHEN detail_tags 추출 실행
- THEN 각 항목에 `detail_tags: {category_tags: [...], free_tags: [...], summary: "..."}`가 부여됨
- AND category_tags는 반드시 28개 허용 목록에서만 선택됨
- AND 이미 detail_tags가 있는 항목은 건너뜀

#### Scenario: 태그 집계
- GIVEN detail_tags가 부착된 전체 데이터
- WHEN 카테고리 태그 집계 실행
- THEN Topic별 태그 분포가 산출됨
- AND "결제 관련 불만이 몇 건?" 같은 구체적 질의에 응답 가능

---

### Requirement: 서브 테마 분석

서비스 이슈 항목을 클러스터링하고, Topic별 부정 항목의 패턴을 LLM으로 요약한다.

#### Scenario: 서비스 이슈 클러스터링
- GIVEN 서비스 이슈로 분류된 항목들
- WHEN 임베딩 → K-Means 클러스터링 실행 (silhouette score로 최적 k 선택)
- THEN 각 클러스터에 LLM 생성 라벨 + 건수가 부여됨
- AND 결과가 `data/sub_themes/{date}.json`에 저장됨

#### Scenario: 부정 테마 요약
- GIVEN 콘텐츠 반응/투자 이야기/커뮤니티 소통의 부정 항목들
- WHEN LLM 패턴 요약 실행
- THEN Topic별 부정 의견의 주요 패턴과 대표 사례가 요약됨

---

### Requirement: 주간 리포트 생성

분류 + 통계 + 서브 테마를 종합하여 마크다운 리포트와 엑셀 파일을 자동 생성한다.

#### Scenario: 마크다운 리포트
- GIVEN 통계, 서브 테마, 마스터 목록이 준비됨
- WHEN 리포트 생성 실행
- THEN `reports/two_axis_report_{date}.md` 파일이 생성됨
- AND 핵심 요약, 카테고리 태그 분포, 서비스 이슈 클러스터, 마스터별 상세가 포함됨

#### Scenario: 엑셀 데이터
- GIVEN 분류 완료 데이터
- WHEN 엑셀 생성 실행
- THEN `reports/two_axis_data_{date}.xlsx` 파일이 생성됨
- AND 편지/게시글 시트에 카테고리 태그, 자유 태그, 요약 컬럼이 포함됨

---

### Requirement: Notion/Slack 배포

생성된 리포트를 Notion에 업로드하고 Slack으로 알림을 전송한다.

#### Scenario: Notion 업로드
- GIVEN 마크다운 리포트가 생성됨
- WHEN Notion 업로드 실행
- THEN Notion 데이터베이스에 리포트 페이지가 생성됨

#### Scenario: Slack 알림
- GIVEN Notion 업로드 완료
- WHEN Slack 알림 실행
- THEN 지정 채널에 리포트 요약 + Notion 링크가 전송됨

---

### Requirement: 사람 검수 프로세스 _(Phase 1 — 미구현)_

자동 분류 결과의 정확도를 측정하고, confidence 기반으로 검수 대상을 추출한다.

#### Scenario: 검수 대상 추출
- GIVEN 주간 분류 완료 데이터
- WHEN 검수 대상 추출 실행
- THEN confidence < 0.7 건 전량 + 고신뢰 건 랜덤 20%가 CSV로 추출됨

#### Scenario: 검수 결과 반영
- GIVEN 검수자가 수정한 라벨 CSV
- WHEN 검수 결과 수집 실행
- THEN 원본 분류 vs 검수 결과 비교 → 정확도 리포트 자동 생성됨
- AND 검수 데이터가 `data/review/`에 축적됨

---

### Requirement: 신뢰 임계값 + 모델 개선 _(Phase 2 — 미구현)_

검수 데이터 기반으로 신뢰 임계값을 설정하고, 사람 검수 비율을 30-50% → 10-15%로 축소한다.

#### Scenario: 자동 통과 판정
- GIVEN 신뢰 임계값이 설정됨 (예: ≥ 0.85)
- WHEN 분류 결과의 confidence가 임계값 이상
- THEN 해당 항목은 사람 검수 없이 자동 통과
- AND 자동 통과 건의 오분류율 5% 이하

---

### Requirement: 재학습 파이프라인 _(Phase 3 — 미구현)_

검수 데이터를 활용하여 주기적으로 모델을 재학습하고, 분류 분포 드리프트를 모니터링한다.

#### Scenario: 월간 재학습
- GIVEN 검수 데이터 500건+ 축적
- WHEN 재학습 파이프라인 실행
- THEN 재학습 전후 테스트셋 정확도 비교 리포트 생성됨

#### Scenario: 드리프트 감지
- GIVEN 주간 분류 분포 이력
- WHEN 부정 비율 10%p+ 급증 감지
- THEN 알림 발생 + 검수 비율 자동 상향

---

### Requirement: Vertex AI 이관 _(Phase 4 — 미구현)_

로컬 파이프라인을 Vertex AI로 이관하여 클라우드 자동화를 달성한다.

#### Scenario: 클라우드 분류
- GIVEN 검수 데이터 1,000건+ & 태깅 스키마 3주+ 안정
- WHEN Vertex AI AutoML 학습
- THEN 로컬 모델 대비 정확도/비용/운영 편의성 비교 리포트 생성됨
