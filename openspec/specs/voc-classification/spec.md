# VOC Classification Specification

## Purpose

금융 콘텐츠 플랫폼(오피셜클럽)의 편지글/게시글/채널톡 VOC 데이터를 다층 분류하여, 주간 인사이트 리포트 자동 생성 및 다른 팀의 자유 질의 응답을 가능하게 하는 시스템.

### 주간 리포트의 핵심 목적

1. **마스터별 여론 파악** — 각 마스터 커뮤니티에 어떤 여론이 형성되고 있는가 (긍/부정, 시장 상황이나 개인 수익에 따른 반응). 사업부서가 맡은 마스터에 대한 여론을 확인한다.
2. **액셔너블 피드백 도출** — 사업부서(운영)와 플랫폼파트(서비스)는 각각 어떤 대응이나 개선이 필요한지 도출한다.
   - 운영 예: 오프라인 세미나 안내에 혼란 → 안내 강화
   - 콘텐츠 예: 특정 콘텐츠에 대한 불만 표출 → 콘텐츠 개선
   - 서비스 예: 서비스 장애/기능 불만/기능 건의 → 개발 반영
3. **증시 여론 감정 확인** — 전체적인 증시 상황에 따른 이용자 감정 변화를 추적한다.
4. **부정 여론 감지** — 심하게 부정적인 여론이 있을 경우 감지하여 필터링 필요 여부를 판단한다.

### 데이터 현실 제약

- **커뮤니티 특성**: 감사·응원 글이 압도적으로 많다 (투자 교육 플랫폼이므로 대부분의 글이 투자/콘텐츠 관련).
- **소수 클래스 문제**: 운영 피드백, 서비스 피드백은 전체 데이터에서 극소수이나, 리포트에서의 비즈니스 가치는 가장 높다.
- **분류 체계 딜레마**: 콘텐츠/투자를 세분화하면 혼동률 증가, 합치면 87.9% 쏠림으로 모델 학습 불균형 발생.

## 설계 원칙

- **자유도 우선**: 특정 부서의 뷰를 하드코딩하지 않고, 풍부한 분류 필드를 제공하여 누구든 자기 목적에 맞게 조합·쿼리할 수 있게 한다.
- **분류 축은 최소한으로**: Topic(주제) × Sentiment(감성) 2축 + detail_tags(카테고리 태그 28종 + 자유 태그 + 요약)로 충분한 분석 자유도를 확보한다.
- **Intent(의도)는 부수 데이터**: detail_tags 추출 시 동일 LLM 호출에서 함께 추출. 별도 축이 아닌 쿼리용 보조 필드.
- **정확도 향상은 검수 → 재학습 사이클로**: 모델 추가가 아닌 기존 모델의 정확도를 높이는 방향.

## Requirements

> **⚠️ v3 전환 진행 중** (2026-03-02)
> 아래 Topic 4분류는 현행 v2 (소재 기반). v3 5분류(운영 피드백/서비스 피드백/콘텐츠 반응/투자 담론/기타)로 전환 진행 중.
> 상세: [phase-0.5-taxonomy-redesign](../changes/phase-0.5-taxonomy-redesign/specs/voc-classification/spec.md)

### Requirement: 2축 분류 (Topic × Sentiment)

파인튜닝된 KcBERT 모델로 편지글/게시글을 Topic 4개 × Sentiment 3개로 분류한다.

#### Topic (v2, 현행)
| 값 | 설명 |
|----|------|
| 콘텐츠 반응 | 마스터 콘텐츠(강의, 리포트, 방송)에 대한 반응 |
| 투자 이야기 | 투자 전략, 종목, 포트폴리오, 시장 분석 |
| 서비스 이슈 | 플랫폼/앱 기능, 결제, 배송, 구독 등 |
| 커뮤니티 소통 | 인사, 안부, 축하, 일상 공유 |

#### Sentiment
| 값 | 설명 |
|----|------|
| 긍정 | 감사, 만족, 기쁨, 칭찬 |
| 부정 | 불만, 실망, 걱정, 비판 |
| 중립 | 질문, 정보 전달, 사실 기술 |

- **모델**: `models/two_axis/topic/final_model`, `models/two_axis/sentiment/final_model`
- **현재 정확도**: Topic 69.3%, Sentiment 84.1%

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

### Requirement: Detail Tags 추출

2축 분류 후, LLM(Claude Haiku)으로 세부 태그를 추출한다.

- **모델**: claude-haiku-4-5-20251001 ($3.12/주, 2,542건 기준)
- **카테고리 태그**: Topic별 6~8개, 총 28개 controlled vocabulary
- **자유 태그**: 구체적 명사구 2~3개 (검색용)
- **요약**: 15~40자 한줄 요약
- **의도(Intent)**: 질문/요청, 피드백/의견, 제보/건의, 정보공유 (동일 호출에서 부수 추출)
- **품질 기준**: 목록 외 태그 비율 < 5%, 태그 커버리지 > 90%

#### Scenario: 태그 + Intent 추출
- GIVEN 2축 분류가 완료된 항목
- WHEN detail_tags 추출 실행
- THEN 각 항목에 detail_tags + intent가 부여됨
- AND category_tags는 반드시 28개 허용 목록에서만 선택됨
- AND 이미 detail_tags가 있는 항목은 건너뜀
- AND 추가 API 호출 없이 기존 Haiku 1회 호출에서 통합 추출

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
- GIVEN 각 Topic의 부정 항목들
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
- AND 편지/게시글 시트에 카테고리 태그, 자유 태그, 요약, 의도 컬럼이 포함됨

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

### Requirement: 자연어 채팅 인터페이스 _(구축 예정)_

분류된 VOC 데이터를 자연어로 질문하고, 인사이트를 받을 수 있는 웹 채팅 인터페이스.

#### Scenario: 자연어 쿼리
- GIVEN 분류 완료 데이터가 있음
- WHEN 사용자가 "이번 주 결제 관련 불만 몇 건이야?" 질문
- THEN LLM이 쿼리 조건 추출 → 데이터 필터링 → 인사이트 요약 → 답변
- AND 답변에 근거 데이터 건수 + 원본 보기 링크 제공

#### Scenario: 원본 검증
- GIVEN 채팅 답변이 제공됨
- WHEN 사용자가 원본 보기 클릭
- THEN 필터링된 원본 데이터 테이블이 표시됨
- AND 오분류 발견 시 피드백 가능 → 검수 데이터로 축적

상세 시나리오: [analysis-scenarios.md](./analysis-scenarios.md)

---

### Requirement: 채널톡 CS 데이터 분류 _(2차 우선순위)_

채널톡 CS 문의 데이터를 BigQuery에서 조회하여 분류한다.

- **데이터소스**: `us-service-data.channel_io.messages`
- **분류 단위**: chatId 기준 1대화 = 1분류
- **전처리**: 사용자 메시지만 연결, 앞 500자 + 마지막 200자

#### Scenario: 채널톡 대화 분류
- GIVEN 1주치 채널톡 대화가 BigQuery에서 조회됨
- WHEN 분류 파이프라인 실행
- THEN chatId별로 Topic, Sentiment, detail_tags가 부여됨

---

### Requirement: 사람 검수 프로세스 _(Phase 1)_

자동 분류 결과의 정확도를 측정하고, confidence 기반으로 검수 대상을 추출한다.

#### Scenario: 검수 대상 추출
- GIVEN 주간 분류 완료 데이터
- WHEN 검수 대상 추출 실행
- THEN confidence 낮은 건 전량 + 고신뢰 건 랜덤 샘플이 CSV로 추출됨

#### Scenario: 검수 결과 반영
- GIVEN 검수자가 수정한 라벨
- WHEN 검수 결과 수집 실행
- THEN 원본 분류 vs 검수 결과 비교 → 정확도 리포트 자동 생성됨
- AND 검수 데이터가 `data/review/`에 축적됨

---

### Requirement: 모델 개선 + 재학습 _(Phase 2)_

검수 데이터 기반으로 모델을 재학습하여 정확도를 향상시킨다.

#### Scenario: 재학습
- GIVEN 검수 데이터 500건+ 축적
- WHEN 재학습 파이프라인 실행
- THEN 재학습 전후 테스트셋 정확도 비교 리포트 생성됨
- AND Topic 정확도 69.3% → 80%+ 목표

---

### Requirement: 드리프트 모니터링 _(Phase 3)_

분류 분포 변화를 감지하고 알림을 발송한다.

#### Scenario: 드리프트 감지
- GIVEN 주간 분류 분포 이력
- WHEN 이상 감지 (분포 급변, confidence 급락)
- THEN Slack 알림 발생
