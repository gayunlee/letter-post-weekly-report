# Proposal: Phase 1 — 사람 검수 기반 구축

## Intent

자동 분류 결과의 정확도 기준선을 확립하고, 검수 프로세스를 정의한다.
현재 분류 품질이 "체감상 좋다"는 수준이지, 정량적 정확도가 측정되지 않았다.
다른 팀이 분류 데이터를 신뢰하고 활용하려면 정확도 수치가 필요하다.

## Scope

In scope:
- confidence score 기반 검수 대상 추출
- 검수용 CSV/Google Sheet 내보내기
- 검수 결과 수집 + 원본 대비 정확도 리포트
- 레이어별 정확도 기준선 측정 (Topic / Sentiment / Category Tag)

Out of scope:
- 모델 재학습 (Phase 2)
- 채널톡 데이터 (Phase 2)
- 드리프트 모니터링 (Phase 3)
- Vertex AI 이관 (Phase 4)

## Approach

1. 2축 분류의 기존 confidence score + detail_tags의 LLM 응답 신뢰도를 결합
2. 주간 데이터에서 검수 대상 자동 추출 (low-confidence 전량 + high-confidence 랜덤 샘플)
3. 비개발 팀이 사용할 수 있는 CSV 형태로 내보내기
4. 검수 결과를 `data/review/`에 수집하고, 원본 대비 차이를 자동 집계
5. 2주간 30~50% 샘플 검수 → 레이어별 정확도 기준선 도출

## 의사결정 포인트

검수 완료 후 정확도에 따라 Phase 2 방향이 결정됨:

| Category Tag 정확도 | 다음 액션 |
|---------------------|----------|
| 85% 이상 | few-shot 유지, Phase 2에서 임계값만 설정 |
| 80~85% | 해당 카테고리 예제 보강 시도 |
| 80% 미만 | Phase 2에서 파인튜닝 검토 |
