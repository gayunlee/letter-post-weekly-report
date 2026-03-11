# 데이터 분리 전략

## 원칙

Golden set과 학습 데이터의 분리는 파이프라인의 신뢰도를 결정한다.
분리에 실패하면 벤치마크 수치가 허위로 높아져 실제 성능을 알 수 없게 된다.

### 핵심 규칙

1. **`_id` 기반 제외**: golden set 항목의 `_id`와 매칭되는 학습 데이터 제거
2. **텍스트 매칭 이중 검증**: `_id` 제외 후에도 텍스트(`text.strip()`) 기준으로 교집합 재확인
3. **교집합 0 필수**: train/val/test 어디에도 golden set 텍스트가 없어야 함
4. **에이전트 소유권 분리**: eval-agent만 golden set을, data-agent만 clean split을 관리

### 3중 방어 구조

| 레이어 | 담당 | 검증 방법 |
|--------|------|----------|
| 1차 | `prepare_v3_training_data.py` | golden set 텍스트 제외 + stratified split |
| 2차 | `train_v3_topic.py` | 학습 시작 전 golden set leak check |
| 3차 | 에이전트 소유권 | eval-agent ≠ data-agent — 교차 수정 불가 |

## 오염 히스토리

### 74건 누수 사건 (2026-02 ~ 2026-03)

golden set과 학습 데이터가 겹치면서 벤치마크 수치가 허위로 높게 나온 사건.

**타임라인**:

| 시점 | 사건 | 영향 |
|------|------|------|
| T1 | golden set을 학습 데이터에서 별도 분리 없이 생성 | golden set 텍스트가 train에 포함 |
| T2 | 벤치마크 실행 → 정확도 높게 보임 | 실제 성능 대비 과대 평가 |
| T3 | leak check 도입 → 74건 겹침 발견 | 문제 인지 |
| T4 | `prepare_v3_training_data.py`에 golden 제외 로직 추가 + 이중 검증 | 문제 해결 |

**교훈**:
- golden set을 나중에 만들면 이미 학습 데이터에 포함된 텍스트가 golden에 들어갈 수 있다
- 반드시 golden set 텍스트를 학습 데이터에서 **적극적으로 제외**해야 한다
- 단순 `_id` 매칭은 불충분 — 같은 텍스트가 다른 `_id`로 존재할 수 있다

### 63.2% 사건 — Golden 라벨 품질 문제

LLM 배치 라벨을 검수 없이 golden set에 넣었을 때, 모델이 golden 라벨보다 정확한 역전 현상 발생.

**원인**: LLM 배치 라벨링의 오류율이 ~15%이므로, 이를 "정답"으로 쓰면 정확한 모델이 오히려 틀린 것으로 측정됨.

**대응**: Golden set 항목은 반드시 1건씩 사람이 텍스트를 읽고 판단 근거를 기록한 뒤 확정.

## 멀티 도메인 확장 원칙

채널톡 등 새 도메인 추가 시:

1. **독립 golden set**: 도메인별 별도 golden set 파일 (`v6_golden_set.json`, `cs_v1_golden_set.json`)
2. **독립 manifest**: 도메인별 manifest (`letter_post.json`, `channel_talk.json`)
3. **동일 3중 방어**: 각 도메인 파이프라인에 동일한 분리 구조 적용
4. **교차 오염 방지**: 한 도메인의 golden set이 다른 도메인 학습 데이터에 영향을 주지 않음

## 검증 체크리스트

clean split 생성 후 반드시 확인:

- [ ] golden set 전체 텍스트 vs train.json 교집합 = 0
- [ ] golden set 전체 텍스트 vs val.json 교집합 = 0
- [ ] golden set 전체 텍스트 vs test.json 교집합 = 0
- [ ] manifest `leak_check_passed` = true
- [ ] `train_v3_topic.py` 자체 leak check 통과
