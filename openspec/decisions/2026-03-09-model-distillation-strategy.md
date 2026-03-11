# Decision: KcBERT → LLM 디스틸레이션 전략 전환

- **날짜**: 2026-03-09
- **상태**: 확정
- **관련**: phase-0.5-taxonomy-redesign

## 배경

v3c 5분류 체계에서 KcBERT(110M) 파인튜닝이 Golden 60.4%에서 정체. 소수 카테고리 데이터 보강(서비스 7배, 운영 5배)에도 +1.4%p만 개선.

**근본 원인**: KcBERT는 토큰 수준 패턴 매칭 모델. 이 태스크는 "같은 단어(멤버십, 세미나)가 맥락에 따라 다른 카테고리"인 의도 추론 태스크 → 110M 파라미터로 해결 불가.

## 벤치마크 결과 (개별 API 호출, Golden 227건)

| 모델 | 정확도 | macro F1 | 비고 |
|------|--------|----------|------|
| KcBERT (fine-tuned, 110M) | 60.4% | — | 의도 추론 불가 |
| Gemini 2.5 Flash (prompt) | 67.4% | — | 기타→콘텐츠 오분류 심함 |
| GPT-4o-mini (prompt) | 75.3% | 0.735 | 운영 recall 59% |
| **Haiku 4.5 (prompt)** | **78.9%** | 0.776 | 가장 균형 |
| Opus 4.6 (라벨러) | ~98%* | — | Teacher 모델 |

*Opus는 Golden set 라벨 생성자이므로 정확한 측정 불가. 사람 검수에서 극소수만 수정.

## 의사결정

### 전환: KcBERT → LLM 디스틸레이션

**Opus(Teacher) → 저렴한 모델(Student) 파인튜닝**

- 분류체계는 건전 (Opus가 증명)
- 프롬프트도 건전 (Opus에게 충분)
- 핵심 문제: Opus 성능을 저렴한 모델로 재현하는 것 = 모델 디스틸레이션
- 학습 데이터: Opus 라벨 13,995건 (이미 확보)

### 파인튜닝 후보

#### 1차 (Vertex AI 병렬 — 한국어 특화 + Gemini)

| 모델 | 크기 | 개발사 | 파인튜닝 방식 | 한국어 |
|------|------|--------|-------------|--------|
| **Gemini 2.5 Flash** | 비공개 | Google | Vertex AI 네이티브 | 지원 |
| **EXAONE 3.5 7.8B** | 7.8B | LG AI Research | Vertex AI A100 + LoRA | 네이티브 (어휘 50%) |
| **Solar 10.7B** | 10.7B | Upstage | Vertex AI A100 + LoRA | 네이티브 (한국 기업) |

**모델 선정 근거:**
- EXAONE: LG AI Research, 한국어 어휘 50%, EXAONE 4.0 후속 로드맵 존재
- Solar: Upstage, "파인튜닝에 최적화된 LLM" 포지셔닝, Ko-LLM 리더보드 최상위
- Gemini: Vertex AI 네이티브로 세팅 최소, 추가 비용 거의 없음
- 한국어 벤치마크에서 Solar Open, EXAONE, K-EXAONE이 나란히 최상위 성능

#### 2차 (1차 90% 미달 시)

| 모델 | 방식 | 비고 |
|------|------|------|
| EXAONE 3.5 32B | Vertex AI A100 80GB or 2장 | 용량 업그레이드 |
| GPT-4.1-mini | OpenAI Fine-tuning API | Vertex AI 불가, OpenAI 전용 |

### 기각한 옵션

- **Qwen 2.5**: 2026.03 핵심 인력 3명 연쇄 퇴사 (테크 리드 Lin Junyang, 포스트트레이닝 헤드 Yu Bowen, 기여자 Lin Kaixin). 알리바바 주가 5.3% 하락. 장기 유지보수 리스크
- **Haiku 파인튜닝**: Anthropic이 Claude 파인튜닝 미제공
- **프롬프트 최적화만**: Opus가 같은 프롬프트로 잘하므로 프롬프트가 아닌 모델 능력 차이. 천장 있음
- **KcBERT 계속**: 데이터가 아닌 모델 용량(110M) 한계. 추가 데이터 무의미
- **EXAONE 2.4B**: 시행착오 최소화를 위해 7.8B로 시작. 7.8B가 충분하면 경량화로 2.4B 재시도 가능

## 비용 비교 (주 1,300건)

### 파인튜닝 비용 (1회)

| 모델 | 환경 | 예상 비용 |
|------|------|----------|
| Gemini Flash FT | Vertex AI 관리형 | ~$5-10 (토큰 과금) |
| EXAONE 7.8B LoRA | Vertex AI A100 ~1-2h | ~$5 |
| Solar 10.7B LoRA | Vertex AI A100 ~1-2h | ~$5 |

### 추론 비용 (주간 운영)

| 방식 | 주간 비용 | 연간 비용 |
|------|----------|----------|
| EXAONE/Solar 자체 서빙 | ~$0.05 | ~$3 |
| Gemini Flash FT API | ~$0.10 | ~$5 |
| Haiku prompt-only | ~$0.20 | ~$10 |
| Opus 매번 호출 | ~$2.00 | ~$100 |

→ 모든 Student 옵션이 연 $10 이하. 비용이 아닌 정확도가 유일한 기준.

## 실행 계획

```
1차: Vertex AI 병렬 (3개 동시)
─────────────────────────────────
1. 데이터 준비
   - Golden 227건 제외 clean split (기존 data-agent 활용)
   - Gemini FT 포맷 변환 (JSONL)
   - EXAONE/Solar SFT 포맷 변환 (instruction-response JSONL)

2. Vertex AI 병렬 파인튜닝
   - Gemini 2.5 Flash: Vertex AI 네이티브 FT
   - EXAONE 3.5 7.8B: Vertex AI A100 + LoRA
   - Solar 10.7B: Vertex AI A100 + LoRA

3. 벤치마크 (bench-agent)
   - Golden 227건 기준 정확도 비교
   - 목표: 90%+
   - version_log 기록

4. 의사결정
   - 90%+ 달성 모델 채택 → 주간 파이프라인 전환
   - 전부 90% 미달 → 2차 진행

2차: 스케일업 (1차 실패 시)
─────────────────────────────────
5. 모델 업그레이드
   - EXAONE 32B (Vertex AI)
   - GPT-4.1-mini FT (OpenAI API)

6. 재벤치마크 + 최종 의사결정
```
