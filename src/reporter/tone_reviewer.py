"""주간 리포트 톤 검수 — 생성된 리포트에서 부서 귀책/평가 표현 탐지 및 수정

마스터를 담당하는 각 부서가 읽기 때문에 민감한 표현을 피해야 한다.
"부서 대응 미흡/부진" 같은 귀책 표현을 탐지하고, 현상 중심의 중립 표현으로 교정한다.
"""
import json
import logging
import re

import boto3

logger = logging.getLogger(__name__)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# 위험 표현 패턴 (사전 탐지)
DANGEROUS_PATTERNS = [
    r"운영진\s*(대응|관리|응대|답변)\s*(미흡|부재|부진|지연|실수)",
    r"(운영|CS|고객지원|제품|사업)\s*(팀|부서)?\s*(대응|관리)\s*(미흡|부재|부진|지연)",
    r"대응\s*(미흡|부재|부진|지연)",
    r"관리\s*(미흡|부재|부족)",
    r"(부서|팀)\s*(의)?\s*(미흡|부진|실수)",
    r"~\s*로\s*인[한해]\s*(불만|실망|혼란)",
    r"~\s*때문에\s*(불만|실망|혼란)",
]


REVIEW_PROMPT = """당신은 금융 교육 플랫폼 주간 리포트의 톤 검수자입니다.

마스터를 담당하는 각 부서(운영팀, CS팀, 제품팀 등)가 리포트를 읽기 때문에,
특정 부서/담당자의 미흡·부진·실수로 이슈가 발생했다는 표현은 **반드시** 피해야 합니다.

## 수정 대상 표현
1. 부서/담당자 귀책: "운영진 대응 미흡", "CS 응대 지연", "관리 부재", "부서 부진"
2. 인과관계 추정: "~로 인한 불만", "~때문에 실망", "~의 영향으로"
3. 자극적 감정 표현: "극심한 불만", "강한 분노", "팽배", "고조"

## 수정 원칙
- 원문의 **사실과 의도는 유지**
- 귀책/평가를 **현상 서술**로 변환
  - X: "운영진 답변 지연으로 불만 고조"
  - O: "문의 후 답변 미수신 사례가 확인되고 있습니다"
  - X: "관리 부재로 혼란"
  - O: "이용 안내에 대한 질문이 반복적으로 확인됩니다"
  - X: "~팀 대응 미흡"
  - O: "문의 진행 상황 확인을 원하는 의견"

## 응답 형식
수정이 필요하면 JSON으로:
{"needs_fix": true, "issues": ["문제 표현 1", "문제 표현 2"], "fixed_text": "수정된 전체 텍스트"}

수정이 필요 없으면:
{"needs_fix": false}

**텍스트의 구조(마크다운, 인용문, 테마 구분 등)와 사실 관계는 그대로 유지**하고,
오직 위험 표현만 교정합니다."""


def detect_dangerous_phrases(text: str) -> list[str]:
    """정규식 기반 사전 탐지"""
    issues = []
    for pattern in DANGEROUS_PATTERNS:
        matches = re.finditer(pattern, text)
        for m in matches:
            issues.append(m.group(0))
    return issues


def review_and_fix(text: str, section_name: str = "") -> tuple[str, list[str]]:
    """LLM으로 톤 검수 + 수정

    Returns:
        (fixed_text, issues_found)
    """
    # 사전 탐지 — 위험 표현이 하나도 없으면 LLM 호출 생략
    detected = detect_dangerous_phrases(text)
    if not detected:
        return text, []

    logger.info(f"  [{section_name}] 위험 표현 {len(detected)}개 탐지: {detected[:3]}")

    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    try:
        resp = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "system": REVIEW_PROMPT,
                "messages": [{"role": "user", "content": text[:8000]}],
            }),
        )
        result = json.loads(resp["body"].read())
        raw = result["content"][0]["text"].strip()

        # JSON 파싱
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            if parsed.get("needs_fix"):
                fixed = parsed.get("fixed_text", text)
                issues = parsed.get("issues", detected)
                return fixed, issues

        return text, detected
    except Exception as e:
        logger.warning(f"톤 검수 실패 ({section_name}): {e}")
        return text, detected


def review_report(report_text: str) -> tuple[str, dict]:
    """리포트 전체 검수 — 섹션 단위로 분할하여 처리

    Returns:
        (fixed_report, stats)
    """
    logger.info("톤 검수 시작")

    # 섹션 단위로 분할 (마스터별 상세는 크기가 크므로 개별 처리)
    # 간단하게 `# ` 또는 `## ` 헤더 기준 분할
    sections = _split_sections(report_text)

    fixed_sections = []
    all_issues = []

    for sec_name, sec_text in sections:
        fixed_text, issues = review_and_fix(sec_text, sec_name)
        fixed_sections.append(fixed_text)
        if issues:
            all_issues.append({"section": sec_name, "issues": issues})

    fixed_report = "\n".join(fixed_sections)

    stats = {
        "total_sections": len(sections),
        "fixed_sections": len(all_issues),
        "total_issues": sum(len(a["issues"]) for a in all_issues),
        "details": all_issues,
    }

    logger.info(f"톤 검수 완료: {stats['fixed_sections']}/{stats['total_sections']} 섹션 수정, {stats['total_issues']}건 교정")
    return fixed_report, stats


def _split_sections(text: str) -> list[tuple[str, str]]:
    """마크다운 헤더 기준으로 섹션 분할 (# 또는 ## 수준)"""
    lines = text.split("\n")
    sections = []
    current_name = "intro"
    current_lines = []

    for line in lines:
        if line.startswith("# ") or line.startswith("## "):
            # 이전 섹션 저장
            if current_lines:
                sections.append((current_name, "\n".join(current_lines)))
            current_name = line.lstrip("#").strip()[:50]
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, "\n".join(current_lines)))

    return sections
