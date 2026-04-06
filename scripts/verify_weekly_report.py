"""주간 리포트 자동 검증 스크립트 (새벽 4시 실행)

검증 항목:
1. 파일 존재 및 크기
2. 필수 섹션 존재 (핵심 요약, 핵심 이슈, 오피셜클럽별 상세)
3. 마크다운 문법 검증 (노션 발행용)
4. 이전 리포트와 구조 비교 (섹션 누락 체크)
5. 톤 검수 (부서 귀책 표현 잔존 여부)

결과는 /tmp/weekly_voc_verify.log + 슬랙 알림 (선택)
"""
import sys
import os
import re
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reporter.tone_reviewer import detect_dangerous_phrases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")


# ── 필수 섹션 ──
REQUIRED_SECTIONS = [
    r"^# 📌.*이용자 반응 리포트",
    r"^# 0\. 핵심 요약",
    r"^# 1\. 오피셜클럽별 상세",
]

REQUIRED_SUBSECTIONS_PER_MASTER = [
    "주요 내용",
    "서비스 피드백",
]


def find_latest_report() -> Path | None:
    """이번 주 월요일 날짜의 리포트 찾기"""
    # 이번 주 월요일
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    monday_str = monday.strftime("%Y-%m-%d")

    candidates = [
        REPORTS_DIR / f"weekly_report_{monday_str}.md",
        REPORTS_DIR / f"weekly_report_v5_{monday_str}.md",
    ]

    for path in candidates:
        if path.exists():
            return path

    # 가장 최근 파일로 fallback
    files = sorted(REPORTS_DIR.glob("weekly_report_*.md"))
    return files[-1] if files else None


def verify_file(path: Path) -> dict:
    """파일 기본 검증"""
    result = {"file": str(path), "ok": True, "issues": []}

    if not path.exists():
        result["ok"] = False
        result["issues"].append("파일 존재하지 않음")
        return result

    size = path.stat().st_size
    result["size_bytes"] = size

    if size < 1000:
        result["ok"] = False
        result["issues"].append(f"파일 크기 너무 작음: {size} bytes (1KB 미만)")
    elif size < 5000:
        result["issues"].append(f"⚠️ 파일 크기 작음: {size} bytes (5KB 미만)")

    return result


def verify_required_sections(text: str) -> dict:
    """필수 섹션 존재 여부"""
    result = {"ok": True, "missing": [], "found": []}

    for pattern in REQUIRED_SECTIONS:
        if re.search(pattern, text, re.MULTILINE):
            result["found"].append(pattern)
        else:
            result["ok"] = False
            result["missing"].append(pattern)

    return result


def verify_markdown_syntax(text: str) -> dict:
    """마크다운 문법 검증 (노션 발행 기준)"""
    result = {"ok": True, "issues": []}

    # 1. 언밸런스 볼드 (**)
    bold_count = text.count("**")
    if bold_count % 2 != 0:
        result["ok"] = False
        result["issues"].append(f"볼드(**) 홀수개 발견: {bold_count}개")

    # 2. 언밸런스 인용 블록 (> 다음 내용 없음)
    quote_lines = [line for line in text.split("\n") if line.strip().startswith(">")]
    empty_quotes = [l for l in quote_lines if l.strip() == ">"]
    if len(empty_quotes) > 5:
        result["issues"].append(f"⚠️ 빈 인용 블록 {len(empty_quotes)}개")

    # 3. 헤더 뒤 빈 줄 체크
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^#+ ", line):
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith("#"):
                # 헤더 바로 뒤 텍스트 붙어있음 (노션에서 렌더 이슈)
                pass  # 경고만, 실패 아님

    # 4. 코드블록 페어
    code_blocks = text.count("```")
    if code_blocks % 2 != 0:
        result["ok"] = False
        result["issues"].append(f"코드블록(```) 홀수개: {code_blocks}개")

    # 5. 테이블 구조 검증 (| 개수 일관성)
    in_table = False
    table_col_count = 0
    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            cols = line.count("|")
            if not in_table:
                table_col_count = cols
                in_table = True
            elif cols != table_col_count:
                # 구분선(|---|---|)은 예외
                if not re.match(r"^[\s|:-]+$", line):
                    result["issues"].append(f"⚠️ 테이블 컬럼 불일치: {line.strip()[:50]}")
        else:
            in_table = False

    # 6. 링크 문법 깨짐 ([text]() 또는 [text](url) 미완성)
    broken_links = re.findall(r"\[[^\]]*\]\([^)]*$", text, re.MULTILINE)
    if broken_links:
        result["ok"] = False
        result["issues"].append(f"깨진 링크 {len(broken_links)}개")

    # 7. 리스트 indentation
    # (노션은 2/4 스페이스 들여쓰기 엄격)

    return result


def verify_master_sections(text: str) -> dict:
    """마스터별 상세 섹션 구조 검증"""
    result = {"ok": True, "masters": [], "issues": []}

    # ## N. 마스터이름 패턴
    master_pattern = re.compile(r"^## \d+\.\s*(.+?)$", re.MULTILINE)
    masters = master_pattern.findall(text)
    result["masters"] = masters
    result["master_count"] = len(masters)

    if len(masters) == 0:
        result["ok"] = False
        result["issues"].append("마스터별 상세 섹션 없음")
        return result

    # 각 마스터 섹션이 주요 내용/서비스 피드백 포함하는지
    # 간단히 전체 텍스트에서 필수 키워드 카운트
    for sub in REQUIRED_SUBSECTIONS_PER_MASTER:
        count = text.count(sub)
        if count < len(masters) * 0.5:  # 절반 이상 있어야
            result["issues"].append(f"⚠️ '{sub}' 섹션 부족: {count}/{len(masters)}")

    return result


def compare_with_previous(current_path: Path) -> dict:
    """이전 주 리포트와 구조 비교"""
    result = {"ok": True, "issues": []}

    # 이전 리포트 찾기
    files = sorted(REPORTS_DIR.glob("weekly_report_*.md"))
    files = [f for f in files if f != current_path]
    if not files:
        result["issues"].append("이전 리포트 없음 (첫 실행)")
        return result

    prev_path = files[-1]
    result["prev_file"] = str(prev_path)

    with open(current_path) as f:
        current = f.read()
    with open(prev_path) as f:
        previous = f.read()

    # 크기 비교
    ratio = len(current) / max(len(previous), 1)
    if ratio < 0.3:
        result["ok"] = False
        result["issues"].append(f"리포트 크기가 이전 대비 {ratio:.0%}로 급감 ({len(current)} vs {len(previous)})")
    elif ratio > 3.0:
        result["issues"].append(f"⚠️ 리포트 크기 급증: {ratio:.0%}")

    # 마스터 수 비교
    cur_masters = len(re.findall(r"^## \d+\.\s", current, re.MULTILINE))
    prev_masters = len(re.findall(r"^## \d+\.\s", previous, re.MULTILINE))
    if cur_masters < prev_masters - 2:
        result["issues"].append(f"⚠️ 마스터 수 감소: {cur_masters} (이전 {prev_masters})")

    return result


def verify_tone(text: str) -> dict:
    """부서 귀책/평가 표현 잔존 여부"""
    dangerous = detect_dangerous_phrases(text)
    return {
        "ok": len(dangerous) == 0,
        "dangerous_phrases": dangerous,
    }


def main():
    print("=" * 60)
    print("📋 주간 리포트 자동 검증")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    path = find_latest_report()
    if not path:
        print("❌ 리포트 파일 없음")
        sys.exit(1)

    print(f"📄 검증 대상: {path}")
    print()

    # 파일 검증
    file_result = verify_file(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # 섹션 검증
    section_result = verify_required_sections(text)
    master_result = verify_master_sections(text)
    markdown_result = verify_markdown_syntax(text)
    tone_result = verify_tone(text)
    compare_result = compare_with_previous(path)

    # 결과 출력
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "file": str(path),
        "file_check": file_result,
        "required_sections": section_result,
        "master_sections": master_result,
        "markdown_syntax": markdown_result,
        "tone_check": tone_result,
        "comparison": compare_result,
    }

    overall_ok = (
        file_result["ok"]
        and section_result["ok"]
        and master_result["ok"]
        and markdown_result["ok"]
        and tone_result["ok"]
        and compare_result["ok"]
    )
    all_results["overall_ok"] = overall_ok

    print(f"1. 파일: {'✅' if file_result['ok'] else '❌'} ({file_result.get('size_bytes', 0):,} bytes)")
    for issue in file_result["issues"]:
        print(f"   - {issue}")

    print(f"2. 필수 섹션: {'✅' if section_result['ok'] else '❌'}")
    for missing in section_result["missing"]:
        print(f"   - 누락: {missing}")

    print(f"3. 마스터 섹션: {'✅' if master_result['ok'] else '⚠️'} ({master_result['master_count']}개)")
    for issue in master_result["issues"]:
        print(f"   - {issue}")

    print(f"4. 마크다운 문법: {'✅' if markdown_result['ok'] else '❌'}")
    for issue in markdown_result["issues"]:
        print(f"   - {issue}")

    print(f"5. 톤 검수: {'✅' if tone_result['ok'] else '⚠️'}")
    for p in tone_result["dangerous_phrases"][:5]:
        print(f"   - 위험 표현: {p}")

    print(f"6. 이전 리포트 비교: {'✅' if compare_result['ok'] else '❌'}")
    for issue in compare_result["issues"]:
        print(f"   - {issue}")

    print()
    print("=" * 60)
    print(f"종합: {'✅ 통과' if overall_ok else '❌ 실패'}")
    print("=" * 60)

    # 결과 저장
    verify_log = Path("/tmp/weekly_voc_verify.json")
    with open(verify_log, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"📝 결과 저장: {verify_log}")

    # 슬랙 알림 (선택)
    try:
        from src.integrations.slack_client import SlackNotifier
        slack = SlackNotifier()

        status = "✅ 통과" if overall_ok else "❌ 실패"
        msg_lines = [
            f"[주간 리포트 자동 검증] {status}",
            f"📄 {path.name}",
            "",
        ]

        if not overall_ok:
            if not file_result["ok"]:
                msg_lines.append(f"• 파일: {', '.join(file_result['issues'])}")
            if not section_result["ok"]:
                msg_lines.append(f"• 누락 섹션: {', '.join(section_result['missing'])}")
            if not markdown_result["ok"]:
                msg_lines.append(f"• 마크다운: {', '.join(markdown_result['issues'])}")
            if not tone_result["ok"]:
                msg_lines.append(f"• 톤 이슈: {len(tone_result['dangerous_phrases'])}건")
            if not compare_result["ok"]:
                msg_lines.append(f"• 비교: {', '.join(compare_result['issues'])}")

        slack._send_message("\n".join(msg_lines))
    except Exception as e:
        logger.warning(f"슬랙 전송 실패: {e}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
