#!/usr/bin/env python3
"""VOC 라벨링 도구

터미널 기반 인터랙티브 라벨링 도구입니다.

사용법:
    python scripts/labeling_tool.py

조작:
    1-5: 카테고리 선택
    s: 건너뛰기 (Skip)
    b: 이전 항목으로 (Back)
    q: 저장 후 종료 (Quit)
    h: 도움말 (Help)
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import random

# 새로운 5개 카테고리 정의
CATEGORIES = {
    "1": "긍정 피드백",      # 감사, 칭찬, 만족 표현
    "2": "부정 피드백",      # 불만, 개선요청, 실망
    "3": "질문/문의",        # 투자 질문 + 서비스 문의
    "4": "정보 공유",        # 뉴스, 분석, 의견 공유
    "5": "일상 소통",        # 인사, 안부, 잡담
}

CATEGORY_HINTS = {
    "1": "감사합니다, 덕분에, 도움이 됐어요, 수익 후기, 만족",
    "2": "불편, 답답, 실망, 아쉽, 개선 요청, 불만",
    "3": "?로 끝남, 어떻게, 궁금, 알려주세요, 서비스 문의",
    "4": "뉴스, 속보, 분석 공유, 본인 의견/전망 제시",
    "5": "인사, 안부, 축하, 개인 이야기, 잡담",
}

# 기존 카테고리 → 새 카테고리 매핑 (참고용)
OLD_TO_NEW_MAPPING = {
    "감사·후기": "긍정 피드백",
    "질문·토론": "질문/문의",
    "정보성 글": "정보 공유",
    "서비스 피드백": "질문/문의",  # 문의 형태로 작성됨
    "불편사항": "부정 피드백",
    "일상·공감": "일상 소통",
}


def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(current: int, total: int, labeled: int):
    """헤더 출력"""
    progress = labeled / total * 100 if total > 0 else 0
    print("=" * 70)
    print(f"  VOC 라벨링 도구  |  진행: {current}/{total} ({progress:.1f}% 완료)  |  라벨링됨: {labeled}")
    print("=" * 70)


def print_categories():
    """카테고리 목록 출력"""
    print("\n[카테고리 선택]")
    for key, name in CATEGORIES.items():
        hint = CATEGORY_HINTS[key]
        print(f"  {key}. {name}")
        print(f"     → {hint}")
    print()


def print_controls():
    """조작 안내 출력"""
    print("-" * 70)
    print("  1-5: 카테고리 선택  |  s: 건너뛰기  |  b: 이전  |  q: 저장후 종료  |  h: 도움말")
    print("-" * 70)


def print_help():
    """도움말 출력"""
    clear_screen()
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         라벨링 도움말                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  [분류 기준]                                                          ║
║                                                                      ║
║  1. 긍정 피드백                                                       ║
║     - "감사합니다", "덕분에", "도움이 됐어요"                            ║
║     - 수익 후기, 만족 표현                                              ║
║     - 복합 시: 감사 표현이 주된 목적이면 긍정 피드백                      ║
║                                                                      ║
║  2. 부정 피드백                                                       ║
║     - "불편", "답답", "실망", "아쉽"                                    ║
║     - 개선 요청, 불만 제기                                              ║
║     - 복합 시: 불만 해소가 주된 목적이면 부정 피드백                      ║
║                                                                      ║
║  3. 질문/문의                                                         ║
║     - 물음표(?)로 끝나는 문장                                           ║
║     - "어떻게", "궁금", "알려주세요"                                     ║
║     - 서비스 문의 (배송, 결제, 링크 등)                                  ║
║     - 복합 시: 답변을 기대하면 질문/문의                                 ║
║                                                                      ║
║  4. 정보 공유                                                         ║
║     - 뉴스, 속보, 분석 공유                                             ║
║     - 본인 의견/전망 제시                                               ║
║     - 복합 시: 정보 전달이 주된 목적이면 정보 공유                        ║
║                                                                      ║
║  5. 일상 소통                                                         ║
║     - 인사, 안부, 축하                                                  ║
║     - 개인 이야기, 잡담                                                 ║
║     - 복합 시: 소통 자체가 목적이면 일상 소통                            ║
║                                                                      ║
║  [복합 의도 판단 규칙]                                                  ║
║  - 감사 + 질문 → 질문이 실질적 내용이면 "질문/문의"                      ║
║  - 인사 + 감사 → 감사가 구체적이면 "긍정 피드백", 형식적이면 "일상 소통"  ║
║  - 정보 + 질문 → 질문이 핵심이면 "질문/문의"                             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    input("\n아무 키나 누르면 계속...")


def print_item(item: Dict, index: int):
    """항목 출력"""
    text = item.get("text", "")
    source = item.get("source", "unknown")
    old_category = item.get("old_category", "없음")
    new_label = item.get("new_label")

    print(f"\n[항목 #{index + 1}] (출처: {source})")
    print(f"기존 분류: {old_category}")
    if new_label:
        print(f"현재 라벨: {new_label} ✓")
    print()
    print("-" * 70)
    # 텍스트를 줄 단위로 출력 (읽기 편하게)
    for line in text.split("\n"):
        print(f"  {line}")
    print("-" * 70)


def load_labeling_data(input_path: Path) -> List[Dict]:
    """라벨링 데이터 로드"""
    if input_path.exists():
        with open(input_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_labeling_data(data: List[Dict], output_path: Path):
    """라벨링 데이터 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_items_from_classified_data(data_dir: Path, sample_size: int = 750) -> List[Dict]:
    """기존 분류 데이터에서 라벨링용 항목 추출

    카테고리별로 균형있게 샘플링합니다.
    """
    all_items = []

    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # 편지
        for item in data.get("letters", []):
            text = item.get("message", "").strip()
            old_cat = item.get("classification", {}).get("category", "")
            if text:
                all_items.append({
                    "id": item.get("_id", ""),
                    "text": text,
                    "source": "letter",
                    "old_category": old_cat,
                    "suggested_label": OLD_TO_NEW_MAPPING.get(old_cat, ""),
                    "new_label": None,
                    "file": json_file.name
                })

        # 게시글
        for item in data.get("posts", []):
            text = (item.get("textBody") or item.get("body") or "").strip()
            old_cat = item.get("classification", {}).get("category", "")
            if text:
                all_items.append({
                    "id": item.get("_id", ""),
                    "text": text,
                    "source": "post",
                    "old_category": old_cat,
                    "suggested_label": OLD_TO_NEW_MAPPING.get(old_cat, ""),
                    "new_label": None,
                    "file": json_file.name
                })

    # 기존 카테고리별로 그룹화
    by_old_category = {}
    for item in all_items:
        cat = item["old_category"]
        if cat not in by_old_category:
            by_old_category[cat] = []
        by_old_category[cat].append(item)

    # 각 카테고리에서 균등하게 샘플링
    sampled = []
    per_category = sample_size // len(by_old_category) if by_old_category else 0

    for cat, items in by_old_category.items():
        random.seed(42)  # 재현성
        random.shuffle(items)
        sampled.extend(items[:per_category])

    # 부족분 채우기
    remaining = sample_size - len(sampled)
    if remaining > 0:
        used_ids = {item["id"] for item in sampled}
        extras = [item for item in all_items if item["id"] not in used_ids]
        random.shuffle(extras)
        sampled.extend(extras[:remaining])

    random.shuffle(sampled)
    return sampled[:sample_size]


def run_labeling_session(data: List[Dict], output_path: Path):
    """라벨링 세션 실행"""
    current_index = 0

    # 첫 번째 미라벨링 항목 찾기
    for i, item in enumerate(data):
        if item.get("new_label") is None:
            current_index = i
            break

    while True:
        clear_screen()

        # 라벨링 완료 수
        labeled_count = sum(1 for item in data if item.get("new_label") is not None)

        print_header(current_index + 1, len(data), labeled_count)
        print_item(data[current_index], current_index)
        print_categories()
        print_controls()

        # 입력 받기
        try:
            choice = input("\n선택: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = 'q'

        if choice in CATEGORIES:
            # 카테고리 선택
            data[current_index]["new_label"] = CATEGORIES[choice]
            # 다음 항목으로
            if current_index < len(data) - 1:
                current_index += 1
            else:
                # 마지막 항목 완료
                save_labeling_data(data, output_path)
                clear_screen()
                print("\n🎉 모든 항목 라벨링 완료!")
                print(f"저장 위치: {output_path}")
                break

        elif choice == 's':
            # 건너뛰기
            if current_index < len(data) - 1:
                current_index += 1
            else:
                print("\n마지막 항목입니다.")
                input("아무 키나 누르면 계속...")

        elif choice == 'b':
            # 이전으로
            if current_index > 0:
                current_index -= 1
            else:
                print("\n첫 번째 항목입니다.")
                input("아무 키나 누르면 계속...")

        elif choice == 'h':
            # 도움말
            print_help()

        elif choice == 'q':
            # 저장 후 종료
            save_labeling_data(data, output_path)
            clear_screen()
            print(f"\n저장 완료: {output_path}")
            print(f"라벨링됨: {labeled_count}/{len(data)}건")
            break

        else:
            print("\n잘못된 입력입니다. 다시 시도하세요.")
            input("아무 키나 누르면 계속...")


def main():
    """메인 함수"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "classified_data"
    output_path = project_root / "data" / "labeling" / "labeling_data.json"

    print("=" * 70)
    print("  VOC 라벨링 도구 초기화")
    print("=" * 70)

    # 기존 라벨링 데이터 확인
    if output_path.exists():
        print(f"\n기존 라벨링 데이터 발견: {output_path}")
        data = load_labeling_data(output_path)
        labeled = sum(1 for item in data if item.get("new_label") is not None)
        print(f"  총 {len(data)}건 중 {labeled}건 라벨링됨")

        choice = input("\n계속하시겠습니까? (y: 계속, n: 새로 시작): ").strip().lower()
        if choice != 'y':
            print("\n새 라벨링 데이터 생성 중...")
            data = extract_items_from_classified_data(data_dir, sample_size=750)
            print(f"  {len(data)}건 추출됨")
    else:
        print("\n라벨링 데이터 생성 중...")
        data = extract_items_from_classified_data(data_dir, sample_size=750)
        print(f"  {len(data)}건 추출됨")

    # 저장
    save_labeling_data(data, output_path)
    print(f"\n라벨링 데이터 저장: {output_path}")

    input("\nEnter를 누르면 라벨링을 시작합니다...")

    # 라벨링 세션 시작
    run_labeling_session(data, output_path)


if __name__ == "__main__":
    main()
