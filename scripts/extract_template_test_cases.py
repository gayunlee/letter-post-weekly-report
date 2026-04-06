"""16종 답변 템플릿별 유저-매니저 대화 쌍 추출 스크립트

BigQuery channel_io.messages에서 매니저 응답 키워드로 필터링하여
템플릿별 3~5건의 테스트 케이스를 추출합니다.
"""
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bigquery.client import BigQueryClient
from src.bigquery.channel_queries import ChannelQueryService


# 16종 템플릿 정의: {type_name: [keywords]}
TEMPLATE_KEYWORDS = {
    "환불_확정": ["카드취소", "카드 취소 처리", "카드 취소가 완료"],
    "환불_접수": ["환불 접수", "접수 완료", "접수되었습니다"],
    "전액환불": ["전액 환불", "전액환불"],
    "부분환불": ["1/3 차감", "수수료", "부분 환불", "부분환불", "차감"],
    "구독해지": ["정기결제 구독해지", "해지 방법", "구독관리", "구독 해지", "자동결제 해지"],
    "카드변경": ["카드 변경", "결제카드 변경", "결제변경", "카드변경"],
    "상품변경": ["상품 변경", "업그레이드", "상품변경"],
    "상품링크": ["us-insight.com", "신청 링크", "상품 링크", "구매 링크"],
    "앱설치": ["앱 설치", "앱스토어", "플레이스토어", "App Store", "Play Store"],
    "회원가입": ["회원가입", "가입 방법", "회원 가입"],
    "로그인변경": ["비밀번호 재설정", "비밀번호 찾기", "로그인 방법"],
    "종료인사": ["추가적으로 문의", "상담 종료", "추가 문의"],
    "본인확인": ["성함", "연락처", "확인 부탁", "본인확인"],
    "기술오류": ["기종", "배속", "재설치", "재부팅", "버전"],
    "수강안내": ["수업", "강의", "녹화본", "VOD", "수강"],
    "플랫폼혼동": ["어스캠퍼스", "어스플러스"],
}

TARGET_PER_TEMPLATE = 5


def classify_manager_message(text: str) -> list[str]:
    """매니저 메시지가 어떤 템플릿에 해당하는지 반환"""
    if not text:
        return []
    matched = []
    for ttype, keywords in TEMPLATE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched.append(ttype)
                break
    return matched


def extract_test_cases():
    client = BigQueryClient()
    cqs = ChannelQueryService(client)

    # 최근 3개월 데이터 조회 (실제 데이터 범위: ~2025-12-05)
    print("BigQuery에서 메시지 조회 중...")
    messages = cqs.get_weekly_messages("2025-09-01", "2025-12-06")
    print(f"총 {len(messages)}건 메시지 조회됨")

    # chatId별 그룹핑
    chats = defaultdict(lambda: {"user": [], "manager": [], "bot": []})
    for msg in messages:
        chat_id = msg.get("chatId")
        person_type = msg.get("personType", "")
        text = msg.get("plainText", "") or ""
        created_at = msg.get("createdAt", 0)

        if person_type == "user":
            chats[chat_id]["user"].append({"text": text, "ts": created_at})
        elif person_type == "manager":
            chats[chat_id]["manager"].append({"text": text, "ts": created_at})
        elif person_type == "bot":
            chats[chat_id]["bot"].append({"text": text, "ts": created_at})

    print(f"총 {len(chats)}개 대화 그룹핑됨")

    # 매니저 응답이 있는 대화만 필터
    manager_chats = {
        cid: data for cid, data in chats.items()
        if data["manager"] and data["user"]
    }
    print(f"유저+매니저 응답이 모두 있는 대화: {len(manager_chats)}개")

    # 템플릿별 분류
    template_cases = defaultdict(list)

    for chat_id, data in manager_chats.items():
        # 매니저 메시지 전체에서 템플릿 매칭
        matched_templates = set()
        for msg in data["manager"]:
            for ttype in classify_manager_message(msg["text"]):
                matched_templates.add(ttype)

        for ttype in matched_templates:
            if len(template_cases[ttype]) >= TARGET_PER_TEMPLATE:
                continue

            # 워크플로우 버튼 (봇 메시지에서 추출)
            workflow_buttons = []
            for bot_msg in data["bot"]:
                text = bot_msg["text"]
                if text and len(text) < 50:  # 짧은 봇 메시지 = 버튼 클릭
                    workflow_buttons.append(text.strip())

            # 시간순 정렬
            user_msgs = [m["text"] for m in sorted(data["user"], key=lambda x: x["ts"]) if m["text"]]
            mgr_msgs = [m["text"] for m in sorted(data["manager"], key=lambda x: x["ts"]) if m["text"]]

            if not user_msgs or not mgr_msgs:
                continue

            template_cases[ttype].append({
                "template_type": ttype,
                "chatId": chat_id,
                "user_messages": user_msgs,
                "manager_messages": mgr_msgs,
                "workflow_buttons": workflow_buttons,
                "notes": "",
            })

    # 결과 출력
    all_cases = []
    print("\n=== 템플릿별 추출 결과 ===")
    for ttype in TEMPLATE_KEYWORDS:
        cases = template_cases.get(ttype, [])
        print(f"  {ttype}: {len(cases)}건")
        all_cases.extend(cases)

    # JSON 저장
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "cs-agent-demo", "data", "template_test_cases.json"
    )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(all_cases)}건 저장됨 → {output_path}")


if __name__ == "__main__":
    extract_test_cases()
