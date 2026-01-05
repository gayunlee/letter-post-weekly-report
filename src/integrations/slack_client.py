"""Slack 연동 모듈"""
import os
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class SlackNotifier:
    """Slack으로 리포트 알림을 전송하는 클라이언트"""

    def __init__(
        self,
        bot_token: str = None,
        channel_id: str = None
    ):
        """
        SlackNotifier 초기화

        Args:
            bot_token: Slack Bot Token
            channel_id: 메시지를 보낼 채널 ID
        """
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.channel_id = channel_id or os.getenv("SLACK_CHANNEL_ID")

        if not self.bot_token:
            raise ValueError("SLACK_BOT_TOKEN이 설정되지 않았습니다.")
        if not self.channel_id:
            raise ValueError("SLACK_CHANNEL_ID가 설정되지 않았습니다.")

        self.base_url = "https://slack.com/api"

    def send_report_notification(
        self,
        week_label: str,
        start_date: str,
        end_date: str,
        notion_url: str
    ) -> Dict[str, Any]:
        """
        리포트 알림 전송 (메인 메시지 + 댓글)

        Args:
            week_label: 주차 라벨 (예: "12월 넷째 주")
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            notion_url: 노션 페이지 URL

        Returns:
            {"ok": bool, "message_ts": str, "thread_ts": str}
        """
        # 날짜 포맷 변환 (YYYY-MM-DD -> YYYY.MM.DD)
        start_formatted = start_date.replace('-', '.')
        # end_date는 exclusive이므로 하루 전 날짜 계산
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        from datetime import timedelta
        actual_end = (end_dt - timedelta(days=1)).strftime('%Y.%m.%d')
        # MM.DD 형식으로 변환
        start_short = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m.%d')
        end_short = (end_dt - timedelta(days=1)).strftime('%m.%d')

        # 메인 메시지
        main_message = f"[{week_label}]오피셜클럽 별 편지·게시글 통합 분석 공유"

        # 댓글 내용
        thread_message = (
            f"오피셜클럽 별 편지·게시글 통합 분석 ({start_formatted} ~ {actual_end.split('.')[-1]})을 "
            f"작성하여 공유드립니다.\n"
            f":pushpin: 이번 주 이용자 반응 리포트 (편지 + 게시글 기준, {start_formatted} ~ {actual_end.split('.')[-1]})\n"
            f"{notion_url}"
        )

        # 1. 메인 메시지 전송
        main_response = self._send_message(main_message)

        if not main_response.get("ok"):
            return {
                "ok": False,
                "error": main_response.get("error", "메인 메시지 전송 실패")
            }

        message_ts = main_response.get("ts")

        # 2. 댓글 전송 (thread)
        thread_response = self._send_message(thread_message, thread_ts=message_ts)

        return {
            "ok": thread_response.get("ok", False),
            "message_ts": message_ts,
            "thread_ts": thread_response.get("ts")
        }

    def _send_message(
        self,
        text: str,
        thread_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Slack 메시지 전송

        Args:
            text: 메시지 내용
            thread_ts: 스레드 타임스탬프 (댓글로 보낼 경우)

        Returns:
            Slack API 응답
        """
        url = f"{self.base_url}/chat.postMessage"

        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "channel": self.channel_id,
            "text": text
        }

        if thread_ts:
            payload["thread_ts"] = thread_ts

        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    @staticmethod
    def get_week_label(date_str: str) -> str:
        """
        날짜를 한글 주차 라벨로 변환
        예: 2025-12-22 -> "12월 넷째 주"

        Args:
            date_str: YYYY-MM-DD 형식의 날짜

        Returns:
            "12월 넷째 주" 형식의 문자열
        """
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month

        # 해당 월의 몇 번째 주인지 계산
        first_day = date.replace(day=1)
        week_of_month = (date.day + first_day.weekday()) // 7 + 1

        week_names = ["첫째", "둘째", "셋째", "넷째", "다섯째"]
        week_name = week_names[min(week_of_month - 1, 4)]

        return f"{month}월 {week_name} 주"
