"""KST 기반 날짜 유틸 회귀 테스트.

Cloud Run UTC, 로컬 KST 어느 환경에서 돌아도 동일한 결과여야 한다.
"""
import unittest
from datetime import datetime, timedelta, timezone

from src.utils.date_utils import KST, next_day, yesterday_kst


class YesterdayKstTest(unittest.TestCase):
    """daily pipeline target_date 계산 — '어제 KST'를 정확히 반환해야 한다."""

    def test_morning_kst_returns_previous_kst_day(self):
        # 5/8 08:09 KST = 5/7 23:09 UTC → yesterday_kst = 5/7
        now_utc = datetime(2026, 5, 7, 23, 9, tzinfo=timezone.utc)
        self.assertEqual(yesterday_kst(now_utc), "2026-05-07")

    def test_kst_midnight_returns_previous_kst_day(self):
        # 5/8 00:00 KST = 5/7 15:00 UTC → yesterday_kst = 5/7
        now_utc = datetime(2026, 5, 7, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(yesterday_kst(now_utc), "2026-05-07")

    def test_kst_late_night_returns_previous_kst_day(self):
        # 5/8 23:59 KST = 5/8 14:59 UTC → yesterday_kst = 5/7
        now_utc = datetime(2026, 5, 8, 14, 59, tzinfo=timezone.utc)
        self.assertEqual(yesterday_kst(now_utc), "2026-05-07")

    def test_one_minute_before_kst_midnight_still_today_yesterday(self):
        # 5/7 23:59 KST = 5/7 14:59 UTC → yesterday_kst = 5/6
        now_utc = datetime(2026, 5, 7, 14, 59, tzinfo=timezone.utc)
        self.assertEqual(yesterday_kst(now_utc), "2026-05-06")

    def test_kst_aware_input_also_works(self):
        # KST aware datetime 입력도 동일 결과
        now_kst = datetime(2026, 5, 8, 8, 9, tzinfo=KST)
        self.assertEqual(yesterday_kst(now_kst), "2026-05-07")

    def test_naive_input_rejected(self):
        with self.assertRaises(ValueError):
            yesterday_kst(datetime(2026, 5, 8, 8, 9))


class NextDayTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(next_day("2026-05-07"), "2026-05-08")

    def test_month_boundary(self):
        self.assertEqual(next_day("2026-04-30"), "2026-05-01")

    def test_year_boundary(self):
        self.assertEqual(next_day("2025-12-31"), "2026-01-01")


if __name__ == "__main__":
    unittest.main()
