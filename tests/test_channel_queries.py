import unittest
from datetime import datetime, timezone

from src.bigquery.channel_queries import ChannelQueryService


class ChannelQueryServiceTest(unittest.TestCase):
    def test_kst_to_unix_ms_is_independent_of_local_timezone(self):
        ts = ChannelQueryService._kst_to_unix_ms("2026-04-27")

        self.assertEqual(
            datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat(),
            "2026-04-26T15:00:00+00:00",
        )


if __name__ == "__main__":
    unittest.main()
