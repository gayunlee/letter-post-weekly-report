import unittest

from scripts.backfill_channel_compound_reason import fetch_rows, parse_json_object


class FakeBigQuery:
    project_id = "test-project"

    def __init__(self):
        self.query = None

    def execute_query(self, query):
        self.query = query
        return []


class ChannelCompoundBackfillTest(unittest.TestCase):
    def test_parse_json_object_from_plain_json(self):
        self.assertEqual(
            parse_json_object('{"is_compound": true, "compound_reason": "환불 맥락"}'),
            {"is_compound": True, "compound_reason": "환불 맥락"},
        )

    def test_parse_json_object_from_text_wrapped_response(self):
        self.assertEqual(
            parse_json_object('응답입니다\n{"summary": "요약", "tags": ["환불"]}\n끝'),
            {"summary": "요약", "tags": ["환불"]},
        )

    def test_fetch_rows_only_selects_missing_summary_and_classifiable_text(self):
        bq = FakeBigQuery()

        fetch_rows(bq, "2026-04-27", "2026-05-04", 20)

        self.assertIn('AND (summary IS NULL OR summary = "")', bq.query)
        self.assertIn("REGEXP_REPLACE", bq.query)
        self.assertIn("👆🏻.*", bq.query)


if __name__ == "__main__":
    unittest.main()
