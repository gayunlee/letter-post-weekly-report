import unittest

from src.bigquery.writer import BigQueryWriter, CHANNEL_TALK_SCHEMA


class _QueryJob:
    def result(self):
        return None


class _Table:
    def __init__(self, schema):
        self.schema = schema


class _FakeBigQueryClient:
    project = "test-project"

    def __init__(self):
        self.created_table_schema = []
        self.updated_schema = None
        self.inserted_rows = None
        self.ignore_unknown_values = None

    def create_table(self, table, exists_ok=False):
        self.created_table_schema = table.schema
        return _Table(schema=table.schema[:3])

    def update_table(self, table, fields):
        self.updated_schema = table.schema
        return table

    def query(self, query):
        return _QueryJob()

    def insert_rows_json(self, table_id, rows, ignore_unknown_values=False):
        self.inserted_rows = rows
        self.ignore_unknown_values = ignore_unknown_values
        return []


class ChannelTalkWriterTest(unittest.TestCase):
    def test_ensure_table_adds_missing_nullable_columns(self):
        client = _FakeBigQueryClient()
        writer = BigQueryWriter(client)

        table_id = writer._ensure_table("channel_talk", CHANNEL_TALK_SCHEMA)

        self.assertEqual(table_id, "test-project.voc_labelled.channel_talk")
        self.assertIsNotNone(client.updated_schema)
        self.assertIn("compound_reason", {field.name for field in client.updated_schema})
        self.assertIn("tags", {field.name for field in client.updated_schema})

    def test_write_channel_talk_persists_compound_fields(self):
        client = _FakeBigQueryClient()
        writer = BigQueryWriter(client)
        items = [
            {
                "chatId": "chat-1",
                "text": "강의가 안 열려서 환불하고 싶어요",
                "first_message_at": 1710000000000,
                "route": "manager_resolved",
                "workflow_buttons": ["수강 및 상품문의"],
                "has_free_text": True,
                "interaction_type": "mixed",
                "classification": {
                    "topic": "결제·구독",
                    "confidence": 0.91,
                    "subtag": "환불",
                    "is_compound": True,
                    "compound_reason": "수강 접근 오류 이후 환불을 요구함",
                    "summary": "강의 접근 오류로 환불을 요청한 문의",
                    "tags": ["환불", "수강오류"],
                    "source": "kcelectra",
                },
            }
        ]

        saved_count = writer.write_channel_talk(items, "2026-05-03")

        self.assertEqual(saved_count, 1)
        self.assertTrue(client.ignore_unknown_values)
        row = client.inserted_rows[0]
        self.assertTrue(row["is_compound"])
        self.assertEqual(row["compound_reason"], "수강 접근 오류 이후 환불을 요구함")
        self.assertEqual(row["summary"], "강의 접근 오류로 환불을 요청한 문의")
        self.assertEqual(row["tags"], ["환불", "수강오류"])
        self.assertEqual(row["workflow_buttons"], ["수강 및 상품문의"])
        self.assertTrue(row["has_free_text"])
        self.assertEqual(row["interaction_type"], "mixed")


if __name__ == "__main__":
    unittest.main()
