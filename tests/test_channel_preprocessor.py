import unittest

from src.bigquery.channel_preprocessor import build_chat_items, detect_route, strip_workflow_only_lines


class ChannelPreprocessorTest(unittest.TestCase):
    def test_closed_with_manager_is_manager_resolved(self):
        self.assertEqual(detect_route(1, 2, "closed"), "manager_resolved")

    def test_closed_without_manager_is_bot_resolved(self):
        self.assertEqual(detect_route(0, 2, "closed"), "bot_resolved")

    def test_opened_state_is_opened_not_abandoned(self):
        self.assertEqual(detect_route(1, 2, "opened"), "opened")

    def test_missing_state_is_opened(self):
        self.assertEqual(detect_route(0, 2, None), "opened")

    def test_strip_workflow_only_lines_removes_button_rows(self):
        self.assertEqual(strip_workflow_only_lines("👆🏻1:1 문의 바로 연결하기"), "")
        self.assertEqual(
            strip_workflow_only_lines("👆🏻1:1 문의 바로 연결하기\n환불신청 부탁드려요"),
            "환불신청 부탁드려요",
        )

    def test_build_chat_items_preserves_workflow_button_only_chat(self):
        messages = [
            {
                "chatId": "chat-1",
                "personType": "user",
                "plainText": "👆🏻1:1 문의 바로 연결하기",
                "createdAt": "2026-05-01T00:00:00Z",
            },
            {
                "chatId": "chat-2",
                "personType": "user",
                "plainText": "환불신청 부탁드려요",
                "createdAt": "2026-05-01T00:01:00Z",
            },
        ]

        items = build_chat_items(messages, min_user_chars=10, chat_states={})

        self.assertEqual([item["chatId"] for item in items], ["chat-1", "chat-2"])
        self.assertEqual(items[0]["interaction_type"], "workflow_only")
        self.assertFalse(items[0]["has_free_text"])
        self.assertEqual(items[1]["interaction_type"], "free_text")
        self.assertTrue(items[1]["has_free_text"])

    def test_build_chat_items_marks_mixed_workflow_and_free_text(self):
        messages = [
            {
                "chatId": "chat-1",
                "personType": "user",
                "plainText": "👆🏻1:1 문의 바로 연결하기",
                "createdAt": "2026-05-01T00:00:00Z",
            },
            {
                "chatId": "chat-1",
                "personType": "user",
                "plainText": "환불신청 부탁드려요",
                "createdAt": "2026-05-01T00:01:00Z",
            },
        ]

        items = build_chat_items(messages, min_user_chars=10, chat_states={})

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["interaction_type"], "mixed")
        self.assertTrue(items[0]["has_free_text"])


if __name__ == "__main__":
    unittest.main()
