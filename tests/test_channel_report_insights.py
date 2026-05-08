import unittest

from src.reporter.channel_talk_insights import (
    build_channel_talk_report_context,
    render_workflow_intent_section,
)


class ChannelTalkReportInsightsTest(unittest.TestCase):
    def test_report_context_separates_voc_and_workflow_intent(self):
        rows = [
            {
                "topic": "결제·구독",
                "subtag": "환불",
                "workflow_buttons": [],
                "has_free_text": True,
                "interaction_type": "free_text",
            },
            {
                "topic": "",
                "subtag": "",
                "workflow_buttons": ["👆🏻1:1 문의 바로 연결하기"],
                "has_free_text": False,
                "interaction_type": "workflow_only",
            },
            {
                "topic": "결제·구독",
                "subtag": "구독해지",
                "workflow_buttons": ["구독해지 및 환불"],
                "has_free_text": True,
                "interaction_type": "mixed",
            },
        ]

        context = build_channel_talk_report_context(rows)

        self.assertEqual(context["total_chats"], 3)
        self.assertEqual(context["voc_chats"], 2)
        self.assertEqual(context["workflow_only_chats"], 1)
        self.assertEqual(context["interaction_type_counts"]["mixed"], 1)
        self.assertEqual(context["workflow_button_counts"]["구독해지 및 환불"], 1)
        self.assertEqual(context["topic_counts"]["결제·구독"], 2)

    def test_render_workflow_intent_section_mentions_button_only_not_voc(self):
        context = build_channel_talk_report_context([
            {
                "workflow_buttons": ["구독해지 및 환불"],
                "has_free_text": False,
                "interaction_type": "workflow_only",
            }
        ])

        section = render_workflow_intent_section(context)

        self.assertIn("문의 진입 의도", section)
        self.assertIn("구독해지 및 환불", section)
        self.assertIn("VOC 주제 카운트에는 포함하지 않습니다", section)


if __name__ == "__main__":
    unittest.main()
