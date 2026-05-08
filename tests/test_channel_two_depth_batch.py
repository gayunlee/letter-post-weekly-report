import unittest

from src.classifier_v4.two_depth_classifier import TwoDepthClassifier


class ChannelTwoDepthBatchTest(unittest.TestCase):
    def test_classify_batch_preserves_workflow_only_without_classifying(self):
        classifier = TwoDepthClassifier.__new__(TwoDepthClassifier)
        calls = []

        def fake_classify(text):
            calls.append(text)
            return {
                "topic": "결제·구독",
                "confidence": 0.91,
                "subtag": "환불",
                "source": "kcelectra",
            }

        classifier.classify = fake_classify
        items = [
            {
                "chatId": "workflow-only",
                "text": "👆🏻1:1 문의 바로 연결하기",
                "classifiable_text": "",
                "has_free_text": False,
                "interaction_type": "workflow_only",
            },
            {
                "chatId": "mixed",
                "text": "👆🏻1:1 문의 바로 연결하기\n환불신청 부탁드려요",
                "classifiable_text": "환불신청 부탁드려요",
                "has_free_text": True,
                "interaction_type": "mixed",
            },
        ]

        results = classifier.classify_batch(items)

        self.assertEqual([item["chatId"] for item in results], ["workflow-only", "mixed"])
        self.assertNotIn("classification", results[0])
        self.assertEqual(results[1]["classification"]["subtag"], "환불")
        self.assertEqual(calls, ["환불신청 부탁드려요"])


if __name__ == "__main__":
    unittest.main()
