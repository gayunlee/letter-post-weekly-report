import unittest

from src.classifier_v4.subtag_detail import (
    empty_subtag_detail,
    normalize_subtag_detail,
)


class SubtagDetailTest(unittest.TestCase):
    def test_normalizes_valid_compound_detail(self):
        detail = normalize_subtag_detail(
            {
                "subtag": "환불",
                "is_compound": True,
                "compound_reason": "수강 접근 오류 이후 환불을 요구함",
                "summary": "강의를 볼 수 없어 환불을 요청한 문의",
                "tags": ["환불", "수강오류", "접근불가", "추가태그", "초과태그"],
            },
            "결제·구독",
        )

        self.assertEqual(detail["subtag"], "환불")
        self.assertTrue(detail["is_compound"])
        self.assertEqual(detail["compound_reason"], "수강 접근 오류 이후 환불을 요구함")
        self.assertEqual(detail["summary"], "강의를 볼 수 없어 환불을 요청한 문의")
        self.assertEqual(detail["tags"], ["환불", "수강오류", "접근불가", "추가태그"])

    def test_invalid_subtag_falls_back_to_etc(self):
        detail = normalize_subtag_detail(
            {
                "subtag": "없는태그",
                "is_compound": False,
                "compound_reason": "반복되면 안 되는 설명",
                "summary": "요약",
                "tags": "문자열은 버림",
            },
            "결제·구독",
        )

        self.assertEqual(detail["subtag"], "기타")
        self.assertFalse(detail["is_compound"])
        self.assertIsNone(detail["compound_reason"])
        self.assertEqual(detail["tags"], [])

    def test_empty_detail_keeps_existing_contract_shape(self):
        self.assertEqual(
            empty_subtag_detail(),
            {
                "subtag": "기타",
                "is_compound": False,
                "compound_reason": None,
                "summary": "",
                "tags": [],
            },
        )


if __name__ == "__main__":
    unittest.main()
