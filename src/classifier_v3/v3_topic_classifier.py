"""v3 Topic 분류기 — KcBERT 파인튜닝 모델 기반

v3 5분류 Topic(운영 피드백/서비스 피드백/콘텐츠 반응/투자 담론/기타) +
v2 Sentiment(긍정/부정/중립) 모델을 조합하여 2축 분류를 수행합니다.

Topic 모델: models/v3/topic/final_model/ (Colab에서 파인튜닝)
Sentiment 모델: models/two_axis/sentiment/final_model/ (v2 재사용)

사용법:
    classifier = V3TopicClassifier()
    result = classifier.classify_content("텍스트")
    results = classifier.classify_batch(items, content_field="message")
"""
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class V3TopicClassifier:
    """v3 Topic(5분류) + Sentiment(3분류) 파인튜닝 분류기"""

    def __init__(
        self,
        topic_model_dir: str = None,
        sentiment_model_dir: str = None,
        device: str = None,
    ):
        project_root = Path(__file__).parent.parent.parent

        if topic_model_dir is None:
            topic_model_dir = project_root / "models" / "v3" / "topic" / "final_model"
        if sentiment_model_dir is None:
            sentiment_model_dir = project_root / "models" / "two_axis" / "sentiment" / "final_model"

        self.topic_dir = Path(topic_model_dir)
        self.sentiment_dir = Path(sentiment_model_dir)

        for d, name in [(self.topic_dir, "v3 Topic"), (self.sentiment_dir, "Sentiment")]:
            if not d.exists():
                raise FileNotFoundError(
                    f"{name} 모델 디렉토리가 없습니다: {d}\n"
                    "Colab에서 파인튜닝 후 make pull로 모델을 다운로드하세요."
                )

        # 디바이스 (이 서버는 GPU 없음 → CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Topic 모델 (v3 5분류)
        print(f"v3 Topic 모델 로드: {self.topic_dir}")
        self.topic_tokenizer = AutoTokenizer.from_pretrained(str(self.topic_dir))
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(str(self.topic_dir))
        self.topic_model.to(self.device).eval()

        cfg = self.topic_dir / "category_config.json"
        with open(cfg, encoding="utf-8") as f:
            self.id_to_topic = {int(k): v for k, v in json.load(f)["id_to_category"].items()}

        # Sentiment 모델 (v2 재사용)
        print(f"Sentiment 모델 로드: {self.sentiment_dir}")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(str(self.sentiment_dir))
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(str(self.sentiment_dir))
        self.sentiment_model.to(self.device).eval()

        cfg = self.sentiment_dir / "category_config.json"
        with open(cfg, encoding="utf-8") as f:
            self.id_to_sentiment = {int(k): v for k, v in json.load(f)["id_to_category"].items()}

        print(f"  Device: {self.device}")
        print(f"  Topic (v3): {list(self.id_to_topic.values())}")
        print(f"  Sentiment: {list(self.id_to_sentiment.values())}")

    def _predict_batch(self, tokenizer, model, id_map, texts: List[str]) -> List[Dict[str, Any]]:
        """배치 추론 공통 로직"""
        if not texts:
            return []

        inputs = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).tolist()
            confs = probs.max(dim=-1).values.tolist()

        return [
            {"label": id_map.get(pid, "미분류"), "confidence": conf}
            for pid, conf in zip(pred_ids, confs)
        ]

    def classify_content(self, content: str) -> Dict[str, Any]:
        """단일 콘텐츠 분류"""
        if not content or not content.strip():
            return {
                "topic": "미분류", "topic_confidence": 0.0,
                "sentiment": "미분류", "sentiment_confidence": 0.0,
                "method": "empty",
            }

        text = content[:500]
        topic_res = self._predict_batch(
            self.topic_tokenizer, self.topic_model, self.id_to_topic, [text]
        )[0]
        sent_res = self._predict_batch(
            self.sentiment_tokenizer, self.sentiment_model, self.id_to_sentiment, [text]
        )[0]

        return {
            "topic": topic_res["label"],
            "topic_confidence": topic_res["confidence"],
            "sentiment": sent_res["label"],
            "sentiment_confidence": sent_res["confidence"],
            "method": "finetuned_v3",
        }

    def classify_batch(
        self,
        contents: List[Dict[str, Any]],
        content_field: str = "message",
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """여러 콘텐츠를 v3 Topic + Sentiment로 일괄 분류"""
        results = [None] * len(contents)

        # 유효 텍스트 분리
        valid_indices = []
        valid_texts = []
        for i, item in enumerate(contents):
            text = item.get(content_field, "")
            if not text:
                text = item.get("textBody") or item.get("body") or ""
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text[:500])
            else:
                result = item.copy()
                result["classification"] = {
                    "topic": "미분류", "topic_confidence": 0.0,
                    "sentiment": "미분류", "sentiment_confidence": 0.0,
                    "method": "empty",
                }
                results[i] = result

        # 배치 처리
        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            batch_idx = valid_indices[batch_start:batch_end]

            topic_preds = self._predict_batch(
                self.topic_tokenizer, self.topic_model, self.id_to_topic, batch_texts
            )
            sent_preds = self._predict_batch(
                self.sentiment_tokenizer, self.sentiment_model, self.id_to_sentiment, batch_texts
            )

            for j, idx in enumerate(batch_idx):
                result = contents[idx].copy()
                result["classification"] = {
                    "topic": topic_preds[j]["label"],
                    "topic_confidence": topic_preds[j]["confidence"],
                    "sentiment": sent_preds[j]["label"],
                    "sentiment_confidence": sent_preds[j]["confidence"],
                    "method": "finetuned_v3",
                }
                results[idx] = result

            processed = min(batch_end, len(valid_texts))
            if processed % 500 == 0 or processed == len(valid_texts):
                print(f"  진행: {processed}/{len(valid_texts)} 완료")

        return results
