"""v3 분류 체계 모듈

5분류 Topic(운영 피드백/서비스 피드백/콘텐츠 반응/투자 담론/기타) +
Sentiment(긍정/부정/중립) + Intent(4종) + Urgency(4단계)
"""
from src.classifier_v3.v3_topic_classifier import V3TopicClassifier

__all__ = ["V3TopicClassifier"]
