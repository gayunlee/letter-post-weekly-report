"""Fine-tuned KcBERT 기반 분류기

기존 VectorContentClassifier와 동일한 인터페이스를 제공합니다.
"""
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .prepare_data import CATEGORY_TO_ID, ID_TO_CATEGORY


class FinetunedClassifier:
    """Fine-tuned KcBERT 기반 VOC 분류기"""

    def __init__(self, model_dir: str = None, device: str = None):
        """
        FinetunedClassifier 초기화

        Args:
            model_dir: 모델 디렉토리 경로 (None이면 기본 경로 사용)
            device: 사용할 디바이스 (None이면 자동 감지)
        """
        if model_dir is None:
            project_root = Path(__file__).parent.parent.parent
            model_dir = project_root / "models" / "kcbert_voc_classifier" / "final_model"

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"모델 디렉토리가 존재하지 않습니다: {self.model_dir}\n"
                "먼저 train.py를 실행하여 모델을 훈련하세요."
            )

        # 디바이스 설정
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 모델 & 토크나이저 로드
        print(f"모델 로드 중: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )
        self.model.to(self.device)
        self.model.eval()

        # 카테고리 매핑 로드
        config_path = self.model_dir / "category_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
                self.id_to_category = {int(k): v for k, v in config["id_to_category"].items()}
        else:
            self.id_to_category = ID_TO_CATEGORY

        print(f"  Device: {self.device}")
        print(f"  카테고리: {len(self.id_to_category)}개")

    def classify_content(self, content: str) -> Dict[str, Any]:
        """
        단일 콘텐츠 분류

        Args:
            content: 분류할 콘텐츠 텍스트

        Returns:
            {"category": str, "confidence": float, "method": str}
        """
        if not content or len(content.strip()) == 0:
            return {
                "category": "내용 없음",
                "confidence": 0.0,
                "method": "empty"
            }

        # 토크나이징
        inputs = self.tokenizer(
            content[:500],  # 최대 500자
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_id].item()

        category = self.id_to_category.get(predicted_id, "미분류")

        return {
            "category": category,
            "confidence": confidence,
            "method": "finetuned"
        }

    def classify_batch(
        self,
        contents: List[Dict[str, Any]],
        content_field: str = "message",
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        여러 콘텐츠를 일괄 분류

        Args:
            contents: 분류할 콘텐츠 리스트
            content_field: 콘텐츠 텍스트가 포함된 필드명
            batch_size: 배치 크기

        Returns:
            분류 결과가 추가된 콘텐츠 리스트
        """
        results = []

        # 배치 단위로 처리
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            texts = []

            for item in batch:
                text = item.get(content_field, "")
                if not text:
                    # 게시글의 경우 textBody/body 필드 확인
                    text = item.get("textBody") or item.get("body") or ""
                texts.append(text[:500] if text else "")

            # 빈 텍스트 처리
            valid_indices = [j for j, t in enumerate(texts) if t.strip()]

            if valid_indices:
                valid_texts = [texts[j] for j in valid_indices]

                # 배치 토크나이징
                inputs = self.tokenizer(
                    valid_texts,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 배치 추론
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    predicted_ids = torch.argmax(probs, dim=-1).tolist()
                    confidences = probs.max(dim=-1).values.tolist()

                # 결과 매핑
                valid_results = {}
                for idx, (pred_id, conf) in enumerate(zip(predicted_ids, confidences)):
                    valid_results[valid_indices[idx]] = {
                        "category": self.id_to_category.get(pred_id, "미분류"),
                        "confidence": conf,
                        "method": "finetuned"
                    }

            # 결과 조합
            for j, item in enumerate(batch):
                result = item.copy()
                if j in valid_results:
                    result["classification"] = valid_results[j]
                else:
                    result["classification"] = {
                        "category": "내용 없음",
                        "confidence": 0.0,
                        "method": "empty"
                    }
                results.append(result)

            # 진행 상황 출력
            processed = min(i + batch_size, len(contents))
            if processed % 500 == 0 or processed == len(contents):
                print(f"  진행: {processed}/{len(contents)} 완료")

        return results
