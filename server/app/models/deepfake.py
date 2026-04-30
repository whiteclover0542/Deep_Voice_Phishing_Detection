import random


class DeepfakeModel:
    """더미 Deepfake 탐지 모델 — 합성 음성 이진 분류 placeholder"""

    def predict(self, audio_chunk: bytes) -> dict:
        is_fake = random.random() < 0.2  # 20% 확률로 합성 음성
        return {
            "is_fake": is_fake,
            "confidence": round(random.uniform(0.7, 0.99), 3),
        }
