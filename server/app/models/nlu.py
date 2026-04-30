import random
from collections import deque

from app.core.config import settings


class NLUModel:
    """더미 NLU 모델 — KoELECTRA+LSTM 연동 전 placeholder"""

    def __init__(self):
        self.window: deque[str] = deque(maxlen=settings.window_size)

    def analyze(self, text: str) -> dict:
        self.window.append(text)
        context = " ".join(self.window)

        # 키워드 기반 더미 위험 점수
        danger_keywords = ["금융감독원", "송금", "계좌", "개인정보", "비밀번호", "검찰"]
        score = sum(10 for kw in danger_keywords if kw in context)
        score = min(score + random.randint(0, 10), 100)

        return {
            "text": text,
            "risk_score": score,
            "context_window": list(self.window),
        }
