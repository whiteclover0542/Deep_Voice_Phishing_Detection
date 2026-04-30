from pydantic import BaseModel


class PipelineResult(BaseModel):
    text: str
    risk_score: int           # 0~100
    warning_level: int        # 0: 안전 / 1: 주의 / 2: 경고 / 3: 위험
    is_fake_voice: bool
    deepfake_confidence: float
    explanation: str          # LLM 자연어 설명
