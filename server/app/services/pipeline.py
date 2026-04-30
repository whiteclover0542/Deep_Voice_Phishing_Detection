import concurrent.futures

from app.core.config import settings
from app.core.schemas import PipelineResult
from app.models.deepfake import DeepfakeModel
from app.models.nlu import NLUModel
from app.models.stt import STTModel
from app.services.audio_processor import convert_to_pcm
from app.services.explainer import explain

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class Pipeline:
    def __init__(self):
        self.stt = STTModel()
        self.deepfake = DeepfakeModel()
        self.nlu = NLUModel()

    def process(self, audio_chunk: bytes) -> dict:
        # FFmpeg으로 PCM 16kHz mono 변환
        pcm_chunk = convert_to_pcm(audio_chunk)

        # STT + Deepfake 병렬 처리
        stt_future = _executor.submit(self.stt.transcribe, pcm_chunk)
        deepfake_future = _executor.submit(self.deepfake.predict, pcm_chunk)

        text = stt_future.result()
        deepfake_result = deepfake_future.result()

        # NLU 위험 점수
        nlu_result = self.nlu.analyze(text)
        risk_score = nlu_result["risk_score"]
        warning_level = 3 if deepfake_result["is_fake"] else self._get_warning_level(risk_score)

        # LLM 설명 (경고 레벨 1 이상일 때만 호출해서 API 비용 절약)
        if warning_level >= 1 or deepfake_result["is_fake"]:
            llm_explanation = explain(
                transcript=text,
                risk_score=risk_score,
                is_fake_voice=deepfake_result["is_fake"],
                warning_level=warning_level,
            )
        else:
            llm_explanation = "정상적인 통화로 보입니다."

        return PipelineResult(
            text=text,
            risk_score=risk_score,
            warning_level=warning_level,
            is_fake_voice=deepfake_result["is_fake"],
            deepfake_confidence=deepfake_result["confidence"],
            explanation=llm_explanation,
        ).model_dump()

    def _get_warning_level(self, score: int) -> int:
        if score < settings.threshold_low:
            return 0
        elif score < settings.threshold_mid:
            return 1
        elif score < settings.threshold_high:
            return 2
        return 3
