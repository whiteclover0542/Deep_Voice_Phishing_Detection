from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "Voice Phishing Detector"
    debug: bool = True

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Audio settings
    sample_rate: int = 16000       # Hz
    bytes_per_sample: int = 2      # PCM 16-bit
    chunk_duration_ms: int = 500
    window_size: int = 5

    # Risk score thresholds (0~100)
    threshold_low: int = 25
    threshold_mid: int = 50
    threshold_high: int = 75

    @property
    def chunk_bytes(self) -> int:
        return int(self.sample_rate * self.bytes_per_sample * self.chunk_duration_ms / 1000)


settings = Settings()
