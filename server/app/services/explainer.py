from openai import OpenAI

from app.core.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

_SYSTEM_PROMPT = """당신은 보이스피싱 탐지 AI 어시스턴트입니다.
통화 내용과 분석 결과를 보고 사용자에게 2~3문장으로 간결하게 설명해주세요.
- 보이스피싱 의심 근거가 있으면 구체적으로 설명
- 딥보이스(합성 음성) 감지 여부도 언급
- 일반 통화면 "정상적인 통화로 보입니다"라고 안내
- 전문 용어 없이 쉬운 말로 설명
"""


def explain(
    transcript: str,
    risk_score: int,
    is_fake_voice: bool,
    warning_level: int,
) -> str:
    user_content = (
        f"통화 내용: {transcript}\n"
        f"보이스피싱 위험 점수: {risk_score}/100\n"
        f"합성 음성 여부: {'의심됨' if is_fake_voice else '정상'}\n"
        f"경고 단계: {warning_level}단계 (0=안전, 1=주의, 2=경고, 3=위험)"
    )

    response = _client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=200,
        temperature=0.3,
    )

    return response.choices[0].message.content or ""
