import asyncio
import os
import edge_tts
from pydub import AudioSegment

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tts_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 한국어 음성 목록
# ko-KR-SunHiNeural  : 여성
# ko-KR-InJoonNeural : 남성
# ko-KR-HyunsuNeural : 남성
VOICES = [
    'ko-KR-SunHiNeural',
    'ko-KR-InJoonNeural',
    'ko-KR-HyunsuNeural',
]

TEXTS = [
    ("안녕하세요 고객님 저는 금융감독원 직원입니다 지금 고객님 계좌에서 의심스러운 거래가 감지되었습니다", "vp_01"),
    ("고객님 지금 즉시 계좌를 정지하지 않으면 큰 피해를 입으실 수 있습니다", "vp_02"),
    ("본인 확인을 위해 주민등록번호와 계좌 비밀번호를 알려주세요", "vp_03"),
    ("검찰청입니다 고객님 명의로 대포통장이 개설되어 수사가 진행 중입니다", "vp_04"),
    ("지금 바로 안전 계좌로 돈을 이체해 주셔야 합니다", "vp_05"),
    ("고객님의 개인정보가 유출되어 즉시 조치가 필요합니다", "vp_06"),
    ("대출 승인이 났습니다 선입금 후 대출금을 지급해 드립니다", "vp_07"),
    ("경찰청 사이버수사대입니다 고객님 계좌가 범죄에 연루되었습니다", "vp_08"),
]


async def make_one(text: str, voice: str, wav_path: str):
    if os.path.exists(wav_path):
        print(f'  건너뜀 (이미 존재): {os.path.basename(wav_path)}')
        return

    mp3_path = wav_path.replace('.wav', '.mp3')
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(mp3_path)

        AudioSegment.from_mp3(mp3_path).set_frame_rate(16000).set_channels(1).export(wav_path, format='wav')
        print(f'  저장: {os.path.basename(wav_path)}')
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


async def main():
    tasks = []
    for text, name in TEXTS:
        for voice in VOICES:
            voice_tag = voice.split('-')[2].replace('Neural', '').lower()  # sunhi / injoon / hyunsu
            wav_path  = os.path.join(OUTPUT_DIR, f'{name}_{voice_tag}.wav')
            tasks.append(make_one(text, voice, wav_path))

    print(f'총 {len(tasks)}개 생성 시작 ({len(TEXTS)}문장 × {len(VOICES)}목소리)\n')
    await asyncio.gather(*tasks)
    print(f'\n완료. 저장 위치: {OUTPUT_DIR}')


if __name__ == '__main__':
    asyncio.run(main())
