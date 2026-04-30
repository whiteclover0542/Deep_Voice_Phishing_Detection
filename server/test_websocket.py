import asyncio
import json
import websockets

async def test():
    uri = "ws://localhost:8000/ws/audio"

    # 더미 오디오 데이터 (16000 bytes = 500ms PCM)
    dummy_audio = bytes(16000)

    async with websockets.connect(uri) as ws:
        print("연결됨")

        # 5번 청크 전송
        for i in range(5):
            await ws.send(dummy_audio)
            response = await ws.recv()
            result = json.loads(response)
            print(f"\n[청크 {i+1}]")
            print(f"  텍스트     : {result['text']}")
            print(f"  위험 점수  : {result['risk_score']}/100")
            print(f"  경고 단계  : {result['warning_level']}단계")
            print(f"  딥보이스   : {'의심' if result['is_fake_voice'] else '정상'}")
            print(f"  LLM 설명   : {result['explanation']}")

asyncio.run(test())
