"""
wav → npy 미리 변환 스크립트
사용: python precache_npy.py
완료 후 finetune.py 재실행 시 wav 읽기 없이 npy로 로드되어 빨라짐.
"""
import os
import sys
import numpy as np
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

SR = 16000

ROOTS = [
    r'D:\User\Desktop\데이터셋\자유대화 음성(일반남녀)\Training_denoised',
    r'D:\User\Desktop\데이터셋\자유대화 음성(일반남녀)\Validation',
    r'D:\User\Desktop\데이터셋\015.감성 및 발화 스타일별 음성합성 데이터\01.데이터\1.Training\원천데이터',
    r'D:\User\Desktop\데이터셋\015.감성 및 발화 스타일별 음성합성 데이터\01.데이터\2.Validation\원천데이터',
    r'D:\User\Desktop\데이터셋\007.저음질 전화망 음성인식 데이터\01.데이터\1.Training\원천데이터_230316\TS_D01',
]

def collect_wavs(roots):
    paths = []
    for root in roots:
        if not os.path.isdir(root):
            print(f'  [스킵] 폴더 없음: {root}', flush=True)
            continue
        print(f'  스캔: {root}', flush=True)
        count_before = len(paths)
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith('.wav'):
                    paths.append(os.path.join(dirpath, f))
            cur = len(paths) - count_before
            if cur % 10000 == 0 and cur > 0:
                print(f'    ... {cur:,}개', flush=True)
        print(f'  → {len(paths) - count_before:,}개 발견', flush=True)
    return paths

def convert_one(wav_path):
    npy_path = wav_path.replace('.wav', '.npy')
    if os.path.exists(npy_path):
        return 'skip'
    try:
        wav, sr = sf.read(wav_path, dtype='float32')
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != SR:
            try:
                import resampy
                wav = resampy.resample(wav, sr, SR)
            except ImportError:
                pass
        np.save(npy_path, wav)
        return 'ok'
    except Exception as e:
        return f'err:{e}'

if __name__ == '__main__':
    print('[1/2] wav 파일 목록 수집 중...', flush=True)
    paths = collect_wavs(ROOTS)
    print(f'\n[2/2] 총 {len(paths):,}개 변환 시작 (workers=8)\n', flush=True)

    CHUNK = 2000
    done, skip, err = 0, 0, 0
    total = len(paths)

    with ProcessPoolExecutor(max_workers=8) as ex:
        for chunk_start in range(0, total, CHUNK):
            chunk = paths[chunk_start:chunk_start + CHUNK]
            futs = {ex.submit(convert_one, p): p for p in chunk}
            for fut in as_completed(futs):
                r = fut.result()
                if r == 'ok':     done += 1
                elif r == 'skip': skip += 1
                else:             err += 1
            processed = chunk_start + len(chunk)
            print(f'  {processed:,}/{total:,}  변환={done:,}  스킵={skip:,}  오류={err:,}', flush=True)

    print(f'\n완료: 변환={done:,}  스킵={skip:,}  오류={err:,}')
