# =====================================================================
# 진짜 음성 전처리 스크립트 - 배경 소음 제거
#
# 실행 전:
#   pip install noisereduce soundfile tqdm
#
# 사용법:
#   python preprocess_denoise.py
#
# 결과:
#   INPUT_DIR  의 wav → OUTPUT_DIR 에 동일한 폴더 구조로 저장
#   이후 finetune.py 의 REAL_DATA_ROOT 를 OUTPUT_DIR 로 변경
# =====================================================================

import os
import numpy as np
import soundfile as sf
import noisereduce as nr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── 경로 설정 ─────────────────────────────────────────────────────
INPUT_DIR  = r'D:\User\Desktop\데이터셋\자유대화 음성(일반남녀)\Training'   # 원본
OUTPUT_DIR = r'D:\User\Desktop\데이터셋\자유대화 음성(일반남녀)\Training_denoised'  # 저장할 곳

# 사용할 스튜디오 폴더 (korean_dataset.py 의 USE_STUDIOS 와 동일하게)
USE_STUDIOS = ['[원천]3.스튜디오_2']  # 스튜디오_1 압축 풀리면 추가

# ── 설정 ──────────────────────────────────────────────────────────
SR          = 16000    # 목표 샘플레이트
NUM_WORKERS = 2        # 병렬 처리 수 (CPU 코어 수에 맞게 조절)
SKIP_EXIST  = True     # True: 이미 처리된 파일 건너뜀 (이어서 실행 가능)


def process_one(args):
    src_path, dst_path = args
    try:
        wav, sr = sf.read(src_path, dtype='float32')

        # 스테레오 → 모노
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        # 배경 소음 제거 (일정한 소음 특화)
        wav_denoised = nr.reduce_noise(y=wav, sr=sr, stationary=True).astype(np.float32)

        # 샘플레이트 변환
        if sr != SR:
            import resampy
            wav_denoised = resampy.resample(wav_denoised, sr, SR)

        # wav + npy 둘 다 저장
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        npy_path = dst_path.replace('.wav', '.npy')
        np.save(npy_path, wav_denoised)
        return dst_path, None

    except Exception as e:
        return src_path, str(e)


def main():
    # 처리할 파일 목록 수집
    file_pairs = []
    for studio in USE_STUDIOS:
        studio_in  = os.path.join(INPUT_DIR,  studio)
        studio_out = os.path.join(OUTPUT_DIR, studio)

        if not os.path.isdir(studio_in):
            print(f'[경고] 폴더 없음: {studio_in}')
            continue

        for dirpath, _, files in os.walk(studio_in):
            for fname in files:
                if not fname.lower().endswith('.wav'):
                    continue
                src = os.path.join(dirpath, fname)
                rel = os.path.relpath(src, INPUT_DIR)
                dst = os.path.join(OUTPUT_DIR, rel)

                if SKIP_EXIST and os.path.exists(dst):
                    continue
                file_pairs.append((src, dst))

    total = len(file_pairs)
    if total == 0:
        print('처리할 파일이 없습니다. (모두 완료됐거나 경로 확인 필요)')
        return

    print(f'처리할 파일: {total:,}개')
    print(f'저장 위치  : {OUTPUT_DIR}')
    print(f'병렬 처리  : {NUM_WORKERS}개\n')

    success, fail = 0, 0
    errors = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_one, pair): pair for pair in file_pairs}
        pbar = tqdm(as_completed(futures), total=total, unit='file', ncols=80)
        for future in pbar:
            path, err = future.result()
            if err is None:
                success += 1
            else:
                fail += 1
                errors.append((path, err))
            pbar.set_postfix(ok=success, fail=fail)

    print(f'\n완료: {success:,}개 성공 / {fail:,}개 실패')

    if errors:
        print('\n실패 목록 (처음 10개):')
        for path, err in errors[:10]:
            print(f'  {path}\n    → {err}')

    print(f'\n이제 finetune.py 의 REAL_DATA_ROOT 를 아래로 변경하세요:')
    print(f'  REAL_DATA_ROOT = r\'{OUTPUT_DIR}\'')


if __name__ == '__main__':
    main()