# =====================================================================
# KoreanDeepfakeDataset - 최종버전
#
# 진짜 (label=0): 109번 자유대화 음성(일반남녀) - 스튜디오 녹음
#   구조: real_root/[원천]3.스튜디오_1/화자폴더/*.wav
#          real_root/[원천]3.스튜디오_2/화자폴더/*.wav
#
# 가짜 (label=1): 466번 감성 발화 스타일별 음성합성 TTS
#   구조: fake_root/TS1/1.기쁨/*.wav
#          fake_root/TS1/2.슬픔/*.wav  ...
#          fake_root/TS2/...
#
# [finetune.py 경로 설정]
# REAL_DATA_ROOT = r'...\자유대화 음성(일반남녀)\Training'
# FAKE_DATA_ROOT = r'...\015.감성 및 발화 스타일별 음성합성 데이터\01.데이터\1.Training\원천데이터'
# =====================================================================

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf

SR      = 16000
MAX_LEN = SR * 4   # 4초
SEED    = 42


class KoreanDeepfakeDataset(Dataset):
    def __init__(
        self,
        real_root,        # 109번: Training 폴더 (스튜디오_1, _2 상위)
        fake_root,        # 466번: 원천데이터 폴더 (TS1, TS2... 상위)
        split='train',
        noise_aug=True,   # 진짜 음성 노이즈 증강 (품질 차이 보정)
        real_limit=None,  # 진짜 최대 샘플 수 (None=전부)
        fake_limit=None,  # 가짜 최대 샘플 수 (None=전부)
    ):
        self.noise_aug = noise_aug
        self.items = []

        # ── 진짜 음성: 사용할 스튜디오 폴더 지정 ─────────────────────
        # 압축 완료 후 리스트에 추가하면 됨
        # 예: USE_STUDIOS = ['[원천]3.스튜디오_2', '[원천]3.스튜디오_1']
        USE_STUDIOS = ['[원천]3.스튜디오_2']

        real_items = []
        studio_dirs = [
            d for d in USE_STUDIOS
            if os.path.isdir(os.path.join(real_root, d))
        ]

        if not studio_dirs:
            print(f'  [경고] 스튜디오 폴더를 찾을 수 없음: {real_root}')
        else:
            print(f'  [진짜] 스튜디오 폴더: {studio_dirs}')
            for studio_dir in studio_dirs:
                studio_path = os.path.join(real_root, studio_dir)
                for dirpath, _, files in os.walk(studio_path):
                    for fname in files:
                        if fname.lower().endswith('.wav'):
                            real_items.append(
                                (os.path.join(dirpath, fname), 0)
                            )

        # train/dev 분할 (8:2, 실행마다 랜덤)
        random.shuffle(real_items)
        cut = int(len(real_items) * 0.8)
        real_items = real_items[:cut] if split == 'train' else real_items[cut:]

        if real_limit is not None:
            real_items = real_items[:real_limit]

        self.items.extend(real_items)


        # ── 가짜 음성: TS1/1.기쁨/*.wav, TS2/... ─────────────────────
        fake_items = []
        for dirpath, _, files in os.walk(fake_root):
            for fname in files:
                if fname.lower().endswith('.wav'):
                    fake_items.append(
                        (os.path.join(dirpath, fname), 1)
                    )

        if not fake_items:
            print(f'  [경고] 가짜(TTS) wav를 찾을 수 없음: {fake_root}')

        random.shuffle(fake_items)
        cut = int(len(fake_items) * 0.8)
        fake_items = fake_items[:cut] if split == 'train' else fake_items[cut:]

        if fake_limit is not None:
            fake_items = fake_items[:fake_limit]

        self.items.extend(fake_items)
        random.shuffle(self.items)

        n_real = sum(1 for _, l in self.items if l == 0)
        n_fake = sum(1 for _, l in self.items if l == 1)
        print(f'  [KoreanDeepfakeDataset/{split}] '
              f'진짜={n_real:,}  가짜={n_fake:,}  합계={len(self.items):,}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, label = self.items[idx]

        try:
            # npy 캐시가 있으면 우선 사용 (wav보다 ~10배 빠름)
            npy_path = wav_path.replace('.wav', '.npy')
            if os.path.exists(npy_path):
                wav = np.load(npy_path)
                sr  = SR   # 전처리 시 이미 SR로 변환됨
            else:
                wav, sr = sf.read(wav_path, dtype='float32')

                # 스테레오 → 모노
                if wav.ndim == 2:
                    wav = wav.mean(axis=1)

                # 샘플레이트 변환
                if sr != SR:
                    try:
                        import resampy
                        wav = resampy.resample(wav, sr, SR)
                    except ImportError:
                        pass
        except Exception:
            return torch.zeros(1, MAX_LEN), label

        # 길이 강제 통일 (리샘플링 후에도 반드시 MAX_LEN)
        if len(wav) < MAX_LEN:
            wav = np.pad(wav, (0, MAX_LEN - len(wav)))
        elif len(wav) > MAX_LEN:
            start = random.randint(0, len(wav) - MAX_LEN)
            wav = wav[start: start + MAX_LEN]
        wav = wav[:MAX_LEN]  # 혹시 남은 오차 제거

        # 노이즈 증강: 진짜 음성에만 (TTS와 품질 차이 보정)
        # 스튜디오 녹음이라도 약간의 노이즈를 추가해 과적합 방지
        if self.noise_aug and label == 0:
            wav = self._add_noise(wav, snr_range=(20, 45))

        return torch.FloatTensor(wav).unsqueeze(0), label

    def _add_noise(self, wav, snr_range=(20, 45)):
        snr_db  = random.uniform(*snr_range)
        sig_pow = np.mean(wav ** 2) + 1e-9
        snr_lin = 10 ** (snr_db / 10)
        noise   = np.random.randn(len(wav)).astype(np.float32)
        noise  *= np.sqrt(sig_pow / snr_lin)
        return np.clip(wav + noise, -1.0, 1.0)


def compute_class_weight(dataset):
    """CrossEntropyLoss(weight=...) 에 넣을 텐서 반환"""
    labels = [label for _, label in dataset.items]
    n_real = labels.count(0)
    n_fake = labels.count(1)
    total  = n_real + n_fake

    if n_real == 0 or n_fake == 0:
        print(f'\n[오류] 진짜={n_real:,}, 가짜={n_fake:,} — 한쪽이 0입니다.')
        print('  FAKE_DATA_ROOT 경로가 잘못됐을 가능성이 높습니다.')
        print('  탐색기에서 TS1 폴더가 보이는 경로를 FAKE_DATA_ROOT에 설정하세요.\n')
        raise ValueError('한쪽 클래스가 0입니다. FAKE_DATA_ROOT 경로를 확인하세요.')

    w_real = total / (2 * n_real)
    w_fake = total / (2 * n_fake)
    print(f'  클래스 가중치 → 진짜: {w_real:.3f}, 가짜: {w_fake:.3f}')
    return torch.tensor([w_real, w_fake], dtype=torch.float32)