# =====================================================================
# KoreanDeepfakeDataset
#
# 데이터셋 109 (자유대화 음성 - 진짜, label=0)
# 데이터셋 466 (감성 발화 TTS - 가짜, label=1)
# =====================================================================

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf


# ── 실제 폴더 구조 ────────────────────────────────────────────────────
#
# [109번 - 진짜]
# real_root/
#   일반남여_통합01_F_HSH00_.../
#     *.wav
#
# [466번 - 가짜 TTS]
# fake_root/
#   1.Training/원천데이터/
#     TS1/TS1/1.감정/1.기쁨/0001_.../  *.wav
#     TS2/TS2/1.감정/2.슬픔/...
#   1.Validation/원천데이터/   (있을 경우)
#     ...
# ─────────────────────────────────────────────────────────────────────


SR      = 16000       # 목표 샘플레이트
MAX_LEN = SR * 4      # 4초 고정 (64,000 샘플)
SEED    = 42


class KoreanDeepfakeDataset(Dataset):
    """
    split     : 'train' | 'dev'
    noise_aug : True면 진짜 음성에 노이즈 증강 적용 (품질 차이 보정용)
    real_limit: 진짜 음성 최대 샘플 수 (None이면 전부 사용)
    """

    def __init__(
        self,
        real_root,        # 109번 데이터 루트
        fake_root,        # 466번 데이터 루트
        split='train',
        noise_aug=True,
        real_limit=None,
    ):
        self.noise_aug = noise_aug
        self.items = []   # (wav_path, label)

        # ── 진짜 음성 (109번) ──────────────────────────────────────────
        real_items = []
        for dirpath, _, files in os.walk(real_root):
            for fname in files:
                if fname.endswith('.wav'):
                    real_items.append((os.path.join(dirpath, fname), 0))

        # train/dev 분할 (8:2)
        random.seed(SEED)
        random.shuffle(real_items)
        cut = int(len(real_items) * 0.8)
        real_items = real_items[:cut] if split == 'train' else real_items[cut:]

        # 최대 샘플 수 제한 (클래스 불균형 보정용)
        if real_limit is not None:
            real_items = real_items[:real_limit]

        self.items.extend(real_items)

        # ── 가짜 음성 (466번 TTS) ──────────────────────────────────────
        # 실제 폴더명: "1.Training" / "1.Validation"
        # os.walk로 재귀 탐색하므로 TS1~TS4, 감정 하위폴더 깊이 무관
        split_folder_candidates = {
            'train': ['1.Training', 'Training'],
            'dev':   ['1.Validation', 'Validation'],
        }

        fake_wav_root = None
        for candidate in split_folder_candidates[split]:
            path = os.path.join(fake_root, candidate, '원천데이터')
            if os.path.isdir(path):
                fake_wav_root = path
                break

        fake_items = []
        if fake_wav_root:
            print(f'  [466번 탐색 경로] {fake_wav_root}')
            for dirpath, _, files in os.walk(fake_wav_root):
                for fname in files:
                    if fname.endswith('.wav'):
                        fake_items.append((os.path.join(dirpath, fname), 1))
        else:
            # 원천데이터 폴더를 못 찾으면 fake_root 전체 재귀 탐색
            print(f'  [466번 경고] 원천데이터 폴더를 찾지 못해 {fake_root} 전체를 탐색합니다.')
            for dirpath, _, files in os.walk(fake_root):
                for fname in files:
                    if fname.endswith('.wav'):
                        fake_items.append((os.path.join(dirpath, fname), 1))

        random.seed(SEED)
        random.shuffle(fake_items)
        cut = int(len(fake_items) * 0.8)
        fake_items = fake_items[:cut] if split == 'train' else fake_items[cut:]

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

        # ── wav 로드 ───────────────────────────────────────────────────
        try:
            wav, sr = sf.read(wav_path, dtype='float32')
        except Exception:
            wav = np.zeros(MAX_LEN, dtype=np.float32)
            return torch.FloatTensor(wav).unsqueeze(0), label

        # 스테레오 → 모노
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        # 샘플레이트 변환 (필요 시)
        if sr != SR:
            import resampy
            wav = resampy.resample(wav, sr, SR)

        # ── 길이 통일 (4초) ────────────────────────────────────────────
        if len(wav) < MAX_LEN:
            wav = np.pad(wav, (0, MAX_LEN - len(wav)))
        else:
            # 랜덤 크롭 (학습 다양성)
            start = random.randint(0, len(wav) - MAX_LEN)
            wav = wav[start: start + MAX_LEN]

        # ── 노이즈 증강 (진짜 음성에만, 품질 차이 보정) ────────────────
        if self.noise_aug and label == 0:
            wav = self._add_noise(wav)

        return torch.FloatTensor(wav).unsqueeze(0), label

    def _add_noise(self, wav, snr_range=(15, 40)):
        """
        랜덤 SNR로 가우시안 노이즈 추가.
        snr_range: (min_db, max_db) — 높을수록 노이즈 작음
        """
        snr_db  = random.uniform(*snr_range)
        sig_pow = np.mean(wav ** 2) + 1e-9
        snr_lin = 10 ** (snr_db / 10)
        noise   = np.random.randn(len(wav)).astype(np.float32)
        noise  *= np.sqrt(sig_pow / snr_lin)
        return np.clip(wav + noise, -1.0, 1.0)


# ── 클래스 가중치 계산 헬퍼 ──────────────────────────────────────────
def compute_class_weight(dataset):
    """
    CrossEntropyLoss(weight=...) 에 넣을 텐서 반환.
    샘플 수에 반비례하게 가중치 계산.
    """
    labels = [label for _, label in dataset.items]
    n_real = labels.count(0)
    n_fake = labels.count(1)
    total  = n_real + n_fake
    w_real = total / (2 * n_real)
    w_fake = total / (2 * n_fake)
    print(f'  클래스 가중치 → 진짜: {w_real:.3f}, 가짜: {w_fake:.3f}')
    return torch.tensor([w_real, w_fake], dtype=torch.float32)