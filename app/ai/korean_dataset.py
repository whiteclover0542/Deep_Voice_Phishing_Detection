# =====================================================================
# KoreanDeepfakeDataset - 최종버전 (RawBoost + 전화망 증강)
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
# REAL_TRAIN_ROOT = r'...\자유대화 음성(일반남녀)\Training_denoised'
# REAL_VALID_ROOT = r'...\자유대화 음성(일반남녀)\Validation_denoised'  # 없으면 None → 80/20 fallback
# FAKE_TRAIN_ROOT = r'...\015.감성...\1.Training\원천데이터'
# FAKE_VALID_ROOT = r'...\015.감성...\2.Validation\원천데이터'          # 없으면 None → 80/20 fallback
# =====================================================================

import os
import json
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wav_list_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)

SR      = 16000
MAX_LEN = SR * 4   # 4초
SEED    = 42


# ── RawBoost 노이즈 함수 ───────────────────────────────────────────────

def LnL_convolutive_noise(x, N_f=128, numb_filt=8, minV=10, maxV=40):
    """Linear & Non-Linear 컨볼루션 노이즈"""
    noise = np.zeros_like(x)
    for _ in range(numb_filt):
        f = np.random.uniform(minV, maxV, N_f).astype(np.float32)
        f /= np.sum(np.abs(f)) + 1e-8
        noise += np.convolve(x, f, mode='same')
    return x + 0.1 * noise / (np.max(np.abs(noise)) + 1e-8)

def ISD_additive_noise(x, P=10, g_sd=2):
    """충격성 신호 의존 노이즈"""
    beta = np.random.randn(len(x)).astype(np.float32) * g_sd
    mask = (np.random.uniform(0, 1, len(x)) > P / 100).astype(np.float32)
    return x + beta * mask * x

def SSI_additive_noise(x, SNRmin=10, SNRmax=40):
    """정상 신호 독립 가산 노이즈"""
    SNR = np.random.uniform(SNRmin, SNRmax)
    noise = np.random.randn(len(x)).astype(np.float32)
    sig_pow = np.mean(x ** 2) + 1e-9
    noise_pow = sig_pow / (10 ** (SNR / 10))
    return x + np.sqrt(noise_pow) * noise

def apply_tel_aug(x, sr=SR):
    """전화망 시뮬레이션: 대역 제한 → 8kHz 다운샘플 → μ-law 코덱 → 패킷 손실"""
    from scipy.signal import butter, sosfilt
    import random as _rng

    sos = butter(4, [300, 3400], btype='band', fs=sr, output='sos')
    x = sosfilt(sos, x).astype(np.float32)

    x_8k = x[::2]
    x = np.repeat(x_8k, 2)[:len(x)].astype(np.float32)

    mu = 255
    xc = np.clip(x, -1.0, 1.0)
    encoded = np.sign(xc) * np.log1p(mu * np.abs(xc)) / np.log1p(mu)
    quantized = np.round(encoded * 127) / 127
    x = (np.sign(quantized) * (np.power(1.0 + mu, np.abs(quantized)) - 1.0) / mu).astype(np.float32)

    frame_len = int(sr * 0.02)
    for i in range(len(x) // frame_len):
        if _rng.random() < 0.02:
            x[i * frame_len:(i + 1) * frame_len] = 0.0

    peak = np.abs(x).max()
    if peak > 1e-8:
        x = x / peak * 0.8

    return x


def apply_rawboost(x, algo=5):
    """
    algo: 1=LnL, 2=ISD, 3=SSI,
          4=LnL+ISD, 5=LnL+SSI (추천), 6=ISD+SSI, 7=전부
    """
    if algo == 1:
        return LnL_convolutive_noise(x)
    elif algo == 2:
        return ISD_additive_noise(x)
    elif algo == 3:
        return SSI_additive_noise(x)
    elif algo == 4:
        return ISD_additive_noise(LnL_convolutive_noise(x))
    elif algo == 5:
        return SSI_additive_noise(LnL_convolutive_noise(x))
    elif algo == 6:
        return SSI_additive_noise(ISD_additive_noise(x))
    elif algo == 7:
        return SSI_additive_noise(ISD_additive_noise(LnL_convolutive_noise(x)))
    return x


# ── Dataset ───────────────────────────────────────────────────────────

def _collect_wavs(root, label):
    """root 하위의 모든 .wav 파일을 (경로, label) 리스트로 반환 (결과 캐싱)"""
    cache_key  = hashlib.md5(root.encode('utf-8')).hexdigest()
    cache_path = os.path.join(_CACHE_DIR, f'{cache_key}_{label}.json')

    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            items = [tuple(x) for x in json.load(f)]
        print(f'    캐시 로드: {len(items):,}개  ({os.path.basename(root)})', flush=True)
        return items

    items = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.lower().endswith('.wav'):
                items.append((os.path.join(dirpath, fname), label))
        if len(items) % 5000 == 0 and len(items) > 0:
            print(f'    스캔 중... {len(items):,}개', flush=True)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False)
    print(f'    스캔 완료 → 캐시 저장: {len(items):,}개', flush=True)
    return items


class KoreanDeepfakeDataset(Dataset):
    def __init__(
        self,
        real_root,             # 109번 Training 폴더 (스튜디오_1, _2 상위)
        fake_root,             # 466번 Training 원천데이터 (TS1, TS2... 상위)
        split='train',
        real_val_root=None,    # 109번 Validation 폴더 — 있으면 공식 분할 사용, 없으면 80/20
        fake_val_root=None,    # 466번 Validation 원천데이터 — 있으면 공식 분할 사용, 없으면 80/20
        phone_real_root=None,  # 전화망 진짜 음성 루트 (train split에만 추가)
        noise_aug=True,        # 진짜 음성 RawBoost 증강
        rawboost_algo=5,       # RawBoost 알고리즘 (기본: LnL+SSI)
        tel_aug_fake=True,     # 가짜(TTS) 음성에 전화망 시뮬레이션 적용
        real_limit=None,       # 진짜 최대 샘플 수 (None=전부)
        fake_limit=None,       # 가짜 최대 샘플 수 (None=전부)
    ):
        self.noise_aug = noise_aug
        self.rawboost_algo = rawboost_algo
        self.tel_aug_fake = tel_aug_fake
        self.items = []

        use_official_split = (real_val_root is not None) or (fake_val_root is not None)

        # ── 진짜 음성 ────────────────────────────────────────────────
        USE_STUDIOS = ['[원천]3.스튜디오_1', '[원천]3.스튜디오_2']

        if use_official_split and split == 'dev' and real_val_root:
            # 공식 Validation 폴더 사용
            real_items = _collect_wavs(real_val_root, 0)
            print(f'  [진짜-공식Val] {len(real_items):,}개  ({real_val_root})')
        else:
            # Training 폴더에서 스튜디오 데이터 수집
            real_items = []
            studio_dirs = [
                d for d in USE_STUDIOS
                if os.path.isdir(os.path.join(real_root, d))
            ]
            if not studio_dirs:
                print(f'  [경고] 스튜디오 폴더를 찾을 수 없음: {real_root}')
            else:
                print(f'  [진짜-스튜디오] 폴더: {studio_dirs}')
                for studio_dir in studio_dirs:
                    real_items.extend(
                        _collect_wavs(os.path.join(real_root, studio_dir), 0)
                    )

            # 전화망 데이터 추가 (train split에만)
            if phone_real_root and split == 'train':
                if os.path.isdir(phone_real_root):
                    phone_items = _collect_wavs(phone_real_root, 0)
                    print(f'  [진짜-전화망] {len(phone_items):,}개  ({phone_real_root})')
                    real_items.extend(phone_items)
                else:
                    print(f'  [경고] 전화망 음성 폴더를 찾을 수 없음: {phone_real_root}')

            # 공식 분할이 없을 때만 80/20
            if not use_official_split:
                random.shuffle(real_items)
                cut = int(len(real_items) * 0.8)
                real_items = real_items[:cut] if split == 'train' else real_items[cut:]

        random.shuffle(real_items)
        if real_limit is not None:
            real_items = real_items[:real_limit]
        self.items.extend(real_items)

        # ── 가짜 음성 ────────────────────────────────────────────────
        if use_official_split and split == 'dev' and fake_val_root:
            # 공식 Validation 폴더 사용
            fake_items = _collect_wavs(fake_val_root, 1)
            print(f'  [가짜-공식Val] {len(fake_items):,}개  ({fake_val_root})')
        else:
            fake_items = _collect_wavs(fake_root, 1)
            if not fake_items:
                print(f'  [경고] 가짜(TTS) wav를 찾을 수 없음: {fake_root}')

            # 공식 분할이 없을 때만 80/20
            if not use_official_split:
                random.shuffle(fake_items)
                cut = int(len(fake_items) * 0.8)
                fake_items = fake_items[:cut] if split == 'train' else fake_items[cut:]

        random.shuffle(fake_items)
        if fake_limit is not None:
            fake_items = fake_items[:fake_limit]
        self.items.extend(fake_items)

        random.shuffle(self.items)

        n_real = sum(1 for _, l in self.items if l == 0)
        n_fake = sum(1 for _, l in self.items if l == 1)
        mode = '공식분할' if use_official_split else '80/20분할'
        print(f'  [KoreanDeepfakeDataset/{split}/{mode}] '
              f'진짜={n_real:,}  가짜={n_fake:,}  합계={len(self.items):,}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, label = self.items[idx]

        try:
            npy_path = wav_path.replace('.wav', '.npy')
            if os.path.exists(npy_path):
                wav = np.load(npy_path)
                sr  = SR
            else:
                wav, sr = sf.read(wav_path, dtype='float32')

                if wav.ndim == 2:
                    wav = wav.mean(axis=1)

                if sr != SR:
                    try:
                        import resampy
                        wav = resampy.resample(wav, sr, SR)
                    except ImportError:
                        pass

                try:
                    np.save(npy_path, wav)
                except Exception:
                    pass
        except Exception:
            return torch.zeros(1, MAX_LEN), label

        # 길이 통일
        if len(wav) < MAX_LEN:
            wav = np.pad(wav, (0, MAX_LEN - len(wav)))
        elif len(wav) > MAX_LEN:
            start = random.randint(0, len(wav) - MAX_LEN)
            wav = wav[start: start + MAX_LEN]
        wav = wav[:MAX_LEN]

        # RawBoost 증강: 진짜 음성에만 적용
        if self.noise_aug and label == 0:
            wav = apply_rawboost(wav, algo=self.rawboost_algo)

        # 전화망 시뮬레이션: 가짜(TTS) 음성에 적용
        if self.tel_aug_fake and label == 1:
            wav = apply_tel_aug(wav)

        wav = np.clip(wav, -1.0, 1.0)

        return torch.FloatTensor(wav).unsqueeze(0), label


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