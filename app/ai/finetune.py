# =====================================================================
# [파인튜닝용] RawNet2 - ASVspoof(영어) → 한국어 딥페이크 탐지
# 데이터셋: 109번(자유대화, 진짜) + 466번(TTS, 가짜)
# =====================================================================

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.amp import autocast, GradScaler
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from tqdm import tqdm
import random

torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from model import RawNet2
from dataset import ASVspoof2019LA
from korean_dataset import KoreanDeepfakeDataset, compute_class_weight

# ── 1. 경로 설정 ───────────────────────────────────────────────────
PRETRAINED_CKPT = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pth')

# 109번: Training 폴더 지정 ([원천]3.스튜디오_1, _2 를 자동 탐색)
REAL_DATA_ROOT = r'D:\User\Desktop\데이터셋\자유대화 음성(일반남녀)\Training_denoised'

# 466번: 원천데이터 폴더 지정 (TS1/1.기쁨/... 를 자동 탐색)
FAKE_DATA_ROOT = r'D:\User\Desktop\데이터셋\015.감성 및 발화 스타일별 음성합성 데이터\01.데이터\1.Training\원천데이터'
EN_DATA_ROOT   = r'D:\User\Desktop\데이터셋\ASVspoof\archive'     # 영어 (catastrophic forgetting 방지)

SAVE_DIR  = os.path.join(BASE_DIR, 'checkpoints_ko')
CKPT_PATH = os.path.join(SAVE_DIR, 'best_model_ko.pth')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 2. 하이퍼파라미터 ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 15
STAGE      = 1       # 1 → 2 → 3 순서로 올리면서 실행

LR_MAP = {1: 1e-4, 2: 1e-5, 3: 1e-6}
LR = LR_MAP[STAGE]

# 영어 데이터 혼합 (True: 영어+한국어 / False: 한국어만)
MIX_ENGLISH = True
EN_RATIO    = 0.2    # 영어 비율 (소량 유지로 catastrophic forgetting 방지)

# ── 데이터 샘플 수 제한 ──────────────────────────────────────────
# 테스트용: 각각 5000개 정도로 빠르게 확인
# 본학습:   None (전부 사용)
MAX_REAL = 5000   # 진짜 음성 최대 샘플 수 (None=전부)
MAX_FAKE = 5000   # 가짜 음성 최대 샘플 수 (None=전부)

# ── Early Stopping ────────────────────────────────────────────────
PATIENCE = 5   # 몇 에폭 연속 개선 없으면 멈출지 (Stage 1: 5, Stage 3: 3 권장)

# ── 3. 레이어 동결 ────────────────────────────────────────────────
#
# Stage 1: FC + GRU만 학습   (한국어 빠른 적응)
# Stage 2: layer4~6 해제     (중간 특징 조정)
# Stage 3: 전체 해제          (전체 미세조정, lr 매우 낮게)
#
FREEZE_MAP = {
    1: ['sinc_conv', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'bn_before_gru'],
    2: ['sinc_conv', 'layer1', 'layer2', 'layer3'],
    3: [],
}

def freeze_layers(model, stage):
    freeze_keywords = FREEZE_MAP[stage]
    for name, param in model.named_parameters():
        param.requires_grad = not any(kw in name for kw in freeze_keywords)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'\n[Stage {stage}] 동결: {freeze_keywords if freeze_keywords else "없음 (전체 학습)"}')
    print(f'  학습 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)')

# ── 4. EER ────────────────────────────────────────────────────────
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100

# ── 5. 에폭 함수 ──────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc=f'  [Train] Epoch {epoch:02d}/{total_epochs}',
                unit='batch', ncols=90, leave=True)
    for wav, label in pbar:
        wav, label = wav.to(device), label.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(wav)
            loss   = criterion(logits, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(label)
        correct    += (logits.argmax(1) == label).sum().item()
        n          += len(label)
        pbar.set_postfix(loss=f'{total_loss/n:.4f}', acc=f'{correct/n:.4f}')
    return total_loss / n, correct / n


def evaluate_eer(model, loader, device, epoch, total_epochs):
    model.eval()
    all_labels, all_scores = [], []
    pbar = tqdm(loader, desc=f'  [Eval]  Epoch {epoch:02d}/{total_epochs}',
                unit='batch', ncols=90, leave=True)
    with torch.no_grad():
        for wav, label in pbar:
            with autocast('cuda'):
                logits = model(wav.to(device))
            scores = torch.softmax(logits, 1)[:, 1]
            all_labels.extend(label.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    return compute_eer(np.array(all_labels), np.array(all_scores))

# ── 6. 데이터 로더 ────────────────────────────────────────────────
TARGET_LEN = 64000  # 4초 @ 16kHz — 모든 데이터셋 공통 기준

def collate_fn(batch):
    """배치 내 wav 길이를 TARGET_LEN으로 강제 통일"""
    wavs, labels = [], []
    for wav, label in batch:
        wav = wav.squeeze(0)  # (1, L) → (L,)
        if wav.shape[0] < TARGET_LEN:
            wav = torch.nn.functional.pad(wav, (0, TARGET_LEN - wav.shape[0]))
        else:
            wav = wav[:TARGET_LEN]
        wavs.append(wav.unsqueeze(0))  # (1, L)
        labels.append(label)
    return torch.stack(wavs, 0), torch.tensor(labels)

def build_loaders(split='train'):
    ko_dataset = KoreanDeepfakeDataset(
        real_root=REAL_DATA_ROOT,
        fake_root=FAKE_DATA_ROOT,
        split=split,
        noise_aug=False,
        real_limit=MAX_REAL,
        fake_limit=MAX_FAKE,
    )

    if MIX_ENGLISH and split == 'train':
        en_dataset  = ASVspoof2019LA(EN_DATA_ROOT, 'train')
        en_target   = int(len(ko_dataset) * EN_RATIO / (1 - EN_RATIO))
        en_target   = min(en_target, len(en_dataset))
        en_indices  = random.sample(range(len(en_dataset)), en_target)
        combined    = ConcatDataset([ko_dataset, Subset(en_dataset, en_indices)])
        print(f'  [영어 혼합] 한국어={len(ko_dataset):,} + 영어={en_target:,} = {len(combined):,}')
    else:
        combined = ko_dataset

    loader = DataLoader(
        combined,
        batch_size=BATCH_SIZE,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    return loader, ko_dataset


# ── 7. 메인 ──────────────────────────────────────────────────────
def main():
    device = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')

    print('=' * 55)
    print(f'  Stage {STAGE}  |  LR {LR}  |  device: {device}')
    print('=' * 55)

    print('\n[데이터 로딩]')
    train_loader, train_dataset = build_loaders('train')
    dev_loader,   _             = build_loaders('dev')

    model = RawNet2().to(device)
    if not os.path.exists(PRETRAINED_CKPT):
        raise FileNotFoundError(f'체크포인트 없음: {PRETRAINED_CKPT}')

    print(f'\n[사전학습 가중치 로드]  {PRETRAINED_CKPT}')
    ckpt       = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)

    freeze_layers(model, STAGE)

    print('\n[클래스 가중치 계산]')
    class_weight = compute_class_weight(train_dataset).to(device)
    criterion    = nn.CrossEntropyLoss(weight=class_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler('cuda')
    best_eer      = 100.0
    no_improve    = 0      # 개선 없는 에폭 카운터

    print(f'\n[학습 시작]  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  LR={LR}  PATIENCE={PATIENCE}\n')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                    device, epoch, EPOCHS, scaler)
        eer       = evaluate_eer(model, dev_loader, device, epoch, EPOCHS)
        scheduler.step()
        elapsed = time.time() - t0
        print(f'Epoch {epoch:02d}/{EPOCHS} | loss {loss:.4f} | acc {acc:.4f} '
              f'| EER {eer:.2f}% | {elapsed/60:.1f}min')

        if eer < best_eer:
            best_eer   = eer
            no_improve = 0
            torch.save({
                'epoch':     epoch,
                'stage':     STAGE,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_eer':  best_eer,
            }, CKPT_PATH)
            print(f'  → best model saved  (EER {best_eer:.2f}%)')
        else:
            no_improve += 1
            print(f'  개선 없음 ({no_improve}/{PATIENCE})')
            if no_improve >= PATIENCE:
                print(f'\n  Early stopping: {PATIENCE}에폭 연속 개선 없음 → 학습 종료')
                break

    print(f'\n파인튜닝 완료. best EER: {best_eer:.2f}%')
    print(f'저장: {CKPT_PATH}')
    if STAGE < 3:
        print(f'\n다음 단계: STAGE={STAGE+1}, PRETRAINED_CKPT="{CKPT_PATH}" 로 수정 후 재실행')


if __name__ == '__main__':
    main()