# =====================================================================
# [VS Code 로컬 실행용] RawNet2 학습 스크립트
#
# 실행 전 패키지 설치:
#   pip install torch soundfile scikit-learn scipy tqdm
# =====================================================================

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

# ── 1. 경로 설정 ───────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

DATA_ROOT = r'D:\User\Desktop\데이터셋\ASVspoof\archive'  # ← 본인 경로
SAVE_DIR  = os.path.join(BASE_DIR, 'checkpoints')
os.makedirs(SAVE_DIR, exist_ok=True)

from model import RawNet2
from dataset import ASVspoof2019LA

# ── 2. 하이퍼파라미터 ──────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 1e-4
MAX_SAMPLES = None

CKPT_PATH   = os.path.join(SAVE_DIR, 'best_model.pth')
RESUME      = False   # ← 이어서 학습: True / 처음부터 시작: False

# ── 3. EER 계산 ────────────────────────────────────────────────────
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100

# ── 4. 에폭 함수 ───────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    pbar = tqdm(
        loader,
        desc=f'  [Train] Epoch {epoch:02d}/{total_epochs}',
        unit='batch',
        ncols=90,
        leave=True,
    )
    for wav, label in pbar:
        wav, label = wav.to(device), label.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            logits = model(wav)
            loss = criterion(logits, label)

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

    pbar = tqdm(
        loader,
        desc=f'  [Eval]  Epoch {epoch:02d}/{total_epochs}',
        unit='batch',
        ncols=90,
        leave=True,
    )
    with torch.no_grad():
        for wav, label in pbar:
            with autocast('cuda'):
                logits = model(wav.to(device))
            scores = torch.softmax(logits, 1)[:, 1]
            all_labels.extend(label.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    return compute_eer(np.array(all_labels), np.array(all_scores))

# ── 5. 학습 루프 ───────────────────────────────────────────────────
def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'device      : {device}')
    print(f'data root   : {DATA_ROOT}')
    print(f'checkpoint  : {SAVE_DIR}')
    print(f'max_samples : {MAX_SAMPLES}')

    train_loader = DataLoader(
        ASVspoof2019LA(DATA_ROOT, 'train', max_samples=MAX_SAMPLES),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(device == 'cuda'),
        persistent_workers=True,
    )
    dev_loader = DataLoader(
        ASVspoof2019LA(DATA_ROOT, 'dev', max_samples=MAX_SAMPLES),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device == 'cuda'),
        persistent_workers=True,
    )

    model     = RawNet2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler('cuda')

    # ── 이어서 학습 ─────────────────────────────────────────────────
    best_eer    = 100.0
    start_epoch = 1

    if RESUME and os.path.exists(CKPT_PATH):
        print(f'\n체크포인트 발견 → 이어서 학습합니다.')
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            # 새 형식 (이어서 학습 가능)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_eer    = ckpt['best_eer']
        else:
            # 옛날 형식 (가중치만 있음 → 1에폭부터 시작)
            model.load_state_dict(ckpt)
            print('  (이전 체크포인트는 에폭 정보 없음 → Epoch 1부터 시작)')
        print(f'재시작: Epoch {start_epoch}/{EPOCHS}  |  이전 best EER: {best_eer:.2f}%')
    else:
        print(f'\n처음부터 학습합니다.')

    print()

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS, scaler)
        eer       = evaluate_eer(model, dev_loader, device, epoch, EPOCHS)
        scheduler.step()

        elapsed = time.time() - t0
        print(f'Epoch {epoch:02d}/{EPOCHS} | loss {loss:.4f} | acc {acc:.4f} | EER {eer:.2f}% | {elapsed/60:.1f}min')

        if eer < best_eer:
            best_eer = eer
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_eer':  best_eer,
            }, CKPT_PATH)
            print(f'  → best model saved  (EER {best_eer:.2f}%)')

    print(f'\n학습 완료. 최종 best EER: {best_eer:.2f}%')
    print(f'가중치 저장 위치: {CKPT_PATH}')


if __name__ == '__main__':
    main()