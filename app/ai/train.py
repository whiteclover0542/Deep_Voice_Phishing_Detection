# =====================================================================
# [Colab 전용] RawNet2 학습 스크립트
# 실행 전: Google Drive에 아래 구조로 데이터를 올려두세요
#
# MyDrive/
# └── RealTimeVoicePhishing/
#     ├── model.py
#     ├── dataset.py
#     └── train.py  ← 이 파일
# MyDrive/
# └── data/
#     └── LA/
#         ├── ASVspoof2019_LA_cm_protocols/
#         ├── ASVspoof2019_LA_train/
#         ├── ASVspoof2019_LA_dev/
#         └── ASVspoof2019_LA_eval/
# =====================================================================

# ── 1. Google Drive 마운트 ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, sys

PROJECT_DIR = '/content/drive/MyDrive/RealTimeVoicePhishing'
sys.path.append(PROJECT_DIR)

# ── 2. 패키지 설치 (최초 1회) ──────────────────────────────────────
os.system('pip install soundfile scikit-learn -q')

# ── 3. 임포트 ──────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from model import RawNet2
from dataset import ASVspoof2019LA

# ── 4. 경로 / 하이퍼파라미터 ───────────────────────────────────────
DATA_ROOT  = '/content/drive/MyDrive/data/LA'
SAVE_DIR   = '/content/drive/MyDrive/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 24
EPOCHS     = 20
LR         = 1e-4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}')

# ── 5. 데이터 로더 ─────────────────────────────────────────────────
train_loader = DataLoader(
    ASVspoof2019LA(DATA_ROOT, 'train'),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
dev_loader = DataLoader(
    ASVspoof2019LA(DATA_ROOT, 'dev'),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

# ── 6. 모델 / 옵티마이저 / 손실 ────────────────────────────────────
model     = RawNet2().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── 7. EER 계산 ────────────────────────────────────────────────────
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100  # %

# ── 8. 에폭 함수 ───────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for wav, label in loader:
        wav, label = wav.to(device), label.to(device)
        optimizer.zero_grad()
        loss = criterion(model(wav), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(label)
        correct += (model(wav).argmax(1) == label).sum().item()
        n += len(label)
    return total_loss / n, correct / n


def evaluate_eer(model, loader, device):
    model.eval()
    all_labels, all_scores = [], []
    with torch.no_grad():
        for wav, label in loader:
            logits = model(wav.to(device))
            scores = torch.softmax(logits, 1)[:, 1]  # spoof 확률
            all_labels.extend(label.numpy())
            all_scores.extend(scores.cpu().numpy())
    return compute_eer(np.array(all_labels), np.array(all_scores))

# ── 9. 학습 루프 ───────────────────────────────────────────────────
best_eer = 100.0

for epoch in range(1, EPOCHS + 1):
    loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    eer = evaluate_eer(model, dev_loader, DEVICE)
    scheduler.step()

    print(f'Epoch {epoch:02d}/{EPOCHS} | loss: {loss:.4f} | acc: {acc:.4f} | dev EER: {eer:.2f}%')

    if eer < best_eer:
        best_eer = eer
        ckpt_path = os.path.join(SAVE_DIR, 'best_model.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f'  → best model saved  (EER {best_eer:.2f}%)')

print(f'\n학습 완료. 최종 best EER: {best_eer:.2f}%')
print(f'가중치 저장 위치: {SAVE_DIR}/best_model.pth')
