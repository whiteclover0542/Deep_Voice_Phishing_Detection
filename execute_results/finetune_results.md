# RawNet2 한국어 파인튜닝 실행 기록

---

## EER 0.00% 방지 조치

EER 0.00%는 모델이 검증 데이터를 완벽하게 분리했다는 뜻으로, 실제로는 **데이터 누출 또는 과적합**일 가능성이 높음.  
이를 방지하기 위해 아래 조치를 적용함.

| 조치 | 적용 위치 | 이유 |
|---|---|---|
| **공식 Train/Val 분할** | `real_val_root`, `fake_val_root` 별도 폴더 지정 | Train 데이터와 Validation 데이터가 겹치지 않도록 완전 분리 |
| **클래스 가중치 적용** | `compute_class_weight` → `CrossEntropyLoss(weight=...)` | 진짜/가짜 샘플 수 불균형 시 한쪽 클래스만 예측하는 붕괴 방지 |
| **RawBoost 증강** | 진짜 음성에 LnL+SSI 노이즈 적용 | 진짜 샘플을 쉽게 외우지 못하도록 매 에폭 무작위 변형 |
| **전화망 시뮬레이션 증강** | 가짜(TTS) 음성에 대역 제한 + μ-law 코덱 적용 | TTS 특유의 깨끗한 음질을 흐려서 단순 음질 차이로 구분하는 것 방지 |
| **영어 데이터 혼합** | `MIX_ENGLISH=True`, `EN_RATIO=0.2` | 한국어 데이터만으로 학습 시 한국어 고유 특성에 과적합되는 것 방지 |
| **샘플 수 제한** | `MAX_REAL`, `MAX_FAKE` | 동일 샘플 반복 학습으로 인한 암기 방지 |
| **Early Stopping** | `PATIENCE` 설정 | EER이 더 이상 개선되지 않으면 조기 종료 |
| **3-Stage 점진적 언프리징** | Stage 1→2→3 순서로 레이어 해제 | 한 번에 전체 학습 시 검증 데이터에만 맞는 가중치로 수렴하는 것 방지 |

---

## 한국어 파인튜닝 실패 기록 (finetune.py)

### 실패 1차 — 데이터 로딩 단계에서 멈춤

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-05-01 |
| 설정 | MAX_REAL=None, MAX_FAKE=None, BATCH=32, num_workers=4 |
| 증상 | `[진짜-스튜디오] 폴더: [...]` 출력 후 수십 분 무한 대기 |
| 원인 | `os.walk`로 약 197만 개 wav 파일 스캔 중 — 진행 표시 없어 멈춘 것처럼 보임 |
| 조치 | `_collect_wavs`에 5,000개마다 진행 출력 추가 |

### 실패 2차 — 학습 속도 과도하게 느림

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-05-01 |
| 설정 | MAX_REAL=None, MAX_FAKE=None, BATCH=32, num_workers=4 |
| 증상 | Epoch 01/30에서 8시간 27분에 24% 진행 (총 58,216 배치) |
| 예상 완료 | 에폭당 약 35시간 → 30에폭 완료에 약 1,050시간 |
| 원인 1 | 데이터 제한 없음 → 총 샘플 약 186만 개 |
| 원인 2 | `LnL_convolutive_noise` N_f=256, numb_filt=20 → 샘플당 20회 convolve (CPU 병목) |
| 원인 3 | num_workers=4, prefetch_factor 없음 |
| 조치 | 아래 개선 사항 참고 |

### 실패 3차 — precache_npy.py 중단

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-05-01 |
| 설정 | 전체 197만 개 wav → npy 변환 시도 |
| 증상 | 수 시간 후 484,000 / 1,976,759 (24%) 진행 상태에서 중단 |
| 원인 | 전체 파일 사전 캐싱은 불필요 — 사용할 샘플만 캐싱하면 충분 |
| 조치 | 샘플 수 제한 후 `__getitem__` 내 lazy 캐싱으로 전환 |

### 실패 4차 — Stage 1 레이어 동결 미작동

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-05-02 |
| 설정 | Stage 1, MAX_REAL=8,000, MAX_FAKE=24,000, BATCH=32, EPOCHS=10 |
| 증상 | `학습 파라미터: 18,541,314 / 18,541,314 (100.0%)` — 전체 모델 학습됨 |
| 원인 | FREEZE_MAP 키워드(`sinc_conv`, `layer1`~`layer6`, `bn_before_gru`)가 실제 파라미터 이름과 불일치 |
| 결과 | Best EER 4.20% (Epoch 07) — 값 자체는 유효하나 의도한 설계(GRU+FC만 학습)와 다름 |
| 조치 | FREEZE_MAP을 실제 이름(`sinc`, `bn0`, `blocks.0`~`blocks.5`, `bn_out`)으로 수정 후 재실행 |

### 실패 5차 — 검증 데이터에 증강 적용으로 EER 측정 불안정 (Stage 1~3 전체)

| 항목 | 내용 |
|---|---|
| 날짜 | 2026-05-03 ~ 2026-05-05 |
| 증상 | EER이 에폭마다 최대 ±15% 폭으로 오르내림 |
| 원인 | `build_loaders`에서 train/dev 구분 없이 `noise_aug=True`, `tel_aug_fake=True` 적용 → 검증 샘플에도 매번 다른 노이즈가 붙어 EER 측정값 자체가 비결정적 |
| 조치 | `noise_aug=is_train`, `tel_aug_fake=is_train` 으로 수정 — dev는 원본 음성으로 평가 |

**Stage 1 결과 (검증 증강 있음 — 신뢰도 낮음)**

| Epoch | Train Loss | Train Acc | Eval EER | 비고 |
|---|---|---|---|---|
| 01 | 0.1583 | 0.9470 | 7.87% | Best |
| 02~06 | — | — | 8.74~10.83% | Early stopping (5/5) |

**Stage 2 결과 (검증 증강 있음 — 신뢰도 낮음)**

| Epoch | Train Loss | Train Acc | Eval EER | 비고 |
|---|---|---|---|---|
| 01 | 0.0722 | 0.9751 | 6.11% | Best |
| 04 | 0.0252 | 0.9913 | 5.23% | Best 갱신 |
| 06 | 0.0183 | 0.9936 | 4.04% | Best 갱신 |
| 13 | 0.0035 | 0.9987 | 20.22% | Early stopping (7/7) |

**Stage 3 결과 (검증 증강 있음 — 신뢰도 낮음, 중단)**

| Epoch | Train Loss | Train Acc | Eval EER | 비고 |
|---|---|---|---|---|
| 01 | 0.0170 | 0.9945 | 7.21% | Best |
| 03 | 0.0118 | 0.9964 | 3.22% | Best 갱신 |
| 10 | 0.0049 | 0.9985 | 10.20% | 중단 (7/10) |

---

## EER 스파이크 원인 분석

| 원인 | 설명 | 상태 |
|---|---|---|
| **검증 데이터에 증강 적용** (주원인) | dev에도 RawBoost/전화망 노이즈가 매번 달리 붙어 EER 측정값 자체가 비결정적 | ✅ 수정 완료 (`is_train` 조건 분기) |
| **레이어 동결 미작동** | 18.5M 전체가 LR=1e-4로 업데이트 → 가중치 변화 과대 | ✅ 수정 완료 (FREEZE_MAP 키워드 수정) |
| **그래디언트 폭발** | 일부 배치에서 손실이 크면 가중치가 크게 이동 | ✅ 클리핑 추가 (`clip_grad_norm_=1.0`) |
| **RawBoost 무작위 증강** | 학습 신호가 에폭마다 달라짐 | ⚠️ 불가피 — 위 3가지 수정 후 잔여 스파이크 모니터링 필요 |

---

## 개선 사항

| 항목 | 변경 전 | 변경 후 | 이유 |
|---|---|---|---|
| 데이터 수 | 제한 없음 (~186만) | MAX_REAL=8,000 / MAX_FAKE=24,000 | 에폭당 35시간 → ~30분 |
| LnL 필터 수 | numb_filt=20, N_f=256 | numb_filt=8, N_f=128 | 증강 속도 약 4배 향상 |
| DataLoader workers | num_workers=4 | num_workers=8 | 데이터 로딩 병목 감소 |
| prefetch_factor | 없음 | prefetch_factor=4 | GPU 대기 시간 감소 |
| npy 캐시 방식 | 별도 precache 스크립트 | `__getitem__` 내 lazy 저장 | 전체 사전 변환 불필요 |
| 학습 전략 | 단일 실행 | 3-Stage 점진적 언프리징 | 과적합 방지, 안정적 수렴 |
| PATIENCE | 15 | Stage별 5 / 7 / 10 | 적은 데이터에서 오버피팅 조기 감지 |
| 검증 증강 | train/dev 동일 증강 | dev 증강 제거 (`is_train` 분기) | EER 측정 안정화 |
| 그래디언트 클리핑 | 없음 | `clip_grad_norm_=1.0` | 가중치 폭발 방지 |
| Eval 배치 크기 | BATCH_SIZE와 동일 | `EVAL_BATCH_SIZE=128` | eval 속도 ~4x 향상 |

---

## 실행 계획 (3-Stage)

| Stage | 학습 레이어 | REAL | FAKE | BATCH | EPOCHS | PATIENCE | LR | PRETRAINED_CKPT |
|---|---|---|---|---|---|---|---|---|
| 1 | GRU + FC | 8,000 | 24,000 | 32 | 10 | 5 | 1e-4 | `checkpoints/best_model.pth` |
| 2 | blocks.3~5 + bn_out + GRU + FC | 15,000 | 45,000 | 32 | 15 | 7 | 1e-5 | `checkpoints_ko/best_model_ko.pth` |
| 3 | 전체 | 25,000 | 75,000 | 16 | 20 | 10 | 1e-6 | `checkpoints_ko/best_model_ko.pth` |

---

### Stage별 finetune.py 수정 방법

#### Stage 1 실행 전 설정

```python
PRETRAINED_CKPT = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pth')  # 영어 사전학습 가중치
RESUME     = False
BATCH_SIZE = 32
EPOCHS     = 10
STAGE      = 1
MAX_REAL   = 8000
MAX_FAKE   = 24000
PATIENCE   = 5
```

#### Stage 1 → Stage 2 전환 시 수정 (Stage 1 완료 후)

```python
PRETRAINED_CKPT = os.path.join(BASE_DIR, 'checkpoints_ko', 'best_model_ko.pth')  # Stage 1 결과물
RESUME     = False  # Stage 전환은 항상 False (이어서가 아닌 새 Stage 시작)
BATCH_SIZE = 32
EPOCHS     = 15
STAGE      = 2
MAX_REAL   = 15000
MAX_FAKE   = 45000
PATIENCE   = 7
```

#### Stage 2 → Stage 3 전환 시 수정 (Stage 2 완료 후)

```python
PRETRAINED_CKPT = os.path.join(BASE_DIR, 'checkpoints_ko', 'best_model_ko.pth')  # Stage 2 결과물
RESUME     = False  # Stage 전환은 항상 False
BATCH_SIZE = 16
EPOCHS     = 20
STAGE      = 3
MAX_REAL   = 25000
MAX_FAKE   = 75000
PATIENCE   = 10
```

> **RESUME=True는 같은 Stage 도중 중단됐을 때만 사용.** Stage가 바뀌면 항상 False.  
> **PRETRAINED_CKPT는 Stage 2/3 모두 `checkpoints_ko/best_model_ko.pth`** — 직전 Stage가 덮어쓴 파일을 그대로 읽음.

---

### Stage 1

**결과:** _(실행 후 기록)_

### Stage 2

**결과:** _(실행 후 기록)_

### Stage 3

**결과:** _(실행 후 기록)_
