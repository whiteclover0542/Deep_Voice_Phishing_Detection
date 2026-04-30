# =====================================================================
# 녹음 파일로 딥페이크 탐지 평가
#
# 사용법:
#   단일 파일:  python eval_recording.py 내음성.wav
#   폴더 전체:  python eval_recording.py --folder 내폴더/
#   EER 측정:   python eval_recording.py --real 진짜폴더/ --fake 가짜폴더/
#
# 옵션:
#   --ckpt   체크포인트 경로 (기본: checkpoints_ko/best_model_ko.pth)
#   --tel    전화 채널 시뮬레이션 적용 (통화 녹음 테스트 시 권장)

# =====================================================================

import os
import sys
import argparse
import numpy as np
import torch
from torch.amp import autocast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from model import RawNet2

SR      = 16000
MAX_LEN = SR * 4  # 4초


# ── 오디오 로드 ───────────────────────────────────────────────────────
def load_audio(path):
    import soundfile as sf

    # soundfile로 직접 읽기 (wav, flac 등)
    try:
        wav, sr = sf.read(path, dtype='float32')
    except Exception:
        # m4a, mp4, mp3 등 → ffmpeg로 변환
        # Windows에서 한글/유니코드 경로가 ffmpeg에 전달되면 Invalid argument 오류 발생
        # 입력 파일을 ASCII 이름의 임시 경로로 먼저 복사해 우회
        import subprocess, tempfile, shutil
        ext = os.path.splitext(path)[1] or '.tmp'
        src_fd, src_tmp = tempfile.mkstemp(suffix=ext)
        os.close(src_fd)
        out_fd, out_tmp = tempfile.mkstemp(suffix='.wav')
        os.close(out_fd)
        try:
            shutil.copy2(os.path.abspath(path), src_tmp)
            ret = subprocess.run(
                ['ffmpeg', '-y', '-i', src_tmp, '-ac', '1', '-ar', str(SR), out_tmp],
                capture_output=True
            )
            if ret.returncode != 0:
                raise RuntimeError(ret.stderr.decode('utf-8', errors='ignore'))
            wav, sr = sf.read(out_tmp, dtype='float32')
        finally:
            for p in [src_tmp, out_tmp]:
                if os.path.exists(p):
                    os.unlink(p)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != SR:
        try:
            import resampy
            wav = resampy.resample(wav, sr, SR)
        except ImportError:
            from scipy.signal import resample as sp_resample
            wav = sp_resample(wav, int(len(wav) * SR / sr)).astype(np.float32)

    return wav.astype(np.float32), SR


# ── 전처리 ────────────────────────────────────────────────────────────
def preprocess(wav, tel_aug=False):
    if len(wav) < MAX_LEN:
        wav = np.pad(wav, (0, MAX_LEN - len(wav)))
    else:
        start = (len(wav) - MAX_LEN) // 2
        wav = wav[start: start + MAX_LEN]

    if tel_aug:
        from scipy.signal import butter, sosfilt
        import random as _rng

        # 1. 대역통과 필터
        sos = butter(4, [300, 3400], btype='band', fs=SR, output='sos')
        wav = sosfilt(sos, wav).astype(np.float32)

        # 2. 8kHz 다운샘플 → 업샘플
        wav_8k = wav[::2]
        wav = np.repeat(wav_8k, 2)[:MAX_LEN].astype(np.float32)

        # 3. G.711 μ-law 코덱 시뮬레이션
        mu      = 255
        wav_c   = np.clip(wav, -1.0, 1.0)
        encoded = np.sign(wav_c) * np.log1p(mu * np.abs(wav_c)) / np.log1p(mu)
        quantized = np.round(encoded * 127) / 127
        wav = (np.sign(quantized) * (np.power(1.0 + mu, np.abs(quantized)) - 1.0) / mu).astype(np.float32)

        # 4. 패킷 손실 (20ms 프레임, 2% 확률)
        frame_len = int(SR * 0.02)
        for i in range(len(wav) // frame_len):
            if _rng.random() < 0.02:
                wav[i * frame_len:(i + 1) * frame_len] = 0.0

        # 5. AGC
        peak = np.abs(wav).max()
        if peak > 1e-8:
            wav = wav / peak * 0.8

    return wav[:MAX_LEN].astype(np.float32)


# ── 모델 로드 ─────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'체크포인트 없음: {ckpt_path}')

    model = RawNet2().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    print(f'[모델] {ckpt_path}')
    if isinstance(ckpt, dict):
        print(f'       Stage={ckpt.get("stage","?")}  '
              f'Epoch={ckpt.get("epoch","?")}  '
              f'Best EER={ckpt.get("best_eer","?")}%')
    return model


# ── 단일 추론 ─────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model, wav_np, device, tel_aug=False):
    wav = preprocess(wav_np, tel_aug=tel_aug)
    x = torch.FloatTensor(wav).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)
    with autocast('cuda' if 'cuda' in device else 'cpu'):
        logits = model(x)
    score = torch.softmax(logits, 1)[0, 1].item()
    return score


def classify(score):
    if score >= 0.7:
        return '가짜 (딥페이크)', score
    elif score <= 0.3:
        return '진짜', 1 - score
    else:
        return '불확실', max(score, 1 - score)


# ── 단일 파일 평가 ────────────────────────────────────────────────────
def eval_file(model, path, device, tel_aug=False):
    wav, _ = load_audio(path)
    score = infer(model, wav, device, tel_aug)
    label, confidence = classify(score)
    return score, label, confidence


# ── 폴더 전체 평가 ────────────────────────────────────────────────────
AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.mp4', '.flac', '.ogg', '.aac'}

def eval_folder(model, folder, device, tel_aug=False, gt_label=None):
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    ])
    if not files:
        print(f'  오디오 파일 없음: {folder}')
        return []

    results = []
    for path in files:
        try:
            score, label, conf = eval_file(model, path, device, tel_aug)
            fname = os.path.basename(path)
            correct = ''
            if gt_label is not None:
                pred = 1 if score >= 0.5 else 0
                correct = '✓' if pred == gt_label else '✗'
            print(f'  {correct} {fname:<40}  {label:<14}  score={score:.4f}  확신={conf:.1%}')
            results.append((path, score))
        except Exception as e:
            print(f'  [오류] {os.path.basename(path)}: {e}')
    return results


# ── EER 측정 ─────────────────────────────────────────────────────────
def eval_eer(model, real_dir, fake_dir, device, tel_aug=False):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    print(f'\n[진짜 음성]  {real_dir}')
    real_results = eval_folder(model, real_dir, device, tel_aug, gt_label=0)

    print(f'\n[가짜 음성]  {fake_dir}')
    fake_results = eval_folder(model, fake_dir, device, tel_aug, gt_label=1)

    if not real_results or not fake_results:
        print('평가 데이터 부족 — 종료')
        return

    labels = [0] * len(real_results) + [1] * len(fake_results)
    scores = [s for _, s in real_results] + [s for _, s in fake_results]
    real_scores = np.array([s for _, s in real_results])
    fake_scores = np.array([s for _, s in fake_results])

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1) * 100

    acc = sum(
        (1 if s >= 0.5 else 0) == l for s, l in zip(scores, labels)
    ) / len(labels) * 100

    print(f'\n{"="*55}')
    print(f'  샘플 수   진짜={len(real_results)}  가짜={len(fake_results)}')
    print(f'  EER       {eer:.2f}%')
    print(f'  정확도    {acc:.2f}%')
    print(f'  진짜 score  평균={real_scores.mean():.4f}  std={real_scores.std():.4f}')
    print(f'  가짜 score  평균={fake_scores.mean():.4f}  std={fake_scores.std():.4f}')
    print(f'{"="*55}')

    _save_dist(real_scores, fake_scores, eer, os.path.dirname(os.path.abspath(real_dir)))


def _save_dist(real_scores, fake_scores, eer, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        font_path = r'C:\Windows\Fonts\malgun.ttf'
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(real_scores, bins=30, alpha=0.6, color='steelblue', label=f'진짜 (n={len(real_scores)})')
        ax.hist(fake_scores, bins=30, alpha=0.6, color='tomato',    label=f'가짜 (n={len(fake_scores)})')
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='임계값 0.5')
        ax.set_xlabel('Score (가짜 확률)')
        ax.set_ylabel('샘플 수')
        ax.set_title(f'외부 평가 Score 분포  (EER={eer:.2f}%)')
        ax.legend()

        out_path = os.path.join(out_dir, 'eval_score_dist.png')
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f'  분포 저장: {out_path}')
    except Exception as e:
        print(f'  시각화 실패: {e}')


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='딥페이크 탐지 평가',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            '예시:\n'
            '  단일 파일:  python eval_recording.py 내음성.wav\n'
            '  폴더 전체:  python eval_recording.py --folder ./recordings/\n'
            '  EER 측정:   python eval_recording.py --real ./real/ --fake ./fake/\n'
            '  통화 파일:  python eval_recording.py 통화.m4a --tel'
        )
    )
    parser.add_argument('file',     nargs='?', help='단일 오디오 파일')
    parser.add_argument('--folder', help='폴더 전체 평가')
    parser.add_argument('--real',   help='진짜 음성 폴더 (EER 측정용)')
    parser.add_argument('--fake',   help='가짜 음성 폴더 (EER 측정용)')
    parser.add_argument('--ckpt',   default=os.path.join(BASE_DIR, 'checkpoints_ko', 'best_model_ko.pth'),
                        help='체크포인트 경로')
    parser.add_argument('--tel',    action='store_true',
                        help='전화 채널 시뮬레이션 적용 (통화 녹음 테스트 시 권장)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}  |  tel_aug: {args.tel}\n')

    model = load_model(args.ckpt, device)

    # ── 단일 파일
    if args.file:
        score, label, conf = eval_file(model, args.file, device, args.tel)
        print(f'\n파일  : {args.file}')
        print(f'결과  : {label}')
        print(f'Score : {score:.4f}   (0=진짜 / 1=가짜)')
        print(f'확신도: {conf:.1%}')

    # ── 폴더 전체
    elif args.folder:
        print(f'[폴더 평가]  {args.folder}\n')
        results = eval_folder(model, args.folder, device, args.tel)
        if results:
            scores = np.array([s for _, s in results])
            real_n = (scores < 0.5).sum()
            fake_n = (scores >= 0.5).sum()
            print(f'\n총 {len(results)}개  →  진짜: {real_n}개 / 가짜: {fake_n}개')
            print(f'평균 score: {scores.mean():.4f}')

    # ── EER 측정
    elif args.real and args.fake:
        eval_eer(model, args.real, args.fake, device, args.tel)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
