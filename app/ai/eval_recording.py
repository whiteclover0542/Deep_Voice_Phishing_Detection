# =====================================================================
# 녹음 파일로 딥페이크 탐지 평가 (수정본 — AI/사람 확률 % + 시각화 + 표)
#
# 사용법:
#   단일 파일:  python eval_recording.py 내음성.wav
#   폴더 전체:  python eval_recording.py --folder 내폴더/
#   EER 측정:   python eval_recording.py --real 진짜폴더/ --fake 가짜폴더/
#
# 옵션:
#   --ckpt   체크포인트 경로 (기본: checkpoints_ko/best_model_ko.pth)
#   --tel    전화 채널 시뮬레이션 적용 (통화 녹음 테스트 시 권장)
#   --no-plot  차트/그래프 출력 비활성화
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

    try:
        wav, sr = sf.read(path, dtype='float32')
    except Exception:
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

        sos = butter(4, [300, 3400], btype='band', fs=SR, output='sos')
        wav = sosfilt(sos, wav).astype(np.float32)

        wav_8k = wav[::2]
        wav = np.repeat(wav_8k, 2)[:MAX_LEN].astype(np.float32)

        mu      = 255
        wav_c   = np.clip(wav, -1.0, 1.0)
        encoded = np.sign(wav_c) * np.log1p(mu * np.abs(wav_c)) / np.log1p(mu)
        quantized = np.round(encoded * 127) / 127
        wav = (np.sign(quantized) * (np.power(1.0 + mu, np.abs(quantized)) - 1.0) / mu).astype(np.float32)

        frame_len = int(SR * 0.02)
        for i in range(len(wav) // frame_len):
            if _rng.random() < 0.02:
                wav[i * frame_len:(i + 1) * frame_len] = 0.0

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
    x = torch.FloatTensor(wav).unsqueeze(0).unsqueeze(0).to(device)
    with autocast('cuda' if 'cuda' in device else 'cpu'):
        logits = model(x)
    probs = torch.softmax(logits, 1)[0]
    human_pct = probs[0].item() * 100   # 사람(진짜) 확률 %
    ai_pct    = probs[1].item() * 100   # AI(가짜) 확률 %
    return ai_pct / 100, human_pct, ai_pct  # score, human%, ai%


def classify(human_pct, ai_pct):
    if ai_pct >= 70:
        return 'AI (딥페이크)', ai_pct
    elif human_pct >= 70:
        return '사람 (진짜)', human_pct
    else:
        return '불확실', max(human_pct, ai_pct)


# ── 결과 출력 헬퍼 ────────────────────────────────────────────────────
def _bar(pct, width=20, fill='█', empty='░'):
    """텍스트 진행 바"""
    filled = int(round(pct / 100 * width))
    return fill * filled + empty * (width - filled)


def print_single_result(path, human_pct, ai_pct, label, confidence):
    """단일 파일 결과를 퍼센트 바와 함께 출력"""
    fname = os.path.basename(path)
    print(f'\n{"─"*55}')
    print(f'  파일   : {fname}')
    print(f'  판정   : {label}  (확신도 {confidence:.1f}%)')
    print(f'{"─"*55}')
    print(f'  사람   {_bar(human_pct)}  {human_pct:5.1f}%')
    print(f'  AI     {_bar(ai_pct)}  {ai_pct:5.1f}%')
    print(f'{"─"*55}')


def print_table(rows, title=''):
    """
    rows: list of dict with keys — fname, label, human_pct, ai_pct, confidence, correct
    """
    if title:
        print(f'\n{title}')

    col_w = [40, 14, 8, 8, 8, 4]
    header = f"  {'파일명':<{col_w[0]}} {'판정':<{col_w[1]}} {'사람%':>{col_w[2]}} {'AI%':>{col_w[3]}} {'확신도':>{col_w[4]}} {'정오':>{col_w[5]}}"
    sep    = '  ' + '─' * (sum(col_w) + len(col_w) * 1)

    print(sep)
    print(header)
    print(sep)

    for r in rows:
        correct_mark = '✓' if r.get('correct') is True else ('✗' if r.get('correct') is False else '')
        print(
            f"  {r['fname']:<{col_w[0]}.{col_w[0]}} "
            f"{r['label']:<{col_w[1]}.{col_w[1]}} "
            f"{r['human_pct']:>{col_w[2]}.1f} "
            f"{r['ai_pct']:>{col_w[3]}.1f} "
            f"{r['confidence']:>{col_w[4]}.1f} "
            f"{correct_mark:>{col_w[5]}}"
        )
    print(sep)


# ── 단일 파일 평가 ────────────────────────────────────────────────────
def eval_file(model, path, device, tel_aug=False):
    wav, _ = load_audio(path)
    score, human_pct, ai_pct = infer(model, wav, device, tel_aug)
    label, confidence = classify(human_pct, ai_pct)
    return score, human_pct, ai_pct, label, confidence


# ── 폴더 전체 평가 ────────────────────────────────────────────────────
AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.mp4', '.flac', '.ogg', '.aac'}

def eval_folder(model, folder, device, tel_aug=False, gt_label=None, show_plot=True):
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    ])
    if not files:
        print(f'  오디오 파일 없음: {folder}')
        return []

    rows   = []
    scores = []

    for path in files:
        try:
            score, human_pct, ai_pct, label, confidence = eval_file(model, path, device, tel_aug)
            fname   = os.path.basename(path)
            correct = None
            if gt_label is not None:
                pred    = 1 if score >= 0.5 else 0
                correct = (pred == gt_label)
            rows.append(dict(
                fname=fname, label=label,
                human_pct=human_pct, ai_pct=ai_pct,
                confidence=confidence, correct=correct,
                score=score, path=path
            ))
            scores.append(score)
        except Exception as e:
            print(f'  [오류] {os.path.basename(path)}: {e}')

    print_table(rows)

    ai_n    = sum(1 for r in rows if r['score'] >= 0.5)
    human_n = len(rows) - ai_n
    print(f'\n  총 {len(rows)}개  →  사람: {human_n}개 / AI: {ai_n}개')
    print(f'  평균 AI 확률: {np.mean([r["ai_pct"] for r in rows]):.1f}%')

    if show_plot and rows:
        _plot_folder(rows, folder)

    return [(r['path'], r['score']) for r in rows]


# ── EER 측정 ─────────────────────────────────────────────────────────
def eval_eer(model, real_dir, fake_dir, device, tel_aug=False, show_plot=True):
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    print(f'\n[진짜(사람) 음성]  {real_dir}')
    real_results = eval_folder(model, real_dir, device, tel_aug, gt_label=0, show_plot=False)

    print(f'\n[가짜(AI) 음성]  {fake_dir}')
    fake_results = eval_folder(model, fake_dir, device, tel_aug, gt_label=1, show_plot=False)

    if not real_results or not fake_results:
        print('평가 데이터 부족 — 종료')
        return

    labels      = [0] * len(real_results) + [1] * len(fake_results)
    scores_all  = [s for _, s in real_results] + [s for _, s in fake_results]
    real_scores = np.array([s for _, s in real_results])
    fake_scores = np.array([s for _, s in fake_results])

    fpr, tpr, _ = roc_curve(labels, scores_all, pos_label=1)
    eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1) * 100

    acc = sum(
        (1 if s >= 0.5 else 0) == l for s, l in zip(scores_all, labels)
    ) / len(labels) * 100

    real_ai_mean    = real_scores.mean() * 100
    fake_ai_mean    = fake_scores.mean() * 100

    print(f'\n{"═"*55}')
    print(f'  샘플 수   사람={len(real_results)}  AI={len(fake_results)}')
    print(f'  EER       {eer:.2f}%')
    print(f'  정확도    {acc:.2f}%')
    print(f'{"─"*55}')
    print(f'  사람 음성 → 평균 AI 확률  {real_ai_mean:.1f}%  (낮을수록 좋음)')
    print(f'  AI   음성 → 평균 AI 확률  {fake_ai_mean:.1f}%  (높을수록 좋음)')
    print(f'{"═"*55}')

    if show_plot:
        _plot_eer(real_scores, fake_scores, eer, os.path.dirname(os.path.abspath(real_dir)))


# ── 시각화: 폴더 결과 바 차트 ────────────────────────────────────────
def _plot_folder(rows, folder):
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

        n     = len(rows)
        fnames = [r['fname'][:28] + ('…' if len(r['fname']) > 28 else '') for r in rows]
        human  = [r['human_pct'] for r in rows]
        ai     = [r['ai_pct']    for r in rows]

        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.55 + 1.5)))
        y = np.arange(n)
        h = 0.38

        bars_h = ax.barh(y + h/2, human, h, color='#378ADD', alpha=0.8, label='사람 (진짜)')
        bars_a = ax.barh(y - h/2, ai,    h, color='#E24B4A', alpha=0.8, label='AI (가짜)')

        for bar, val in zip(bars_h, human):
            ax.text(min(val + 1, 98), bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=8, color='#0C447C')
        for bar, val in zip(bars_a, ai):
            ax.text(min(val + 1, 98), bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=8, color='#791F1F')

        ax.set_yticks(y)
        ax.set_yticklabels(fnames, fontsize=9)
        ax.set_xlabel('확률 (%)')
        ax.set_xlim(0, 110)
        ax.axvline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='임계값 50%')
        ax.set_title(f'AI/사람 판별 결과  ({os.path.basename(folder)})')
        ax.legend(loc='lower right', fontsize=9)
        ax.invert_yaxis()

        out_path = os.path.join(folder, 'eval_ai_human.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'\n  차트 저장: {out_path}')
    except Exception as e:
        print(f'  시각화 실패: {e}')


# ── 시각화: EER score 분포 ────────────────────────────────────────────
def _plot_eer(real_scores, fake_scores, eer, out_dir):
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

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # ── 왼쪽: score 분포 히스토그램
        ax = axes[0]
        ax.hist(real_scores * 100, bins=25, alpha=0.65, color='#378ADD',
                label=f'사람(진짜)  n={len(real_scores)}')
        ax.hist(fake_scores * 100, bins=25, alpha=0.65, color='#E24B4A',
                label=f'AI(가짜)  n={len(fake_scores)}')
        ax.axvline(50, color='black', linestyle='--', alpha=0.5, label='임계값 50%')
        ax.set_xlabel('AI 확률 (%)')
        ax.set_ylabel('샘플 수')
        ax.set_title(f'AI 확률 분포  (EER={eer:.2f}%)')
        ax.legend(fontsize=9)

        # ── 오른쪽: 평균 AI/사람 확률 막대 비교
        ax2 = axes[1]
        categories = ['사람 음성', 'AI 음성']
        human_means = [
            (1 - real_scores.mean()) * 100,
            (1 - fake_scores.mean()) * 100,
        ]
        ai_means = [
            real_scores.mean() * 100,
            fake_scores.mean() * 100,
        ]
        x = np.arange(2)
        w = 0.35
        ax2.bar(x - w/2, human_means, w, label='사람 확률%', color='#378ADD', alpha=0.8)
        ax2.bar(x + w/2, ai_means,    w, label='AI 확률%',   color='#E24B4A', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 110)
        ax2.set_ylabel('평균 확률 (%)')
        ax2.set_title('그룹별 평균 AI/사람 확률')
        ax2.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax2.legend(fontsize=9)

        for rect, val in zip(
            list(ax2.patches[:2]) + list(ax2.patches[2:]),
            human_means + ai_means
        ):
            ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        out_path = os.path.join(out_dir, 'eval_score_dist.png')
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  차트 저장: {out_path}')
    except Exception as e:
        print(f'  시각화 실패: {e}')


# ── 간편 실행 함수 ────────────────────────────────────────────────────
def predict(path, tel=False, ckpt=None):
    """
    단일 파일 판별. 코드 안에서 직접 호출:
        predict(r'D:\경로\파일.wav')
        predict(r'D:\경로\통화.m4a', tel=True)   # 통화 녹음
    """
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = ckpt or os.path.join(BASE_DIR, 'checkpoints_ko', 'best_model_ko.pth')
    model     = load_model(ckpt_path, device)
    score, human_pct, ai_pct, label, confidence = eval_file(model, path, device, tel)
    print_single_result(path, human_pct, ai_pct, label, confidence)
    return label, human_pct, ai_pct


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='딥페이크 탐지 평가 (AI/사람 확률 % 출력)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            '예시:\n'
            '  단일 파일:  python eval_recording.py 내음성.wav\n'
            '  폴더 전체:  python eval_recording.py --folder ./recordings/\n'
            '  EER 측정:   python eval_recording.py --real ./real/ --fake ./fake/\n'
            '  통화 파일:  python eval_recording.py 통화.m4a --tel\n'
            '  차트 끄기:  python eval_recording.py --folder ./recordings/ --no-plot'
        )
    )
    parser.add_argument('file',       nargs='?', help='단일 오디오 파일')
    parser.add_argument('--folder',   help='폴더 전체 평가')
    parser.add_argument('--real',     help='진짜(사람) 음성 폴더 (EER 측정용)')
    parser.add_argument('--fake',     help='가짜(AI) 음성 폴더 (EER 측정용)')
    parser.add_argument('--ckpt',     default=os.path.join(BASE_DIR, 'checkpoints_ko', 'best_model_ko.pth'),
                        help='체크포인트 경로')
    parser.add_argument('--tel',      action='store_true',
                        help='전화 채널 시뮬레이션 적용')
    parser.add_argument('--no-plot',  action='store_true',
                        help='차트/그래프 출력 비활성화')
    args = parser.parse_args()

    show_plot = not args.no_plot
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}  |  tel_aug: {args.tel}\n')

    model = load_model(args.ckpt, device)

    # ── 단일 파일
    if args.file:
        score, human_pct, ai_pct, label, confidence = eval_file(
            model, args.file, device, args.tel
        )
        print_single_result(args.file, human_pct, ai_pct, label, confidence)

        # 단일 파일도 미니 바 차트 저장
        if show_plot:
            out_dir = os.path.dirname(os.path.abspath(args.file)) or '.'
            _plot_folder(
                [dict(fname=os.path.basename(args.file),
                      human_pct=human_pct, ai_pct=ai_pct,
                      label=label, score=score)],
                out_dir
            )

    # ── 폴더 전체
    elif args.folder:
        print(f'[폴더 평가]  {args.folder}\n')
        eval_folder(model, args.folder, device, args.tel, show_plot=show_plot)

    # ── EER 측정
    elif args.real and args.fake:
        eval_eer(model, args.real, args.fake, device, args.tel, show_plot=show_plot)

    else:
        parser.print_help()


if __name__ == '__main__':
    predict(r'D:\IT\AiHuman4_SR\project_1\Real_time_Voice_Phishing_Detection\app\ai\진짜_통화_3.m4a', tel=True)