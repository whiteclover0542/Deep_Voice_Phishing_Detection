import os
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
import io

FIXED_LEN = 64600  # 16kHz 기준 약 4초


class ASVspoof2019LA(Dataset):
    PROTOCOLS = {
        'train': 'ASVspoof2019.LA.cm.train.trn.txt',
        'dev':   'ASVspoof2019.LA.cm.dev.trl.txt',
        'eval':  'ASVspoof2019.LA.cm.eval.trl.txt',
    }

    def __init__(self, data_root, split='train', max_samples=None):  # max_samples 추가
        """
        data_root : LA 폴더 경로 (예: /content/drive/MyDrive/data/LA)
        split     : 'train' | 'dev' | 'eval'
        """
        assert split in self.PROTOCOLS, f"split은 {list(self.PROTOCOLS)} 중 하나여야 합니다"

        audio_base = os.path.join(data_root, f'ASVspoof2019_LA_{split}')
        # flac 파일이 하위 flac/ 폴더에 있는 경우와 바로 있는 경우 모두 처리
        flac_subdir = os.path.join(audio_base, 'flac')
        self.audio_dir = flac_subdir if os.path.isdir(flac_subdir) else audio_base

        proto_path = os.path.join(
            data_root, 'ASVspoof2019_LA_cm_protocols', self.PROTOCOLS[split]
        )

        # 수정 코드 - 파일 존재 여부 확인 후 추가
        self.samples = []
        missing = 0
        with open(proto_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                file_id = parts[1]
                label = 0 if parts[-1] == 'bonafide' else 1
                path = os.path.join(self.audio_dir, f'{file_id}.flac')
                if os.path.exists(path):
                    self.samples.append((file_id, label))
                else:
                    missing += 1


        if max_samples is not None:
            # 1. 클래스별로 분리
            genuine = [s for s in self.samples if s[1] == 0]
            spoof   = [s for s in self.samples if s[1] == 1]
            
            # [추가] 2. 랜덤하게 섞기 (numpy 활용)
            # 실험의 재현성을 위해 시드(seed)를 고정하는 것이 좋습니다.
            np.random.seed(42) 
            np.random.shuffle(genuine)
            np.random.shuffle(spoof)
            
            # 3. 정해진 양만큼 자르기
            half = max_samples // 2
            self.samples = genuine[:half] + spoof[:half]
            
            # [추가] 4. 합쳐진 리스트를 한 번 더 섞기 
            # (학습 시 첫 번째 배치에 정답만 몰려있는 것을 방지)
            np.random.shuffle(self.samples)

        print(f'[{split}] 로드: {len(self.samples)}개, 누락: {missing}개')  # ← 이 줄 추가

        # ── 추가: 전체 데이터를 RAM에 캐싱 ──────────────────────────
        #print(f'[{split}] RAM 캐싱 중...')
        #self.cache = [self._load_wav(fid) for fid, _ in self.samples]
        #print(f'[{split}] 캐싱 완료')

    def _load_wav(self, file_id):
        path = os.path.join(self.audio_dir, f'{file_id}.flac')
        with open(path, 'rb') as f:
            wav, _ = sf.read(io.BytesIO(f.read()), dtype='float32')

        if len(wav) >= FIXED_LEN:
            wav = wav[:FIXED_LEN]
        else:
            wav = np.pad(wav, (0, FIXED_LEN - len(wav)))

        return torch.FloatTensor(wav).unsqueeze(0)  # (1, FIXED_LEN)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id, label = self.samples[idx]
        #return self.cache[idx], label   # 디스크 대신 RAM에서 읽기
        return self._load_wav(file_id), label   # 디스크에서 읽기
