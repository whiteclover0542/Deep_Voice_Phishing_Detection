import os
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf

FIXED_LEN = 64600  # 16kHz 기준 약 4초


class ASVspoof2019LA(Dataset):
    PROTOCOLS = {
        'train': 'ASVspoof2019.LA.cm.train.trn.txt',
        'dev':   'ASVspoof2019.LA.cm.dev.trl.txt',
        'eval':  'ASVspoof2019.LA.cm.eval.trl.txt',
    }

    def __init__(self, data_root, split='train'):
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

        self.samples = []
        with open(proto_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # 형식: SPEAKER_ID  FILE_ID  -  ATTACK_TYPE  LABEL
                file_id = parts[1]
                label = 0 if parts[-1] == 'genuine' else 1
                self.samples.append((file_id, label))

    def _load_wav(self, file_id):
        path = os.path.join(self.audio_dir, f'{file_id}.flac')
        wav, _ = sf.read(path, dtype='float32')

        if len(wav) >= FIXED_LEN:
            wav = wav[:FIXED_LEN]
        else:
            wav = np.pad(wav, (0, FIXED_LEN - len(wav)))

        return torch.FloatTensor(wav).unsqueeze(0)  # (1, FIXED_LEN)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id, label = self.samples[idx]
        return self._load_wav(file_id), label
