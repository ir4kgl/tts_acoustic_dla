
import logging

import torch
from torch.utils.data import Dataset

import time
import os
from tqdm import tqdm
import numpy as np

from hw_tts.utils import ROOT_PATH
from hw_tts.datasets.text import text_to_sequence


import sys
sys.path.append('.')


logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    "train_texts": "https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx",
    "alignments": "https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip",
    "mels": "https://docs.google.com/uc?export=download&confirm=t&id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)
        return txt


class LJSpeechDataset(Dataset):
    def __init__(self, data_dir, alignments_dir=None, mels_dir=None):

        self._data_dir_ = data_dir
        if alignments_dir is None:
            alignments_dir = os.path.join(data_dir, "alignments/")
        self._alignments_dir_ = alignments_dir
        if mels_dir is None:
            mels_dir = os.path.join(data_dir, "mels/")
        self._mels_dir_ = mels_dir
        self.buffer = self.get_data_to_buffer()
        self.length_dataset = len(self.buffer)

    def get_data_to_buffer(self):
        buffer = list()
        text = process_text(self._data_dir_ / "train.txt")

        start = time.perf_counter()
        for i in tqdm(range(len(text))):

            mel_gt_name = os.path.join(
                self._mels_dir_, "ljspeech-mel-%05d.npy" % (i+1))
            mel_gt_target = np.load(mel_gt_name)
            duration = np.load(os.path.join(
                self._alignments_dir_, str(i)+".npy"))
            character = text[i][0:len(text[i])-1]
            character = np.array(
                text_to_sequence(character, ['english_cleaners']))

            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            mel_gt_target = torch.from_numpy(mel_gt_target)

            buffer.append({"text": character, "duration": duration,
                           "mel_target": mel_gt_target})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end-start))

        return buffer

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
