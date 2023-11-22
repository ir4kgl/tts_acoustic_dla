
import logging

import torch
from torch.utils.data import Dataset

import time
import os
from tqdm import tqdm
import numpy as np

from hw_tts.utils import ROOT_PATH
from hw_tts.datasets.text import text_to_sequence

from pathlib import Path

import sys
sys.path.append('.')


logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)
        return txt


class LJSpeechDataset(Dataset):
    def __init__(self, data_dir, alignments_dir=None, mels_dir=None, pitch_dir=None, energy_dir=None):
        self._data_dir_ = Path(data_dir)
        if alignments_dir is None:
            alignments_dir = self._data_dir_ / "alignments/"
        self._alignments_dir_ = alignments_dir
        if mels_dir is None:
            mels_dir = self._data_dir_ / "mels/"
        self._mels_dir_ = mels_dir
        if pitch_dir is None:
            pitch_dir = self._data_dir_ / "pitch/"
        self._pitch_dir_ = pitch_dir
        if energy_dir is None:
            energy_dir = self._data_dir_ / "energy/"
        self._energy_dir_ = energy_dir
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

            pitch_gt_name = os.path.join(
                self._pitch_dir_, "ljspeech-pitch-%05d.npy" % (i+1))
            pitch_gt_target = np.load(pitch_gt_name)

            energy_gt_name = os.path.join(
                self._pitch_dir_, "ljspeech-energy-%05d.npy" % (i+1))
            energy_gt_target = np.load(energy_gt_name)

            duration = np.load(os.path.join(
                self._alignments_dir_, str(i)+".npy"))
            character = text[i][0:len(text[i])-1]
            character = np.array(
                text_to_sequence(character, ['english_cleaners']))

            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            mel_gt_target = torch.from_numpy(mel_gt_target)
            pitch_gt_target = torch.from_numpy(pitch_gt_target)
            energy_gt_target = torch.from_numpy(energy_gt_target)

            buffer.append({"text": character, "duration": duration,
                           "mel_target": mel_gt_target, "energy": energy_gt_target, "pitch": pitch_gt_target})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end-start))

        return buffer

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
