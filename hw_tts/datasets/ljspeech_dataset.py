from hw_tts.datasets.text import text_to_sequence

from speechbrain.utils.data_utils import download_file
from hw_tts.utils import ROOT_PATH
from pathlib import Path
from curses.ascii import isascii
import shutil
import logging
import json

import pathlib
import random
import itertools
from tqdm import tqdm_notebook

from IPython import display
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import distributions
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import MelSpectrogram
import math
import time
import os
import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from dataclasses import dataclass
from collections import OrderedDict

from hw_tts.utils import ROOT_PATH
import gdown

import sys
sys.path.append('.')


logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    "alignments": "https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip",
    "mels": "https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j",
    "waveglow": "https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx",
    "train_texts": "https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded
    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_1D_tensor(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded
    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")
        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]
    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")
        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]
    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])
    return output


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)
        return txt


class BufferDataset(Dataset):
    def __init__(self, data_dir=None, alignments_dir=None, mels_dir=None):
        if data_dir is None:
            pass

        self._data_dir_ = data_dir
        if alignments_dir is None:
            alignments_dir = ROOT_PATH / "data" / "datasets" / "LJspeech" / "alignments"
            alignments_dir.mkdir()
            self._alignments_dir_ = alignments_dir
            arch_path = self._alignments_dir_ / f"alignments.zip"
            print(f"Loading alignments")
            download_file(URL_LINKS["alignments"], arch_path)
            shutil.unpack_archive(arch_path, self._alignments_dir_)
            for fpath in (self._alignments_dir_ / "alignments").iterdir():
                shutil.move(str(fpath), str(
                    self._alignments_dir_ / fpath.name))
            os.remove(str(arch_path))
            shutil.rmtree(str(self._alignments_dir_ / "alignments"))
        self._alignments_dir_ = alignments_dir

        if mels_dir is None:
            mels_dir = ROOT_PATH / "data" / "datasets" / "LJspeech" / "mels"
            mels_dir.mkdir()
            self._mels_dir_ = mels_dir
            arch_path = self._mels_dir_ / f"mel.tar.gz"
            print(f"Loading mels")
            gdown.download(URL_LINKS["mels"], arch_path, quiet=True)
            shutil.unpack_archive(arch_path, self._mels_dir_)
            for fpath in (self._mels_dir_ / "mels").iterdir():
                shutil.move(str(fpath), str(
                    self._mels_dir_ / fpath.name))
            os.remove(str(arch_path))
            shutil.rmtree(str(self._mels_dir_ / "mels"))
        self._mels_dir_ = mels_dir

        self.buffer = self.get_data_to_buffer()
        self.length_dataset = len(self.buffer)

    def get_data_to_buffer(self):
        buffer = list()
        text = process_text(self._data_dir_)

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


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out


def collate_fn_tensor(batch):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    batch_expand_size = 32
    real_batchsize = batchsize // batch_expand_size

    cut_list = list()
    for i in range(batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output
