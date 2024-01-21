import os
import numpy as np
import librosa
import hw_tts.audio as Audio
import torch

wav_directory = './wavs/'
mel_directory = './mels/'
energy_directory = './energy/'


if not os.path.exists(energy_directory):
    os.makedirs(energy_directory)

def extract_energy(audio_path):
    wav, _ = librosa.load(audio_path)
    wav = torch.FloatTensor(wav)
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav)

    return energy * 1000

wav_files = [f for f in os.listdir(wav_directory) if f.endswith(".wav")]
wav_files.sort()

min_energy = 100.
max_energy = 0.
for file_id, filename in enumerate(wav_files):

    if filename.endswith(".wav"):
        audio_path = os.path.join(wav_directory, filename)
        energy = extract_energy(audio_path)
        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())
        np.save(os.path.join(energy_directory, "ljspeech-energy-{:05d}.npy".format(file_id+1)), energy)

print(min_energy)
print(max_energy)
