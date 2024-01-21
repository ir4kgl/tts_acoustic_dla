import os
import numpy as np
import librosa
import pyworld as pw


wav_directory = './wavs/'
pitch_directory = './pitch/'

SR=22050
HOP_LENGTH=256

if not os.path.exists(pitch_directory):
    os.makedirs(pitch_directory)

def extract_pitch(audio_path):
    wav, _ = librosa.load(audio_path)
    pitch, t = pw.dio(
        wav.astype(np.float64),
        SR,
        frame_period=HOP_LENGTH / SR * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, SR)

    return pitch

wav_files = [f for f in os.listdir(wav_directory) if f.endswith(".wav")]
wav_files.sort()

min_pitch = 100.
max_pitch = 0.

for file_id, filename in enumerate(wav_files):
    if filename.endswith(".wav"):
        audio_path = os.path.join(wav_directory, filename)
        pitch = extract_pitch(audio_path)
        min_pitch = min(min_pitch, pitch.min())
        max_pitch = max(max_pitch, pitch.max())
        np.save(os.path.join(pitch_directory, "ljspeech-pitch-{:05d}.npy".format(file_id+1)), pitch)


print(min_pitch)
print(max_pitch)