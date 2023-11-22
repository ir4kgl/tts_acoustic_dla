import numpy as np
import torch

from hw_tts.waveglow.inference import inference
import hw_tts.text
import hw_tts.audio
from hw_tts.synthesis.utils import get_WaveGlow

WaveGlow = get_WaveGlow()
WaveGlow = WaveGlow.cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def synthesis(model, phn, alpha=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha)["mel_output"]
    mel = mel.contiguous().transpose(1, 2)
    return inference(mel, WaveGlow)