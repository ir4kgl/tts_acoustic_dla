import torch
import torch.nn as nn


class FastSpeechLoss():
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, batch):
        mel_loss = self.l1_loss(batch["mel_output"], batch["mel_target"])

        duration_predictor_loss = self.mse_loss(batch["duration_predictor_output"],
                                               batch["duration"].float())
        pitch_predictor_loss = self.mse_loss(batch["pitch_predictor_output"],
                                             batch["pitch"].float())
        energy_predictor_loss = self.mse_loss(batch["energy_predictor_output"],
                                              batch["energy"].float())
        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss
