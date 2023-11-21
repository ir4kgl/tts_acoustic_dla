import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, batch):
        mel_loss = self.mse_loss(batch["mel_output"], batch["mel_target"])

        duration_predictor_loss = self.l1_loss(batch["duration_predictor_output"],
                                               batch["duration"].float())

        return mel_loss, duration_predictor_loss
