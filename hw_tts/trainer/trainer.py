import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import MetricTracker
from hw_tts.trainer.eval_texts import EVAL_DATA
from hw_tts.synthesis.synthesis import synthesis


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.len_epoch = len(self.train_dataloader)

        self.lr_scheduler = lr_scheduler
        self.log_step = 10

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "dp_loss", "pitch_loss", "energy_loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["text", "duration", "mel_pos", "src_pos"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].long().to(device)
        batch["mel_target"] = batch["mel_target"].float().to(device)
        batch["pitch"] = batch["pitch"].float().to(device)
        batch["energy"] = batch["energy"].float().to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(
                ), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

        log = last_train_metrics
        self._evaluation_epoch()
        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(batch)
        batch.update(outputs)

        batch["mel_loss"], batch["dp_loss"], batch["pitch_loss"], batch["energy_loss"] = self.criterion.forward(
            batch)
        batch["loss"] = batch["mel_loss"] + batch["dp_loss"] + batch["pitch_loss"] 
        # + batch["energy_loss"]
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        metrics.update("mel_loss", batch["mel_loss"].item())
        metrics.update("dp_loss", batch["dp_loss"].item())
        metrics.update("pitch_loss", batch["pitch_loss"].item())
        metrics.update("energy_loss", batch["energy_loss"].item())

        return batch

    def _evaluation_epoch(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        configurations = [
            (1, 1, 1),
            (1, 1, 1.2),
            (1, 1, 0.8),
            (1, 1.2, 1),
            (1, 0.8, 1),
            (1.2, 1, 1),
            (0.8, 1, 1),
            (1.2, 1.2, 1.2),
            (0.8, 0.8, 0.8),
        ]
        for i, phn in enumerate(EVAL_DATA):
            for (alpha, c_pitch, c_energy) in configurations:
                audio = synthesis(self.model, phn, alpha=alpha,
                                  c_pitch=c_pitch, c_energy=c_energy)
                self.writer.add_audio("synthesised_audio_{}_{}_{}_{}".format(i, alpha, c_pitch, c_energy),
                                      audio, sample_rate=22050)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu()
                 for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(
                f"{metric_name}", metric_tracker.avg(metric_name))
