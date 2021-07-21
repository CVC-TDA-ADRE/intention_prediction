import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
from torch.functional import Tensor
import torchmetrics
from collections import deque

from typing import Dict

from utils.utils import video_to_stream
from utils.visualization import VideoVisualizer
from models.x3d import X3D
from models.slow_r50 import SlowR50


class IntentionPredictor(pl.LightningModule):
    def __init__(self, data_kwargs, training_kwargs, **model_kwargs) -> None:
        super(IntentionPredictor, self).__init__()

        self.training_kwargs = training_kwargs
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.fps = data_kwargs["data_fps"]
        self.lr = self.training_kwargs["lr"]
        self.model_type = data_kwargs["dataset_type"]
        self.checkpoints = deque()

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()
        self.val_precision = torchmetrics.Precision()
        self.train_recall = torchmetrics.Recall()
        self.val_recall = torchmetrics.Recall()
        self.train_F1 = torchmetrics.F1()
        self.val_F1 = torchmetrics.F1()

        if training_kwargs["model_to_use"] == "x3d":
            self.model = X3D(
                model_type=data_kwargs["dataset_type"],
                crop_size=data_kwargs["resize"],
                clip_length=data_kwargs["input_seq_size"],
                **model_kwargs,
            )
        elif training_kwargs["model_to_use"] == "slow":
            self.model = SlowR50(
                crop_size=data_kwargs["resize"],
                clip_length=data_kwargs["input_seq_size"],
                model_type=data_kwargs["dataset_type"],
                **model_kwargs,
            )
        print(self.model)

        # if self.logger is not None:
        #     self.logger.experiment.watch(self.model, log="all")
        class_names = {0: "not_crossing", 1: "crossing"}
        self.visualization = VideoVisualizer(
            num_classes=1, class_names=class_names, thres=0.5, mode="binary"
        )

    def forward(self, x, boxes=None) -> Tensor:
        return self.model(x, boxes)

    def configure_optimizers(self):

        if self.training_kwargs["optimizer"] == "adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.lr,
                betas=self.training_kwargs["betas"],
            )
        elif self.training_kwargs["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.lr,
                betas=self.training_kwargs["betas"],
            )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.training_kwargs["epochs"], last_epoch=-1
        )
        return [optimizer], [scheduler]

    @torch.no_grad()
    def save_video(self, sample_clip, sig_preds, sample_boxes, labels):

        if self.model_type == "detection":
            sample_preds = sig_preds[: len(sample_boxes)].cpu()
            preview_video = self.visualization.draw_clip_range(
                sample_clip, sample_preds, sample_boxes
            )
            self.logger.experiment.log(
                {
                    "train/video": wandb.Video(
                        video_to_stream(np.array(preview_video), fps=self.fps),
                        format="mp4",
                        caption=f"True label: {labels[: len(sample_boxes)].cpu().numpy()}",
                    ),
                    "global_step": self.global_step,
                }
            )
        else:
            pred = 1.0 if sig_preds[0].detach() > 0.5 else 0.0
            self.logger.experiment.log(
                {
                    "train/video": wandb.Video(
                        video_to_stream(np.array(sample_clip), fps=self.fps),
                        format="mp4",
                        caption=f"True label: {labels[0].detach()}, Pred: {pred} ({sig_preds[0].detach():.2f})",
                    ),
                    "global_step": self.global_step,
                }
            )

    def training_step(self, batch, batch_idx) -> Tensor:
        clip, boxes, labels = batch["clip"], batch["boxes"], batch["label"]
        preds = self(clip, boxes)
        loss = self.bce_loss(preds, labels.float())

        # Metrics
        sig_preds = torch.sigmoid(preds.detach())
        acc = self.train_accuracy(sig_preds, labels)
        recall = self.train_recall(sig_preds, labels)
        precision = self.train_precision(sig_preds, labels)
        f1 = self.train_F1(sig_preds, labels)

        # Log loss and metric
        self.log("train/loss", loss)
        self.log("train/accuracy", acc)
        self.log("train/recall", recall)
        self.log("train/precision", precision)
        self.log("train/F1", f1)

        # Log video
        if self.logger is not None and (
            batch_idx % self.training_kwargs["video_every"] == 0
        ):
            sample_clip = batch["original_clip"][0].cpu() / 255.0
            sample_boxes = batch["original_boxes"][0]
            self.save_video(sample_clip, sig_preds, sample_boxes, labels)

        return loss

    def validation_step(self, batch, batch_idx) -> Dict:
        clip, boxes, labels = batch["clip"], batch["boxes"], batch["label"]
        preds = self(clip, boxes)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, labels.float())
        sig_preds = torch.sigmoid(preds)

        # Metrics
        recall = self.val_recall(sig_preds, labels)
        precision = self.val_precision(sig_preds, labels)
        f1 = self.val_F1(sig_preds, labels)
        acc = self.val_accuracy(sig_preds, labels)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/acc", acc, on_step=True, on_epoch=True)
        self.log("val/recall", recall, on_step=True, on_epoch=True)
        self.log("val/precision", precision, on_step=True, on_epoch=True)
        self.log("val/f1", f1, on_step=True, on_epoch=True)

        return loss

    # def training_epoch_end(self, outs):
    #     # additional log mean accuracy at the end of the epoch
    #     self.log("train/acc_epoch", self.train_accuracy.compute())

    # def validation_epoch_end(self, outs):
    #     # additional log mean accuracy at the end of the epoch
    #     self.log("val/acc_epoch", self.val_accuracy.compute())

    # def training_epoch_end(self, outputs) -> Dict:
    #     # Log train loss for epoch + image
    #     pass

    # def validation_epoch_end(self, outputs) -> Dict:
    #     # Log val loss for epoch + image
    #     pass
