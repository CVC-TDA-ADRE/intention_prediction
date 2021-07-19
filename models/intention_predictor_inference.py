import torch.nn as nn
import pytorch_lightning as pl
from torch.functional import Tensor
import torchmetrics
from collections import deque
from models.x3d import X3D
from models.slow_r50 import SlowR50
from utils.visualization import VideoVisualizer


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

        class_names = {0: "not_crossing", 1: "crossing"}
        self.visualization = VideoVisualizer(
            num_classes=1, class_names=class_names, thres=0.5, mode="binary"
        )

    def forward(self, x, boxes=None) -> Tensor:
        return self.model(x, boxes)
