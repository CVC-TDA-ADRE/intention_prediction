import time
import yaml
import datetime
import torch
import numpy as np
import wandb
import argparse
import torch.nn as nn
import torch.optim as optim
from utils.utils import video_to_stream
from utils.visualization import VideoVisualizer
from models.x3d import X3D_RoI
from models.slow_r50 import SlowR50
from data.intention_dataloader import IntentionDataloader
import torchmetrics


def epoch_step(
    model,
    optimizer,
    loader,
    criterion,
    acc_func,
    epoch=0,
    epoch_type="training",
    device="cpu",
    verbose=False,
    log_every=1,
):
    log = {}
    log["epoch"] = epoch
    if epoch_type == "training":
        model.train()
    else:
        model.eval()

    for i, batch in enumerate(loader):

        clip, boxes, labels = (
            batch["clip"].to(device),
            batch["boxes"],
            batch["label"].to(device),
        )
        boxes = [_data.to(device) for _data in boxes]

        # Generate the videos
        preds = model(clip, boxes)

        # Discriminator Training
        if epoch_type == "training":
            optimizer.zero_grad()

            loss = criterion(preds, labels.float())
            acc = acc_func(torch.sigmoid(preds), labels)

            loss.backward()
            optimizer.step()

        elif epoch_type == "validation":
            loss = criterion(preds, labels.float())
            acc = acc_func(torch.sigmoid(preds), labels)

        log[f"{epoch_type}/loss"] = loss
        log[f"{epoch_type}/acc"] = acc

        if verbose and i % log_every == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            terminal_log = f"({epoch_type}) Elapsed [{et}], iter {i}"
            for k in log.keys():
                terminal_log += f", {k.split('/')[-1]}: {log[k]:.4f}"
            print(terminal_log, end="\r")

        if args.wandb and i % args.log_every == 0:
            wandb.log(log)
            if i % training_config["video_every"] == 0:
                sample_clip = batch["original_clip"][0].cpu() / 255.0
                sample_preds = torch.sigmoid(preds[: len(boxes[0])]).detach().cpu()
                sample_boxes = batch["original_boxes"][0]

                preview_video = visualization.draw_clip_range(
                    sample_clip, sample_preds, sample_boxes
                )
                wandb.log(
                    {
                        "train/video": wandb.Video(
                            video_to_stream(
                                np.array(preview_video), fps=data_config["data_fps"]
                            ),
                            format="mp4",
                            caption=f"True label: {labels[: len(boxes[0])].detach().cpu().numpy()}",
                        ),
                        # "video": wandb.Video(video_to_stream(clip, fps=self.fps), format="mp4")
                    }
                )


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--name", type=str, default="IntentionPred")
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--debug", action="store_true", help="Run in debug mode")
parser.add_argument("--verbose", action="store_true", help="Run in debug mode")
parser.add_argument("--wandb", action="store_true", help="Start wandb")
parser.add_argument(
    "--accumulate_batch", type=int, default=1, help="Number of batch to accumulate"
)
parser.add_argument("--save_checkpoints", action="store_true", help="Store checkpoints")
# parser.add_argument("--auto_lr_find", action="store_true", help="Store checkpoints")
args = parser.parse_args()

device = torch.device("cpu")

with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

data_config = config["DATA"]
training_config = config["TRAINING"]
model_config = config["MODEL"]

if isinstance(data_config["resize"], int):
    data_config["resize"] = (data_config["resize"], data_config["resize"])

if training_config["gpus"] >= 0:
    device = torch.device("cuda:" + str(training_config["gpus"]))

class_names = {0: "not_crossing", 1: "crossing"}
visualization = VideoVisualizer(
    num_classes=1, class_names=class_names, thres=0.5, mode="binary"
)

if args.wandb:
    wandb.init(project=args.name, config=config)

if training_config["model_to_use"] == "x3d":
    model = X3D_RoI(
        crop_size=data_config["resize"],
        clip_length=data_config["input_seq_size"],
        **model_config,
    ).to(device)
elif training_config["model_to_use"] == "slow_r50":
    model = SlowR50(**model_config).to(device)
print(model)

# Losses
criterion = nn.BCEWithLogitsLoss().to(device)

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=training_config["lr"],
    betas=training_config["betas"],
)

# Datasets
data = IntentionDataloader(**data_config)
data.setup("fit")

train_dataloader = data.train_dataloader()
val_dataloader = data.train_dataloader()

# Metrics
train_accuracy = torchmetrics.Accuracy().to(device)
val_accuracy = torchmetrics.Accuracy().to(device)


start_time = time.time()
start_epoch = 0
for epoch in range(start_epoch, training_config["epochs"]):
    epoch_step(
        model,
        optimizer,
        train_dataloader,
        criterion,
        train_accuracy,
        epoch=epoch,
        epoch_type="training",
        device=device,
        verbose=args.verbose,
        log_every=training_config["log_every"],
    )
    if val_dataloader is not None:
        epoch_step(
            model,
            None,
            val_dataloader,
            criterion,
            val_accuracy,
            epoch=epoch,
            epoch_type="validation",
            device=device,
            verbose=args.verbose,
            log_every=training_config["log_every"],
        )
