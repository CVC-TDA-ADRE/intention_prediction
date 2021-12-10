import os
import sys
import torch
import wandb
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.intention_dataloader import IntentionDataloader
from models.intention_predictor import IntentionPredictor
from utils.callbacks import SaveModelDescription, CircularModelCheckpoint
from utils.nested_dict import modify_nested_dict

assert len(sys.argv) > 1, "No config file specified"

config_path = sys.argv[1]

with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

wandb_conf = config["WANDB"]
sweep_config = {}
if wandb_conf["activate"]:
    if wandb_conf["save_dir"] is not None:
        os.makedirs(wandb_conf["save_dir"], exist_ok=True)
    wandb.init(name=wandb_conf["name"], project=wandb_conf["project"], dir=wandb_conf["save_dir"])
    sweep_config = wandb.config


def train(config):

    if wandb_conf["activate"]:
        logger = WandbLogger()

    for k, v in sweep_config.items():
        modify_nested_dict(config, k, v)

    data_config = config["DATA"]
    training_config = config["TRAINING"]
    model_config = config["MODEL"]

    if isinstance(data_config["resize"], int):
        data_config["resize"] = (data_config["resize"], data_config["resize"])

    n_gpus = training_config["gpus"]

    if data_config["num_workers"] < 0:
        if n_gpus < 0:
            data_config["num_workers"] = 6 * torch.cuda.device_count()
        else:
            data_config["num_workers"] = 6 * n_gpus

    print(config)

    data = IntentionDataloader(**data_config)

    predictor = IntentionPredictor(data_config, training_config, **model_config)

    callbacks = [SaveModelDescription()]
    if wandb_conf["activate"]:
        if training_config["save_checkpoints_every"] is not None:
            callbacks.append(
                CircularModelCheckpoint(
                    period=training_config["save_checkpoints_every"], save_last_k=training_config["save_last_k"] - 1
                )
            )
    else:
        logger = False

    if training_config["stochastic_weight_avg"]:
        callbacks.append(pl.callbacks.StochasticWeightAveraging())

    trainer = pl.Trainer(
        gpus=n_gpus,
        logger=logger,
        fast_dev_run=wandb_conf["debug"],
        max_epochs=training_config["epochs"],
        max_steps=training_config["max_steps"],
        log_every_n_steps=training_config["log_every"],
        limit_train_batches=training_config["frac_train"],
        limit_val_batches=training_config["frac_val"],
        checkpoint_callback=False,
        callbacks=callbacks,
        profiler=training_config["profiler"],
        # auto_lr_find=args.auto_lr_find,
    )
    if wandb_conf["activate"]:
        logger.watch(predictor.model)

    trainer.fit(predictor, data)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser.add_argument("--config_path", type=str)
    # parser.add_argument("--project", type=str, default="IntentionPred")
    # parser.add_argument("--name", type=str, default=None)
    # parser.add_argument("--save_dir", type=str, default=None)
    # parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    # parser.add_argument("--wandb", action="store_true", help="Start wandb")
    # parser.add_argument(
    #     "--accumulate_batch", type=int, default=1, help="Number of batch to accumulate"
    # )
    # parser.add_argument("--auto_lr_find", action="store_true", help="Store checkpoints")
    # Good practices:
    #  --accelerator 'ddp'
    #  --accelerator 'ddp'
    # args = parser.parse_args()
    train(config)
