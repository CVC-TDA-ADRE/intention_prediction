import torch
import wandb
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.intention_dataloader import IntentionDataloader
from models.intention_predictor import IntentionPredictor
from utils.callbacks import SaveModelDescription, CircularModelCheckpoint


def train(args):

    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data_config = config["DATA"]
    training_config = config["TRAINING"]
    model_config = config["MODEL"]

    if isinstance(data_config["resize"], int):
        data_config["resize"] = (data_config["resize"], data_config["resize"])

    data = IntentionDataloader(**data_config)
    data.setup("fit")

    n_gpus = training_config["gpus"] if not args.gpus else args.gpus
    if (n_gpus < 2 and n_gpus > 0) or (n_gpus < 0 and torch.cuda.device_count() < 2):
        print("Warning: Only 1 GPU using no accelerator")
        args.accelerator = None

    predictor = IntentionPredictor(data_config, training_config, **model_config)

    if args.wandb:
        wandb.login()
        logger = WandbLogger(
            name=args.name, project=args.project, config=config, save_dir=args.save_dir
        )
    else:
        logger = False

    callbacks = [SaveModelDescription()]
    if training_config["save_checkpoints_every"] is not None:
        callbacks.append(
            CircularModelCheckpoint(
                period=training_config["save_checkpoints_every"], save_last_k=3
            )
        )

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=n_gpus,
        logger=logger,
        fast_dev_run=args.debug,
        max_epochs=training_config["epochs"],
        log_every_n_steps=training_config["log_every"],
        limit_train_batches=training_config["frac_train"],
        limit_val_batches=training_config["frac_val"],
        accumulate_grad_batches=args.accumulate_batch,
        checkpoint_callback=False,
        callbacks=callbacks,
        # auto_lr_find=args.auto_lr_find,
    )
    if args.wandb:
        logger.watch(predictor.model)

    if args.auto_lr_find:
        trainer.tune(predictor, data)
    trainer.fit(predictor, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--project", type=str, default="IntentionPred")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--wandb", action="store_true", help="Start wandb")
    parser.add_argument(
        "--accumulate_batch", type=int, default=1, help="Number of batch to accumulate"
    )
    # parser.add_argument("--auto_lr_find", action="store_true", help="Store checkpoints")
    args = parser.parse_args()
    train(args)
