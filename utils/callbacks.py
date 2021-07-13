import os
from pytorch_lightning.callbacks import Callback
import warnings


class SaveModelDescription(Callback):
    def on_train_start(self, trainer, pl_module):
        with open("model.txt", "w") as f:
            print(pl_module, file=f)
        if trainer.logger is not None:
            trainer.logger.experiment.save("model.txt")


class CircularModelCheckpoint(Callback):
    """
    Save a checkpoint every N steps with circular buffer
    """

    def __init__(self, dirpath=None, filename=None, period=1, save_last_k=3):
        """
        Args:
            save_every: how often to save
            no_checkpoints: number of checkpoints to keep

        """
        self.dirpath = dirpath
        self.period = period
        self.filename = filename
        self.save_last_k = save_last_k

    def on_batch_end(self, trainer, pl_module):
        # epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.period:
            return

        if self.dirpath is None:
            self.dirpath = os.path.join(trainer.logger.experiment.dir, "checkpoints")

        if self.filename is None:
            self.filename = str(pl_module.logger.experiment.name)

        filename = self.filename + "_step_" + str(global_step) + ".ckpt"

        ckpt_path = os.path.join(self.dirpath, filename)

        if len(pl_module.checkpoints) > self.save_last_k:
            try:
                os.remove(pl_module.checkpoints.popleft())
            except Exception as e:
                warnings.warn(str(e), RuntimeWarning)
                pass

        try:
            trainer.save_checkpoint(ckpt_path)
            pl_module.checkpoints.append(ckpt_path)
        except Exception as e:
            warnings.warn(str(e), RuntimeWarning)
            pass
