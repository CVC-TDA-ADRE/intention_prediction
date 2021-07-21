import pytorch_lightning as pl
from data.intention_dataset import IntentionDataset
from data.intention_dataset_classification import IntentionDatasetClass
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np


def collate(batch):
    clips = []
    original_clips = []
    boxes = []
    original_boxes = []
    paths = []
    labels = []

    for item in batch:
        clips.append(item[0])
        original_clips.append(item[1])
        labels.extend(item[2]) if isinstance(item[2], list) else labels.append(item[2])
        paths.append(item[3])
        if len(item) > 4:
            boxes.append(item[4].float())
            original_boxes.append(item[5])
        else:
            boxes = None
            original_boxes = None

    output = {
        "clip": torch.stack(clips),
        "original_clip": torch.stack(original_clips),
        "boxes": boxes,
        "original_boxes": original_boxes,
        "label": torch.as_tensor(labels).unsqueeze(1),
        "video_paths": paths,
    }

    return output


class IntentionDataloader(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path=None,
        dataset_type="detection",
        weighted_sampler=False,
        batch_size=4,
        num_workers=2,
        pin_memory=False,
        **kwargs
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_type = dataset_type
        self.weighted_sampler = weighted_sampler
        self.kwargs = kwargs

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.dataset_type == "detection":
                if "scale_crop" in self.kwargs:
                    del self.kwargs["scale_crop"]

                self.train_data = IntentionDataset(self.train_path, **self.kwargs)

                if self.val_path is not None:
                    self.val_data = IntentionDataset(self.val_path, **self.kwargs)
                else:
                    self.val_data = None
            elif self.dataset_type == "classification":
                self.train_data = IntentionDatasetClass(self.train_path, **self.kwargs)

                if self.val_path is not None:
                    self.val_data = IntentionDatasetClass(self.val_path, **self.kwargs)
                else:
                    self.val_data = None
            else:
                raise ValueError(
                    "Please enter a valid dataset type (classification, detection)"
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            pass

    def generate_weights(self):
        count_label = self.train_data.count_labels_max()
        weights_class = [
            count_label["total"] / count_label[0],
            count_label["total"] / count_label[1],
        ]
        weights = []
        for sample in self.train_data.dataset:
            labels = sample["label"]
            res, count = np.unique(labels, return_counts=True)
            if len(res) == 2:
                weights += [weights_class[count.argmax()]]
            else:
                weights += [weights_class[int(res)]]

        return weights

    def train_dataloader(self):

        if self.weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=self.generate_weights(), num_samples=self.train_data.__len__()
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            sampler=sampler,
            collate_fn=collate,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_data is None:
            return None
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    dataloader = IntentionDataloader(
        "/datatmp/Datasets/intention_prediction/PIE/processed_annotations/train.csv",
        resize=[256, 256],
        dataset_type="detection",
    )
    dataloader.setup("fit")
    batch = next(iter(dataloader.train_dataloader()))
    # print("boxes: ", len(batch["boxes"]), type(batch["boxes"]))
    print("clip: ", batch["clip"].shape, type(batch["clip"]))
    print("label: ", batch["label"].shape, type(batch["label"]))
    print(batch["label"])
    print("original_clip: ", batch["original_clip"][0].shape, type(batch["original_clip"]))
    print("video_paths: ", len(batch["video_paths"]), type(batch["video_paths"]))
