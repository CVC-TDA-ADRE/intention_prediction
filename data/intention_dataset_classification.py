from torch.utils.data import Dataset
from torchvision import transforms
from pytorchvideo.transforms import (
    Normalize,
    # ShortSideScale,
    UniformTemporalSubsample,
)
import os
import pickle
import decord
import numpy as np
import pandas as pd
from collections import defaultdict

from utils.video_transforms import ToTensorVideo, crop_video_bbox
from torchvision.transforms import Resize
from utils.utils import assert_arr_continuous


class IntentionDatasetClass(Dataset):
    """Can sample data from  pedestrian intention databases"""

    def __init__(
        self,
        annotation_path,
        scale_crop=2,
        frame_future=0,
        desired_fps=20,
        input_seq_size=10,
        resize=None,
        overlap_percent=0.8,
        data_fps=30,
        image_mean=[0.45, 0.45, 0.45],
        image_std=[0.225, 0.225, 0.225],
    ):
        # self.data_path = data_path
        self.input_seq_size = input_seq_size
        self.overlap_percent = overlap_percent
        self.data_fps = data_fps
        self.desired_fps = desired_fps
        self.resize = resize
        self.scale_crop = scale_crop
        self.frame_future = frame_future

        # df = pd.read_csv(
        #     os.path.join(self.data_path, f"processed_annotations/{split_type}.csv")
        # )
        self.cache_dir = os.path.join(os.path.dirname(annotation_path), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        file_name = os.path.basename(annotation_path).split(".")[0]
        self.dataset_name = f"class_{file_name}_{input_seq_size}_{int(overlap_percent*100)}_{data_fps}_{desired_fps}.pkl"

        df = pd.read_csv(annotation_path)
        self.dataset = self.process_df(df)

        transform_chain = [ToTensorVideo(), UniformTemporalSubsample(input_seq_size)]
        if resize:
            transform_chain += [Resize(resize)]
        transform_chain += [Normalize(mean=image_mean, std=image_std)]

        self.video_transform = transforms.Compose(transform_chain)

        decord.bridge.set_bridge("torch")

    def count_labels_tot(self):
        count = defaultdict(int)
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            labels = sample["label"]
            for label in labels:
                count[int(label)] += 1
                count["total"] += 1

        return count

    def count_labels_max(self):
        count = defaultdict(int)
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            count[int(np.mean(sample["label"]) >= 0.5)] += 1
            count["total"] += 1

        return count

    def process_df(self, df):

        if not os.path.exists(os.path.join(self.cache_dir, self.dataset_name)):

            df["bounding_box"] = df[["x1", "y1", "x2", "y2"]].apply(
                lambda row: [row.x1, row.y1, row.x2, row.y2], axis=1
            )

            input_seq = int((self.data_fps / self.desired_fps) * self.input_seq_size)
            stride = int(input_seq * (1 - self.overlap_percent)) + 1
            output_seq = int((self.data_fps / self.desired_fps) * self.frame_future)

            dataset = []
            for name, group in df.groupby(["ID", "video_path"]):
                k = 0
                while k + input_seq + output_seq <= len(group):
                    end_range = k + input_seq
                    frame_range = group.frame.values[k:end_range]
                    if assert_arr_continuous(frame_range):
                        vid_path = name[1]
                        start = frame_range[0]
                        end = frame_range[-1]
                        label = group.crossing_true.values[end_range - 1 + output_seq]
                        bbox = list(group.bounding_box.values[k:end_range])
                        ids = group.ID.values[end_range - 1]
                        dataset.append(
                            {
                                "video_path": vid_path,
                                "start": start,
                                "end": end,
                                "label": label,
                                "bbox": bbox,
                                "ids": ids,
                            }
                        )
                    k += stride

            with open(os.path.join(self.cache_dir, self.dataset_name), "wb") as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            print("Loading dataset from cache...")
            with open(os.path.join(self.cache_dir, self.dataset_name), "rb") as handle:
                dataset = pickle.load(handle)

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, sample_idx):

        sample = self.dataset[sample_idx]
        original_boxes = np.array(sample["bbox"])
        label = sample["label"]
        vr = decord.VideoReader(sample["video_path"])
        original_clip = vr.get_batch(range(sample["start"], sample["end"] + 1))
        height, width = original_clip.shape[1], original_clip.shape[2]
        original_clip, _ = crop_video_bbox(
            original_clip, original_boxes, height, width, scale=self.scale_crop
        )

        clip = self.video_transform(original_clip)

        output = [clip, original_clip, label, sample["video_path"]]

        return output


if __name__ == "__main__":
    dataset = IntentionDatasetClass(
        "/datatmp/Datasets/intention_prediction/JAAD/processed_annotations/train.csv",
        20,
        10,
    )
    print(dataset.__len__())
    print(dataset.__getitem__(0))
