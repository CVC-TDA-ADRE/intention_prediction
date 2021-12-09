import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--jaad_path", default="./", type=str, help="Complete path to dataset")
parser.add_argument("--train_ratio", default=1, type=float, help="Ratio of train video")
parser.add_argument("--val_ratio", default=0, type=float, help="Ratio of val video")
parser.add_argument("--test_ratio", default=0, type=float, help="Ratio of test video")
parser.add_argument(
    "--out_folder",
    default="processed_annotations",
    type=str,
    help="Name of the output folder",
)

args = parser.parse_args()
jaad_folder = args.jaad_path
out_folder = args.out_folder

import sys

sys.path.insert(0, jaad_folder)
import jaad_data

if not os.path.isdir(os.path.join(jaad_folder, out_folder)):
    os.mkdir(os.path.join(jaad_folder, out_folder))

# if not os.path.isdir(os.path.join(jaad_folder, out_folder, "train")):
#     os.mkdir(os.path.join(jaad_folder, out_folder, "train"))

# if not os.path.isdir(os.path.join(jaad_folder, out_folder, "val")):
#     os.mkdir(os.path.join(jaad_folder, out_folder, "val"))

# if not os.path.isdir(os.path.join(jaad_folder, out_folder, "test")):
#     os.mkdir(os.path.join(jaad_folder, out_folder, "test"))

jaad = jaad_data.JAAD(jaad_folder)
dataset = jaad.generate_database()

n_train_video = round(args.train_ratio * len(dataset))
n_val_video = round(args.val_ratio * len(dataset))
n_test_video = round(args.test_ratio * len(dataset))

videos = list(dataset.keys())
train_videos = videos[:n_train_video]
if n_val_video:
    val_videos = videos[n_train_video : n_train_video + n_val_video]
if n_test_video:
    test_videos = videos[n_train_video + n_val_video :]

df_train = pd.DataFrame()
df_val = pd.DataFrame()
df_test = pd.DataFrame()

for video in tqdm(dataset):
    vid = dataset[video]
    data = np.empty((0, 8))
    id_ped = 0
    for ped in vid["ped_annotations"]:
        if vid["ped_annotations"][ped]["behavior"]:
            frames = np.array(vid["ped_annotations"][ped]["frames"]).reshape(-1, 1)
            ids = np.repeat(id_ped, frames.shape[0]).reshape(-1, 1)
            id_ped += 1
            bbox = np.array(vid["ped_annotations"][ped]["bbox"])
            x1 = bbox[:, 0].reshape(-1, 1)
            y1 = bbox[:, 1].reshape(-1, 1)
            x2 = bbox[:, 2].reshape(-1, 1)
            y2 = bbox[:, 3].reshape(-1, 1)
            video_path = np.repeat(
                os.path.join(jaad_folder, "videos", video + ".mp4"),
                frames.shape[0],
            ).reshape(-1, 1)

            cross = np.array(vid["ped_annotations"][ped]["behavior"]["cross"]).reshape(-1, 1)

            ped_data = np.hstack((frames, ids, x1, y1, x2, y2, video_path, cross))
            data = np.vstack((data, ped_data))
    data_to_write = pd.DataFrame(
        {
            "frame": data[:, 0].reshape(-1),
            "ID": data[:, 1].reshape(-1),
            "x1": data[:, 2].reshape(-1),
            "y1": data[:, 3].reshape(-1),
            "x2": data[:, 4].reshape(-1),
            "y2": data[:, 5].reshape(-1),
            "video_path": data[:, 6].reshape(-1),
            "crossing_true": data[:, 7].reshape(-1),
        }
    )
    data_to_write["filename"] = video
    # data_to_write.filename = data_to_write.filename.apply(lambda x: x.split("/")[-1].split(''))

    if video in train_videos:
        df_train = df_train.append(data_to_write, ignore_index=True)
        # data_to_write.to_csv(
        #     os.path.join(jaad_folder, "processed_annotations", "train", video + ".csv"),
        #     index=False,
        # )
    elif video in val_videos:
        df_val = df_val.append(data_to_write, ignore_index=True)
        # data_to_write.to_csv(
        #     os.path.join(jaad_folder, "processed_annotations", "val", video + ".csv"),
        #     index=False,
        # )
    elif video in test_videos:
        df_test = df_test.append(data_to_write, ignore_index=True)
        # data_to_write.to_csv(
        #     os.path.join(jaad_folder, "processed_annotations", "test", video + ".csv"),
        #     index=False,
        # )
df_train.to_csv(
    os.path.join(jaad_folder, out_folder, "train.csv"),
    index=False,
)
df_val.to_csv(
    os.path.join(jaad_folder, out_folder, "val.csv"),
    index=False,
)
df_test.to_csv(
    os.path.join(jaad_folder, out_folder, "test.csv"),
    index=False,
)
