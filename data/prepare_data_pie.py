import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pie_path", default="./", type=str, help="Complete path to dataset")
parser.add_argument(
    "--train_sets",
    default=[1, 2, 3, 4, 5, 6],
    type=int,
    nargs="+",
    help="Number of the set as list",
)
parser.add_argument("--val_sets", default=[], type=int, nargs="+", help="Number of the set as list")
parser.add_argument("--test_sets", default=[], type=int, nargs="+", help="Number of the set as list")
parser.add_argument(
    "--out_folder",
    default="processed_annotations",
    type=str,
    help="Name of the output folder",
)

args = parser.parse_args()
pie_folder = args.pie_path
out_folder = args.out_folder

import sys

sys.path.insert(0, pie_folder)
import pie_data

if not os.path.isdir(os.path.join(pie_folder, out_folder)):
    os.mkdir(os.path.join(pie_folder, out_folder))

# if not os.path.isdir(os.path.join(pie_folder, out_folder, "train")):
#     os.mkdir(os.path.join(pie_folder, out_folder, "train"))

# if not os.path.isdir(os.path.join(pie_folder, out_folder, "val")):
#     os.mkdir(os.path.join(pie_folder, out_folder, "val"))

# if not os.path.isdir(os.path.join(pie_folder, out_folder, "test")):
#     os.mkdir(os.path.join(pie_folder, out_folder, "test"))

pie = pie_data.PIE(data_path=pie_folder)
dataset = pie.generate_database()

sets = sorted(list(dataset.keys()))

train_sets = [sets[i - 1] for i in args.train_sets]
val_sets = [sets[i - 1] for i in args.val_sets]
test_sets = [sets[i - 1] for i in args.test_sets]


df_train = pd.DataFrame()
df_val = pd.DataFrame()
df_test = pd.DataFrame()

for set_ in tqdm(sets):
    for video in dataset[set_].keys():
        vid = dataset[set_][video]
        data = np.empty((0, 9))
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
                    os.path.join(pie_folder, "videos", set_, video + ".mp4"),
                    frames.shape[0],
                ).reshape(-1, 1)

                cross = np.array(vid["ped_annotations"][ped]["behavior"]["cross"]).reshape(-1, 1)
                cross[cross < 0] = 0
                vehicle_speed = np.array([vid["vehicle_annotations"][i[0]]["GPS_speed"] for i in frames]).reshape(
                    -1, 1
                )
                ped_data = np.hstack((frames, ids, x1, y1, x2, y2, video_path, cross, vehicle_speed))
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
                "vehicle_speed": data[:, 8].reshape(-1),
            }
        )
        data_to_write["filename"] = video
        # data_to_write.filename = data_to_write.filename.apply(lambda x: x.split("/")[-1].split(''))

        if set_ in train_sets:
            df_train = df_train.append(data_to_write, ignore_index=True)
            # data_to_write.to_csv(
            #     os.path.join(pie_folder, "processed_annotations", "train", video + ".csv"),
            #     index=False,
            # )
        elif set_ in val_sets:
            df_val = df_val.append(data_to_write, ignore_index=True)
            # data_to_write.to_csv(
            #     os.path.join(pie_folder, "processed_annotations", "val", video + ".csv"),
            #     index=False,
            # )
        elif set_ in test_sets:
            df_test = df_test.append(data_to_write, ignore_index=True)
            # data_to_write.to_csv(
            #     os.path.join(pie_folder, "processed_annotations", "test", video + ".csv"),
            #     index=False,
            # )
df_train.to_csv(
    os.path.join(pie_folder, out_folder, "train.csv"),
    index=False,
)
df_val.to_csv(
    os.path.join(pie_folder, out_folder, "val.csv"),
    index=False,
)
df_test.to_csv(
    os.path.join(pie_folder, out_folder, "test.csv"),
    index=False,
)
