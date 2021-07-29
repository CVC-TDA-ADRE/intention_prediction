#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField

from perception_messages.msg import Detections2DArray

from message_filters import ApproximateTimeSynchronizer, Subscriber

import time
from cv_bridge import CvBridge

import os

# import yaml
import torch
import cv2
import numpy as np
from torchvision import transforms

import sys
import copy
from collections import deque, defaultdict

sys.path.insert(0, os.getcwd())
from models.intention_predictor_inference import IntentionPredictor
from utils.video_transforms import crop_video_bbox, UniformTemporalSubsample


def clip_boxes_to_image(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Clip an array of boxes to an image with the given height and width.
    Args:
        boxes (tensor): bounding boxes to perform clipping.
            Dimension is `num boxes` x 4.
        height (int): given image height.
        width (int): given image width.
    Returns:
        clipped_boxes (tensor): the clipped boxes with dimension of
            `num boxes` x 4.
    """
    clipped_boxes = copy.deepcopy(boxes)
    clipped_boxes[:, [0, 2]] = np.minimum(width - 1.0, np.maximum(0.0, boxes[:, [0, 2]]))
    clipped_boxes[:, [1, 3]] = np.minimum(height - 1.0, np.maximum(0.0, boxes[:, [1, 3]]))
    return clipped_boxes


class Detectron_Detector(Node):
    """
    Detectron detector class for ROS2
    Accepeted arguments:
        topic_in: String value. Message name to subscribe containing the RGB image.
        topic_in_detect: String value. Message name to publish containing the detections.
        config_file: String value. Configuration file with detectron2 format.
        weights_file: String value. Weights file to load the model.
        gpu: Integer value. GPU id to run the model.
    """

    def __init__(self):
        super().__init__("detector_detectron_node")

        # Declare parameters
        self.declare_parameters(
            namespace="",
            parameters=[
                ("topic_in", "image"),
                ("topic_in_detect", "detections"),
                ("topic_intent", "intent"),
                ("path_out", ""),
                # ("config_file", ""),
                ("weights_file", ""),
                ("gpu", 0),
            ],
        )

        # get parameters
        topic_in = self.get_parameter("topic_in").get_parameter_value().string_value
        topic_in_detect = (
            self.get_parameter("topic_in_detect").get_parameter_value().string_value
        )
        topic_intent = self.get_parameter("topic_intent").get_parameter_value().string_value
        # config_file = self.get_parameter("config_file").get_parameter_value().string_value
        weights_file = self.get_parameter("weights_file").get_parameter_value().string_value
        gpu = self.get_parameter("gpu").get_parameter_value().integer_value
        self.path_out = self.get_parameter("path_out").get_parameter_value().string_value
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # create subscriber
        # self.subscription = self.create_subscription(
        #     Image, topic_in, self.listener_callback, 5
        # )
        sub_img = Subscriber(self, Image, topic_in)
        sub_detections = Subscriber(self, Detections2DArray, topic_in_detect)

        ats = ApproximateTimeSynchronizer(
            [sub_img, sub_detections],
            queue_size=5,
            slop=0.2,
            allow_headerless=False,
        )
        ats.registerCallback(self.listener_callback_withdets)
        self.get_logger().info("Listening to (%s, %s) topics" % (topic_in, topic_in_detect))

        # # self.subscription  # prevent unused variable warning
        # self.get_logger().info("Listening to %s topic" % topic_in)

        # create publisher
        self.publisher_intent = self.create_publisher(PointCloud2, topic_intent, 5)
        self.get_logger().info("Publishing to topic %s" % (topic_intent))

        # initialize variables
        self.last_time = time.time()
        self.last_update = self.last_time
        self.cv_br = CvBridge()

        # Detector initialization
        self.model = self._init_model(weights_file)
        self.get_logger().info("intention model initialized")

        # Image stack
        self.store_detections = defaultdict(deque)
        self.last_detect_update = {}
        self.count = 0

    def _msg2detections(self, msg):
        detections = []
        idx_detect = []
        for obj_indx in range(len(msg.detections)):
            detection = msg.detections[obj_indx]
            # TODO: Adapt for each possible dataset
            if detection.label == 0:
                bbox = {
                    "x1": detection.center_x - detection.size_x / 2.0,
                    "y1": detection.center_y - detection.size_y / 2.0,
                    "x2": detection.center_x + detection.size_x / 2.0,
                    "y2": detection.center_y + detection.size_y / 2.0,
                    # "instance": detection.instance,
                    # "label": detection.label,
                }
                idx_detect.append(detection.instance)
                detections.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])

        return np.asarray(detections), np.asarray(idx_detect)

    def prepare_detections(self, img, detections, idx_detection):

        warmup = True
        for box, idx in zip(detections, idx_detection):
            if idx == -1:
                continue
            ped_clip = crop_video_bbox(
                torch.from_numpy(img).unsqueeze(0),
                [box],
                self.original_height,
                self.original_width,
                scale=self.scale_crop,
            )
            ped_clip = self.transform(ped_clip.squeeze(0).numpy())
            if len(self.store_detections[idx]) < self.input_size_needed:
                self.store_detections[idx].append(ped_clip.squeeze(0))
            else:
                self.store_detections[idx].popleft()
                self.store_detections[idx].append(ped_clip.squeeze(0))
                warmup = False
            self.last_detect_update[idx] = -1

        # Cleanup
        for idx in list(self.store_detections.keys()):
            self.last_detect_update[idx] += 1
            if self.last_detect_update[idx] > 5:
                del self.last_detect_update[idx]
                del self.store_detections[idx]

        return warmup

    def listener_callback_withdets(self, msg_img, msg_dets):
        # transform message
        original_img = np.asarray(self.cv_br.imgmsg_to_cv2(msg_img), dtype=np.uint8)
        self.original_height, self.original_width = (
            original_img.shape[0],
            original_img.shape[1],
        )
        detections, idx_detect = self._msg2detections(msg_dets)
        warmup = self.prepare_detections(original_img, detections, idx_detect)

        if warmup:
            return
        else:

            if detections.size == 0:
                return

            # Compute intention
            intentions = self._run_model(idx_detect)

            if not all(intentions.shape):
                return

            # Publish
            msg_intentions = self.array_to_msg(intentions)
            msg_intentions.header = msg_img.header
            self.publisher_intent.publish(msg_intentions)

            # Save Image
            if self.path_out != "":
                os.makedirs(self.path_out, exist_ok=True)
                filename_out = os.path.join(self.path_out, f"image_{self.count}.jpg")
                self.count += 1
                image_with_pred = self.draw_image(
                    original_img, detections, torch.sigmoid(intentions)
                )
                cv2.imwrite(filename_out, image_with_pred)

            # compute true fps
            curr_time = time.time()
            fps = 1 / (curr_time - self.last_time)
            self.last_time = curr_time
            if (curr_time - self.last_update) > 5.0:
                self.last_update = curr_time
                self.get_logger().info("Computing intention at %.01f fps" % fps)

    def _init_model(self, weights_file):

        model = IntentionPredictor.load_from_checkpoint(weights_file)

        data_config = model.data_kwargs

        if isinstance(data_config["resize"], int):
            data_config["resize"] = (data_config["resize"], data_config["resize"])

        self.resize = data_config["resize"]
        input_seq_size = data_config["input_seq_size"]
        self.image_mean = data_config["image_mean"]
        self.image_std = data_config["image_std"]
        self.sample_rate = data_config["sample_rate"]
        self.input_size_needed = self.sample_rate * input_seq_size

        increase_perc = 0.2
        self.scale_crop = data_config["scale_crop"]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (
                        int(self.resize[0] * (1 + increase_perc)),
                        int(self.resize[1] * (1 + increase_perc)),
                    )
                ),
                transforms.RandomCrop(self.resize),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        self.sampling = UniformTemporalSubsample(input_seq_size)

        model.eval()
        model.cuda()

        return model

    def draw_image(self, frame, boxes, preds):
        new_image = self.model.visualization.draw_one_frame(frame, preds, boxes)
        return new_image

    @torch.no_grad()
    def _run_model(self, idx_detect):

        predictions = []
        for idx in idx_detect:
            if (
                idx in self.store_detections
                and len(self.store_detections[idx]) == self.input_size_needed
            ):
                clip = torch.stack(list(self.store_detections[idx]), axis=1)
                clip = self.sampling(clip).unsqueeze(0).float().cuda()
                # Get predictions
                pred = self.model(clip, None).squeeze(0)
            else:
                pred = torch.as_tensor([-1], device=torch.device("cuda"))
            # self.get_logger().info(f"Pred shape {pred.shape}")
            predictions += [pred]

        torch.cuda.synchronize()

        return torch.stack(predictions, axis=0)

    def array_to_msg(self, intents):
        intents = intents.detach().cpu().numpy()
        dtype = np.float32
        ros_dtype = PointField.FLOAT32
        itemsize = np.dtype(dtype).itemsize

        N, n_points = intents.shape

        fields = [PointField(name="intent", offset=0, datatype=ros_dtype, count=1)]

        data = intents.astype(dtype).tobytes()

        msg = PointCloud2(
            height=1,
            width=N * n_points,
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=itemsize,
            row_step=itemsize * N * n_points,
            data=data,
        )

        return msg


def main(args=None):
    rclpy.init(args=args)

    detector_publisher = Detectron_Detector()

    rclpy.spin(detector_publisher)

    detector_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
