import copy
import glob
import json
import os
from typing import Any, Iterator, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rosbag
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from transforms3d.quaternions import *


DEPTH_OPTICAL_FRAME_POSE = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]
INTRINSICS = np.array(
    [
        [892, 0.0, 660],
        [0.0, 892, 355],
        [0.0, 0.0, 1.0],
    ]
)
WIDTH = 1280
HEIGHT = 720


def read_ros_bag(ros_bag_path):
    try:
        bag = rosbag.Bag(ros_bag_path)
    except rosbag.bag.ROSBagUnindexedException:
        os.system(f"rosbag reindex {ros_bag_path}")
        bag = rosbag.Bag(ros_bag_path)

    pc_msgs = []
    ee_pos = None
    ee_quat = None
    finger_width = None
    for topic, msg, t in bag.read_messages():
        if "/depth/color/points" in topic:
            pc_msgs.append(msg)

        if topic == "/ee_pose":
            ee_pos = np.array([msg.position.x, msg.position.y, msg.position.z])
            ee_quat = np.array(
                [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ]
            )

        if topic == "/gripper_width":
            finger_width = msg.data

    bag.close()

    return pc_msgs, ee_pos, ee_quat, finger_width


def parse_pointcloud2(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    data = data.reshape(-1, msg.point_step)

    cloud_xyz = copy.deepcopy(data[:, :12]).view(dtype=np.float32).reshape(-1, 3)
    cloud_bgr = copy.deepcopy(data[:, 16:20]) / 255
    cloud_rgb = cloud_bgr[:, -2::-1]

    return cloud_xyz, cloud_rgb


def merge(pc_msgs, extrinsics, visualize=False):
    points_list = []
    rgb_list = []
    depth_list = []
    for i in range(len(pc_msgs)):
        cloud_xyz, cloud_rgb = parse_pointcloud2(pc_msgs[i])

        pixel_pos = cloud_xyz @ INTRINSICS.T
        pixel_pos[:, :2] /= pixel_pos[:, 2:]
        pixel_pos = np.round(pixel_pos[:, :2]).astype(int)

        # Create depth map
        valid_indices = (
            (pixel_pos[:, 0] >= 0)
            & (pixel_pos[:, 0] < WIDTH)
            & (pixel_pos[:, 1] >= 0)
            & (pixel_pos[:, 1] < HEIGHT)
        )

        # Create RGB image
        rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        rgb[pixel_pos[valid_indices, 1], pixel_pos[valid_indices, 0]] = (
            cloud_rgb[valid_indices] * 255
        ).astype(np.uint8)
        # print(rgb.max())

        depth = np.zeros((HEIGHT, WIDTH))
        depth[pixel_pos[valid_indices, 1], pixel_pos[valid_indices, 0]] = cloud_xyz[
            :, 2
        ][valid_indices]

        depth = np.clip(depth, 0, 2.0) * 1000
        # print(depth.max())

        points = (
            quat2mat(DEPTH_OPTICAL_FRAME_POSE[3:]) @ cloud_xyz[:, :3].T
        ).T + DEPTH_OPTICAL_FRAME_POSE[:3]
        points_transform = points @ extrinsics[i][:3, :3].T + extrinsics[i][:3, 3]

        depth_filter = cloud_xyz[:, 2] < 1.0
        points_transform = points_transform[depth_filter]
        cloud_rgb = cloud_rgb[depth_filter]

        rgb_list.append(rgb)
        depth_list.append(depth)
        points_list.append(np.concatenate([points_transform, cloud_rgb], axis=-1))

    merge_points = np.concatenate(points_list, axis=0)

    if visualize:
        figure, axes = plt.subplots(len(pc_msgs), 2, figsize=(16, 16))
        for i in range(len(pc_msgs)):
            axes[i][0].imshow(rgb_list[i])
            axes[i][1].imshow(depth_list[i])

        plt.show()
        plt.close()

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(merge_points[:, :3])
        pc.colors = o3d.utility.Vector3dVector(merge_points[:, 3:])
        o3d.visualization.draw_geometries([pc])

    return merge_points


class Robocook(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(64, 64, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(64, 64, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(10,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [7x robot joint angles, "
                                        "2x gripper position, 1x door opening angle].",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(10,),
                                dtype=np.float32,
                                doc="Robot action, consists of [7x joint velocities, "
                                "2x gripper velocities, 1x terminate episode].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(path="data/train/episode_*.npy"),
            # "val": self._generate_examples(path="data/val/episode_*.npy"),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        with open(os.path.join("cam_pose.json"), "r") as f:
            cam_pose_dict = json.load(f)

        extrinsics = [np.eye(4) for _ in range(4)]
        for i in range(4):
            extrinsics[i][:3, :3] = quat2mat(cam_pose_dict[f"cam_{i+1}"]["orientation"])
            extrinsics[i][:3, 3] = np.array(cam_pose_dict[f"cam_{i+1}"]["position"])

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # data = np.load(
            #     episode_path, allow_pickle=True
            # )  # this is a list of dicts in our case

            # for i, step in enumerate(data):

            rosbag_path_list = glob.glob(os.path.join(episode_path, "*.bag"))

            episode = []
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            for i, rosbag_path in enumerate(rosbag_path_list):
                rosbag_path = "/media/haochen/Game Drive PS41/robocook/raw_data/roller_large_robot_v4/ep_000/seq_000/0.000.bag"
                pc_msgs, ee_pos, ee_quat, finger_width = read_ros_bag(rosbag_path)

                step = merge(pc_msgs, extrinsics, visualize=False)
                # compute Kona language embedding
                language_embedding = self._embed([step["language_instruction"]])[
                    0
                ].numpy()

                episode.append(
                    {
                        "observation": {
                            "image": step["image"],
                            "wrist_image": step["wrist_image"],
                            "state": step["state"],
                        },
                        "action": step["action"],
                        "discount": 1.0,
                        "reward": float(i == (len(data) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(data) - 1),
                        "is_terminal": i == (len(data) - 1),
                        "language_instruction": step["language_instruction"],
                        "language_embedding": language_embedding,
                    }
                )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        ps4_drive_prefix = "/media/haochen/Game Drive PS4/robocook/raw_data"
        ps4_drive_task_list = [
            "gripper_sym_rod_robot_v4",
            "gripper_sym_plane_robot_v4",
            "gripper_asym_robot_v4",
            "roller_large_robot_v4",
            "roller_small_robot_v4",
        ]

        wd_drive_prefix = "/media/haochen/wd_drive/robocook/raw_data"
        wd_drive_task_list = [
            "press_circle_robot_v4",
            "press_square_robot_v4",
            "punch_circle_robot_v4",
            "punch_square_robot_v4",
        ]

        episode_paths = []
        for ps4_task in ps4_drive_task_list:
            path = os.path.join(ps4_drive_prefix, ps4_task, "ep_*/seq_*")
            episode_paths += glob.glob(path)

        for wd_task in wd_drive_task_list:
            path = os.path.join(wd_drive_prefix, wd_task, "ep_*/seq_*")
            episode_paths += glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
