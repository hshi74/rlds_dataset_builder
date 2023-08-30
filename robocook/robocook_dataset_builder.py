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
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import quat2euler
from transforms3d.quaternions import *

DEPTH_OPTICAL_FRAME_POSE = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]

with open(
    os.path.join("/home/haochen/projects/rlds_dataset_builder/cam_pose.json"), "r"
) as f:
    cam_pose_dict = json.load(f)

EXTRINSICS = [np.eye(4) for _ in range(4)]
for i in range(4):
    EXTRINSICS[i][:3, :3] = quat2mat(
        cam_pose_dict[f"cam_{i+1}"]["orientation"]
    ) @ quat2mat(DEPTH_OPTICAL_FRAME_POSE[3:])
    EXTRINSICS[i][:3, 3] = np.array(cam_pose_dict[f"cam_{i+1}"]["position"])
    EXTRINSICS[i] = EXTRINSICS[i].astype(np.float32)

INTRINSICS = np.array(
    [
        [891.4971313476562, 0.0, 651.0776977539062],
        [0.0, 891.4971313476562, 357.4739990234375],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
WIDTH = 1280
HEIGHT = 720


def add_angles(delta, source, degrees=False):
    delta_rot = R.from_euler("xyz", delta, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)


def angle_diff(target, source, degrees=False):
    target_rot = R.from_euler("xyz", target, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")


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


def process(rosbag_path, visualize=False):
    pc_msgs, ee_pos, ee_quat, finger_width = read_ros_bag(rosbag_path)
    state = np.concatenate(
        [ee_pos, add_angles([np.pi, 0, 0], quat2euler(ee_quat)), [finger_width]]
    ).astype(np.float32)

    observation = {"state": state}
    merge_pc = o3d.geometry.PointCloud()
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

        depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        depth[pixel_pos[valid_indices, 1], pixel_pos[valid_indices, 0]] = cloud_xyz[
            :, 2
        ][valid_indices]

        depth = np.clip(depth, 0.0, 2.0) * 1000.0
        # print(depth.max())

        # points = cloud_xyz @ EXTRINSICS[i][:3, :3].T + EXTRINSICS[i][:3, 3]
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(points[cloud_xyz[:, 2] < 1.0])
        # pc.colors = o3d.utility.Vector3dVector(cloud_rgb[cloud_xyz[:, 2] < 1.0])
        # merge_pc += pc

        observation[f"image_{i+1}"] = rgb
        observation[f"depth_{i+1}"] = depth

    if visualize:
        o3d.visualization.draw_geometries([merge_pc])

        figure, axes = plt.subplots(len(pc_msgs), 2, figsize=(16, 16))
        for i in range(len(pc_msgs)):
            axes[i][0].imshow(observation[f"image_{i+1}"])
            axes[i][1].imshow(observation[f"depth_{i+1}"])

        plt.show()
        plt.close()

    return observation


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
                                    "image_1": tfds.features.Image(
                                        shape=(720, 1280, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Camera 1 RGB observation.",
                                    ),
                                    "image_2": tfds.features.Image(
                                        shape=(720, 1280, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Camera 2 RGB observation.",
                                    ),
                                    "image_3": tfds.features.Image(
                                        shape=(720, 1280, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Camera 3 RGB observation.",
                                    ),
                                    "image_4": tfds.features.Image(
                                        shape=(720, 1280, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Camera 4 RGB observation.",
                                    ),
                                    "depth_1": tfds.features.Tensor(
                                        shape=(720, 1280),
                                        dtype=np.float32,
                                        doc="Camera 1 Depth observation.",
                                    ),
                                    "depth_2": tfds.features.Tensor(
                                        shape=(720, 1280),
                                        dtype=np.float32,
                                        doc="Camera 2 Depth observation.",
                                    ),
                                    "depth_3": tfds.features.Tensor(
                                        shape=(720, 1280),
                                        dtype=np.float32,
                                        doc="Camera 3 Depth observation.",
                                    ),
                                    "depth_4": tfds.features.Tensor(
                                        shape=(720, 1280),
                                        dtype=np.float32,
                                        doc="Camera 4 Depth observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [3x robot end-effector position, "
                                        "3x robot end-effector euler angles, 1x gripper position].",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [3x robot end-effector velocities, "
                                "3x robot end-effector angular velocities, 1x gripper velocity].",
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
                            "intrinsics_1": tfds.features.Tensor(
                                shape=(3, 3),
                                dtype=np.float32,
                                doc="Camera 1 Intrinsic Matrix.",
                            ),
                            "intrinsics_2": tfds.features.Tensor(
                                shape=(3, 3),
                                dtype=np.float32,
                                doc="Camera 2 Intrinsic Matrix.",
                            ),
                            "intrinsics_3": tfds.features.Tensor(
                                shape=(3, 3),
                                dtype=np.float32,
                                doc="Camera 3 Intrinsic Matrix.",
                            ),
                            "intrinsics_4": tfds.features.Tensor(
                                shape=(3, 3),
                                dtype=np.float32,
                                doc="Camera 4 Intrinsic Matrix.",
                            ),
                            "extrinsics_1": tfds.features.Tensor(
                                shape=(4, 4),
                                dtype=np.float32,
                                doc="Camera 1 Extrinsic Matrix.",
                            ),
                            "extrinsics_2": tfds.features.Tensor(
                                shape=(4, 4),
                                dtype=np.float32,
                                doc="Camera 2 Extrinsic Matrix.",
                            ),
                            "extrinsics_3": tfds.features.Tensor(
                                shape=(4, 4),
                                dtype=np.float32,
                                doc="Camera 3 Extrinsic Matrix.",
                            ),
                            "extrinsics_4": tfds.features.Tensor(
                                shape=(4, 4),
                                dtype=np.float32,
                                doc="Camera 4 Extrinsic Matrix.",
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(),
            # "val": self._generate_examples(path="data/val/episode_*.npy"),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, language_instruction):
            # load raw data --> this should change for your dataset
            # data = np.load(
            #     episode_path, allow_pickle=True
            # )  # this is a list of dicts in our case

            # for i, step in enumerate(data):

            rosbag_path_list = sorted(glob.glob(os.path.join(episode_path, "*.bag")))

            observation_list = []
            for i, rosbag_path in enumerate(rosbag_path_list):
                observation = process(rosbag_path, visualize=False)
                observation_list.append(observation)

            episode = []
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            for i, observation in enumerate(observation_list):
                action = np.zeros(7, dtype=np.float32)
                if i < len(observation_list) - 1:
                    action[:3] = (
                        observation_list[i + 1]["state"][:3] - observation["state"][:3]
                    )
                    action[3:6] = angle_diff(
                        observation_list[i + 1]["state"][3:6], observation["state"][3:6]
                    )
                    action[6] = (
                        observation_list[i + 1]["state"][6] - observation["state"][6]
                    )

                # # compute Kona language embedding
                language_embedding = self._embed([language_instruction])[0].numpy()

                episode.append(
                    {
                        "observation": observation,
                        "action": action,
                        "discount": 1.0,
                        "reward": 0.0,  # float(i == (len(rosbag_path_list) - 1)),
                        "is_first": i == 0,
                        "is_last": i == (len(rosbag_path_list) - 1),
                        "is_terminal": i == (len(rosbag_path_list) - 1),
                        "language_instruction": language_instruction,
                        "language_embedding": language_embedding,
                    }
                )

            metadata = {"file_path": episode_path}
            for i in range(len(EXTRINSICS)):
                metadata[f"intrinsics_{i+1}"] = INTRINSICS
                metadata[f"extrinsics_{i+1}"] = EXTRINSICS[i]

            # create output data sample
            sample = {"steps": episode, "episode_metadata": metadata}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        ps4_drive_prefix = "/media/haochen/Game Drive PS41/robocook/raw_data"
        ps4_drive_tasks = {
            "gripper_sym_rod_robot_v4": "pinch the dough with a two-rod symmetric gripper",
            # "gripper_sym_plane_robot_v4": "pinch the dough with a two-plane symmetric gripper",
            # "gripper_asym_robot_v4": "pinch the dough with an asymmetric gripper",
            # "roller_large_robot_v4": "roll the dough with a large rolling pin",
            # "roller_small_robot_v4": "roll the dough with a small rolling pin",
        }

        wd_drive_prefix = "/media/haochen/wd_drive/robocook/raw_data"
        wd_drive_tasks = {
            # "press_circle_robot_v4": "press the dough with a circle press",
            # "press_square_robot_v4": "press the dough with a square press",
            # "punch_circle_robot_v4": "press the dough with a circle punch",
            # "punch_square_robot_v4": "press the dough with a square punch",
        }

        episode_paths = []
        instructions = []
        for ps4_task, instruction in ps4_drive_tasks.items():
            path = os.path.join(ps4_drive_prefix, ps4_task, "ep_*/seq_*")
            path_list = sorted(glob.glob(path))
            episode_paths += path_list
            instructions += [instruction] * len(path_list)

        for wd_task, instruction in wd_drive_tasks.items():
            path = os.path.join(wd_drive_prefix, wd_task, "ep_*/seq_*")
            path_list = sorted(glob.glob(path))
            episode_paths += path_list
            instructions += [instruction] * len(path_list)

        # # for smallish datasets, use single-thread parsing
        # for sample, instruction in zip(episode_paths, instructions):
        #     yield _parse_example(sample, instruction)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        beam = tfds.core.lazy_imports.apache_beam
        return beam.Create(zip(episode_paths, instructions)) | beam.Map(_parse_example)
