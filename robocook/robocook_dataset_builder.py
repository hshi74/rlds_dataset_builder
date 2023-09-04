import copy
import glob
import json
import os
from time import time
from typing import Any, Iterator, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rosbag
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from parse_point_cloud import EXTRINSICS, INTRINSICS
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import quat2euler
from transforms3d.quaternions import *


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):
    data_dict = {}

    for key in hdf5_file.keys():
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(
                curr_data, index, keys_to_ignore=keys_to_ignore
            )
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict


def angle_diff(target, source, degrees=False):
    target_rot = R.from_euler("xyz", target, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")


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
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Camera 1 RGB observation.",
                                    ),
                                    "image_2": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Camera 2 RGB observation.",
                                    ),
                                    "image_3": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Camera 3 RGB observation.",
                                    ),
                                    "image_4": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Camera 4 RGB observation.",
                                    ),
                                    "depth_1": tfds.features.Tensor(
                                        shape=(256, 256),
                                        dtype=np.float32,
                                        doc="Camera 1 Depth observation.",
                                    ),
                                    "depth_2": tfds.features.Tensor(
                                        shape=(256, 256),
                                        dtype=np.float32,
                                        doc="Camera 2 Depth observation.",
                                    ),
                                    "depth_3": tfds.features.Tensor(
                                        shape=(256, 256),
                                        dtype=np.float32,
                                        doc="Camera 3 Depth observation.",
                                    ),
                                    "depth_4": tfds.features.Tensor(
                                        shape=(256, 256),
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
                            # "intrinsics_1": tfds.features.Tensor(
                            #     shape=(3, 3),
                            #     dtype=np.float32,
                            #     doc="Camera 1 Intrinsic Matrix.",
                            # ),
                            # "intrinsics_2": tfds.features.Tensor(
                            #     shape=(3, 3),
                            #     dtype=np.float32,
                            #     doc="Camera 2 Intrinsic Matrix.",
                            # ),
                            # "intrinsics_3": tfds.features.Tensor(
                            #     shape=(3, 3),
                            #     dtype=np.float32,
                            #     doc="Camera 3 Intrinsic Matrix.",
                            # ),
                            # "intrinsics_4": tfds.features.Tensor(
                            #     shape=(3, 3),
                            #     dtype=np.float32,
                            #     doc="Camera 4 Intrinsic Matrix.",
                            # ),
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

        def _parse_example(episode_info):
            # load raw data --> this should change for your dataset
            # data = np.load(
            #     episode_path, allow_pickle=True
            # )  # this is a list of dicts in our case

            # for i, step in enumerate(data):

            episode_path, language_instruction = episode_info

            h5_path_list = sorted(glob.glob(os.path.join(episode_path, "*.h5")))

            observation_list = []
            for i, h5_path in enumerate(h5_path_list):
                h5_file = h5py.File(h5_path, "r")
                observation = load_hdf5_to_dict(h5_file, 0)
                h5_file.close()
                observation_new = {}
                for key, value in observation.items():
                    if key.startswith("image") or key.startswith("depth"):
                        observation_new[key] = np.array(
                            Image.fromarray(value).resize(
                                (256, 256), Image.Resampling.LANCZOS
                            )
                        )
                    else:
                        observation_new[key] = value
                observation_list.append(observation_new)

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
                        "is_last": i == (len(h5_path_list) - 1),
                        "is_terminal": i == (len(h5_path_list) - 1),
                        "language_instruction": language_instruction,
                        "language_embedding": language_embedding,
                    }
                )

            metadata = {"file_path": episode_path}
            for i in range(len(EXTRINSICS)):
                # metadata[f"intrinsics_{i+1}"] = INTRINSICS
                metadata[f"extrinsics_{i+1}"] = EXTRINSICS[i]

            # create output data sample
            sample = {"steps": episode, "episode_metadata": metadata}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        seagate_drive_prefix = "/media/haochenshi/seagate/robocook/cache"
        seagate_drive_tasks = {
            "gripper_sym_rod_robot_v4": "pinch the dough with a two-rod symmetric gripper",
            "gripper_sym_plane_robot_v4": "pinch the dough with a two-plane symmetric gripper",
            "gripper_asym_robot_v4": "pinch the dough with an asymmetric gripper",
            "roller_large_robot_v4": "roll the dough with a large rolling pin",
            "roller_small_robot_v4": "roll the dough with a small rolling pin",
            "press_circle_robot_v4": "press the dough with a circle press",
            "press_square_robot_v4": "press the dough with a square press",
            "punch_circle_robot_v4": "press the dough with a circle punch",
            "punch_square_robot_v4": "press the dough with a square punch",
        }

        episode_paths = []
        instructions = []
        for seagate_task, instruction in seagate_drive_tasks.items():
            path = os.path.join(seagate_drive_prefix, seagate_task, "ep_*/seq_*")
            path_list = sorted(glob.glob(path))
            episode_paths += path_list
            instructions += [instruction] * len(path_list)

        episode_info_list = list(zip(episode_paths, instructions))

        # for smallish datasets, use single-thread parsing
        for episode_info in episode_info_list:
            yield _parse_example(episode_info)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(episode_info_list) | beam.Map(_parse_example)
