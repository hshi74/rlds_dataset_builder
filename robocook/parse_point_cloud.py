import copy
import glob
import json
import os
from typing import Any, Iterator, Tuple

import boto3
import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rosbag
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from transforms3d.euler import quat2euler
from transforms3d.quaternions import *

DEPTH_OPTICAL_FRAME_POSE = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]

with open(
    os.path.join("/home/haochenshi/projects/rlds_dataset_builder/cam_pose.json"), "r"
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


def write_dict_to_hdf5(hdf5_file, data_dict, keys_to_ignore=[]):
    for key in data_dict.keys():
        # Pass Over Specified Keys #
        if key in keys_to_ignore:
            continue

        # Examine Data #
        curr_data = data_dict[key]
        if type(curr_data) == list:
            curr_data = np.array(curr_data)
        dtype = type(curr_data)

        # Unwrap If Dictionary #
        if dtype == dict:
            if key not in hdf5_file:
                hdf5_file.create_group(key)
            write_dict_to_hdf5(hdf5_file[key], curr_data)
            continue

        # Make Room For Data #
        if key not in hdf5_file:
            if dtype != np.ndarray:
                dshape = ()
            else:
                dtype, dshape = curr_data.dtype, curr_data.shape
            hdf5_file.create_dataset(
                key, (1, *dshape), maxshape=(None, *dshape), dtype=dtype
            )
        else:
            hdf5_file[key].resize(hdf5_file[key].shape[0] + 1, axis=0)

        # Save Data #
        hdf5_file[key][-1] = curr_data


def add_angles(delta, source, degrees=False):
    delta_rot = R.from_euler("xyz", delta, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)


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
    hdf5_dir = os.path.join(
        "/home/haochenshi/projects/rlds_dataset_builder/robocook/cache",
        *rosbag_path.split("/")[-4:-1],
    )
    os.makedirs(hdf5_dir, exist_ok=True)

    hdf5_path = os.path.join(
        hdf5_dir,
        ".".join([*rosbag_path.split("/")[-1].split(".")[:-1], "h5"]),
    )
    if os.path.exists(hdf5_path):
        return

    hdf5_file = h5py.File(hdf5_path, "w")

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

    write_dict_to_hdf5(hdf5_file, observation)
    hdf5_file.close()

    if visualize:
        o3d.visualization.draw_geometries([merge_pc])

        figure, axes = plt.subplots(len(pc_msgs), 2, figsize=(16, 16))
        for i in range(len(pc_msgs)):
            axes[i][0].imshow(observation[f"image_{i+1}"])
            axes[i][1].imshow(observation[f"depth_{i+1}"])

        plt.show()
        plt.close()

    return observation


if __name__ == "__main__":
    # create list of all examples
    ps4_drive_prefix = "/media/haochenshi/Game Drive PS4/robocook/raw_data"
    ps4_drive_tasks = {
        "gripper_sym_rod_robot_v4": "pinch the dough with a two-rod symmetric gripper",
        "gripper_sym_plane_robot_v4": "pinch the dough with a two-plane symmetric gripper",
        "gripper_asym_robot_v4": "pinch the dough with an asymmetric gripper",
        "roller_large_robot_v4": "roll the dough with a large rolling pin",
        "roller_small_robot_v4": "roll the dough with a small rolling pin",
    }

    wd_drive_prefix = "/media/haochenshi/wd_drive/robocook/raw_data"
    wd_drive_tasks = {
        "press_circle_robot_v4": "press the dough with a circle press",
        "press_square_robot_v4": "press the dough with a square press",
        "punch_circle_robot_v4": "press the dough with a circle punch",
        "punch_square_robot_v4": "press the dough with a square punch",
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

    for i in tqdm(range(len(episode_paths))):
        rosbag_path_list = sorted(glob.glob(os.path.join(episode_paths[i], "*.bag")))

        observation_list = []
        for j, rosbag_path in enumerate(rosbag_path_list):
            process(rosbag_path, visualize=False)
