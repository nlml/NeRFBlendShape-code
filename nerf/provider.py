from operator import index
import os
import time
import glob
from typing import Literal
import numpy as np
from copy import deepcopy

import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from scipy.spatial.transform import Slerp, Rotation

import json
import math

from .so3_exp_map import so3_exp_map as _so3_exp_map
from .flame.flame import FlameHead


def nerf_matrix_to_ngp(pose, scale=0.33):
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
            [0, 0, 0, 1],
        ],
        dtype=pose.dtype,
    )
    return new_pose


class NeRFDataset(Dataset):
    def __init__(
        self,
        data_folder,
        basis_num,
        type:Literal['train', 'val', 'test', 'normal_test']="train",
        downscale=1,
        downsample=1,
        test_basis_inter=1,
        add_mean=False,
        to_mem=False,
        rot_cycle=100,
        apply_neck_rot_to_flame_pose=True,
        apply_flame_poses_to_cams=True,
        neck_pose_to_expr=False,
        eyes_pose_to_expr=False,
    ):
        super().__init__()
        self.apply_neck_rot_to_flame_pose = apply_neck_rot_to_flame_pose
        self.apply_flame_poses_to_cams = apply_flame_poses_to_cams
        self.neck_pose_to_expr = neck_pose_to_expr
        self.eyes_pose_to_expr = eyes_pose_to_expr

        # path: the json file path.
        self.data_folder = data_folder
        self.basis_num = basis_num
        self.add_mean = add_mean
        self.type = type
        self.downscale = downscale
        self.to_mem = to_mem
        self.test_basis_inter = test_basis_inter
        max_path = os.path.join(
            self.data_folder, f"max_{self.basis_num}.txt"
        )
        min_path = os.path.join(
            self.data_folder, f"min_{self.basis_num}.txt"
        )
        self.load_max(max_path, min_path)

        # load nerf-compatible format data.
        transform_file = f'transforms_nb_{type}.json' if type != 'normal_test' else 'transforms_nb_test.json'
        with open(os.path.join(self.data_folder, transform_file), "r") as f:
            transform_data = json.load(f)

        self.bc_img = np.ones([transform_data["h"], transform_data["w"], 3]).astype(
            np.float32
        )

        self.H = int(transform_data["h"]) // downscale
        self.W = int(transform_data["w"]) // downscale
        self.intrinsics = []
        self.poses = []
        self.images_list = []
        self.mask_paths_list = []
        self.exps = []
        self.parsings = []
        self.lms = []
        flame_model = FlameHead(300, 100)
        extra_max = None
        extra_min = None

        frames = transform_data['frames'][::downsample]
        self.num_frames = len(frames)
        for frame in frames:
            # load image size

            # load intrinsics
            intrinsic = np.eye(3, dtype=np.float32)
            intrinsic[0, 0] = transform_data["fl_x"] / downscale
            intrinsic[1, 1] = transform_data["fl_y"] / downscale
            intrinsic[0, 2] = transform_data["cx"] / downscale
            intrinsic[1, 2] = transform_data["cy"] / downscale
            self.intrinsics.append(intrinsic)

            img_path = os.path.join(self.data_folder, frame["file_path"])
                # f_path = os.path.join(
                #     self.root_img, "head_imgs", str(frames_img[f_id]["img_id"]) + ".jpg"
                # )
            parsing_path = None
                # parsing_path = os.path.join(
                #     self.root_img, "parsing", str(frames_img[f_id]["img_id"]) + ".png"
                # )
            lms_path = None
                # lms_path = os.path.join(
                #     self.root_img, "ori_imgs", str(frames_img[f_id]["img_id"]) + ".lms"
                # )
            
            assert os.path.exists(img_path)

            camera_pose = np.array(
                frame["transform_matrix"], dtype=np.float32
            )  # [4, 4]
            # Here we should also apply the FLAME rotation to the pose.

            mask_path = os.path.join(self.data_folder, frame["fg_mask_path"])
            flame_param_fname = os.path.join(
                self.data_folder, frame["flame_param_path"]
            )
            fp = np.load(flame_param_fname)
            R = _so3_exp_map(torch.from_numpy(fp["rotation"]).float())[0]  # [3, 3]
            T = torch.from_numpy(fp["translation"]).float()[0]  # [3,]

            if self.apply_neck_rot_to_flame_pose:
                # neck_rot = _so3_exp_map(torch.from_numpy(fp["neck_pose"]).float())[0]
                # R = R @ neck_rot

                verts, posed_joints, landmarks = flame_model(
                    torch.tensor(fp["shape"][None, ...]).float(),
                    torch.tensor(fp["expr"]).float(),
                    torch.tensor(fp["rotation"]).float(),
                    torch.tensor(fp["neck_pose"]).float(),
                    torch.tensor(fp["jaw_pose"]).float(),
                    torch.tensor(fp["eyes_pose"]).float(),
                    torch.tensor(fp["translation"]).float(),
                    return_landmarks=True,
                )

                neck_translation = posed_joints[:, 1]
                T = neck_translation[...][0].float()
                assert list(T.shape) == [3], T.shape

                neck_rot = _so3_exp_map(torch.from_numpy(fp["neck_pose"]).float())[0]
                R = R @ neck_rot

            if self.apply_flame_poses_to_cams:
                # Compute inverse flame pose
                Rflameinv = torch.eye(4, dtype=torch.float32)
                Rflameinv[:3, :3] = R.T
                Rflameinv[:3, 3] = -Rflameinv[:3, :3] @ T
                # Shift the camera by the inverse flame pose
                Rcam = torch.from_numpy(camera_pose)
                camera_pose = (Rflameinv @ Rcam)[:3]
                # Set the flame pose to identity since it has been applied to the camera
                R = torch.eye(3, dtype=torch.float32)
                T = torch.zeros(3, 1, dtype=torch.float32)
                camera_pose = camera_pose.numpy()

            camera_pose = nerf_matrix_to_ngp(camera_pose)

            # Adding neck pose and eye pose to expression
            if not self.neck_pose_to_expr:
                maybe_neck_pose = []
            else:
                maybe_neck_pose = [fp["neck_pose"]]
            # Add eyes pose to expression
            maybe_eyes_pose = [fp["eyes_pose"]] if self.eyes_pose_to_expr else []
            exp_code = maybe_neck_pose + maybe_eyes_pose + [fp["jaw_pose"], fp["expr"]]
            exp_code = np.concatenate(exp_code, axis=1)[0].tolist()[: self.basis_num]

            # For the new components we are adding to expression, we need to keep track of the max, min
            # And later we will add these elements to self.maxper and self.minper
            new_max = np.concatenate(
                maybe_neck_pose + maybe_eyes_pose + [fp["jaw_pose"]], axis=1
            )[0]
            if extra_max is None:
                extra_max = new_max
            else:
                extra_max = np.maximum(extra_max, new_max)
            new_min = np.concatenate(
                maybe_neck_pose + maybe_eyes_pose + [fp["jaw_pose"]], axis=1
            )[0]
            if extra_min is None:
                extra_min = new_min
            else:
                extra_min = np.minimum(extra_min, new_min)

            if add_mean == True:
                exp_code.insert(0, 1.0)
            exp = np.array(exp_code, dtype=np.float32)
            if lms_path is not None:
                lms = np.loadtxt(lms_path)
            else:
                lms = None

            self.poses.append(camera_pose)
            self.images_list.append(img_path)
            self.exps.append(exp)
            self.parsings.append(parsing_path)
            self.lms.append(lms)
            if mask_path is not None:
                self.mask_paths_list.append(mask_path)

        self.load_max(max_path, min_path, extra_max=extra_max, extra_min=extra_min)

        # print(self.poses)
        self.poses = np.stack(self.poses, axis=0).astype(np.float32)
        self.exps = np.stack(self.exps, axis=0).astype(np.float32)
        print("poses shape is :")
        print(self.poses.shape)
        print("exps shape is :")
        print(self.exps.shape)

        if self.lms[0] is not None:
            self.lms = np.stack(self.lms, axis=0).astype(np.int32)  # (N,478,2)
        else:
            self.lms = None

        if self.lms is not None:
            if self.type == "normal_test":
                self.rects, self.rects_mouth, self.rects_eyes = self.get_rect_test(
                    self.lms, self.W, self.H
                )
            else:
                self.rects, self.rects_mouth, self.rects_eyes = self.get_rect(
                    self.lms, self.W, self.H
                )
        else:
            self.rects, self.rects_mouth, self.rects_eyes = None, None, None

        if self.to_mem == True and self.type!= 'normal_test':
            self.mem_images = []
            self.mem_masks = []
            for index in range(self.num_frames):
                img_path = self.images_list[index]
                image = cv2.imread(
                    img_path, cv2.IMREAD_UNCHANGED
                )  # [H, W, 3] o [H, W, 4]
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                image = cv2.resize(
                    image, (self.W, self.H), interpolation=cv2.INTER_AREA
                )
                image = image.astype(np.uint8)  # [H, W, 3/4]
                self.mem_images.append(image)

                parsing_path = self.parsings[index]
                if parsing_path is None:
                    # we have masks, load them up
                    mask_path = self.mask_paths_list[index]
                    mask = cv2.imread(
                        os.path.join(self.data_folder, self.mask_paths_list[index]),
                        cv2.IMREAD_UNCHANGED,
                    )
                    mask = mask > 200
                else:
                    seg = cv2.imread(
                        parsing_path, cv2.IMREAD_UNCHANGED
                    )  # [H, W, 3] o [H, W, 4]
                    if seg.shape[-1] == 3:
                        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
                    else:
                        seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2RGBA)
                    seg = cv2.resize(
                        seg, (self.W, self.H), interpolation=cv2.INTER_AREA
                    )
                    mask = (
                        (seg[:, :, 0] == 0)
                        * (seg[:, :, 1] == 0)
                        * (seg[:, :, 2] == 255)
                    )
                self.mem_masks.append(mask)
            

        if self.type == "normal_test":
            self.poses_l1 = []

            vec = np.array([0, 0, 1])
            for i in range(self.num_frames):
                tmp_pose = np.identity(4, dtype=np.float32)
                r1 = Rotation.from_euler(
                    "y", 15 + (-30) * ((i % rot_cycle) / rot_cycle), degrees=True
                )
                tmp_pose[:3, :3] = r1.as_matrix()
                trans = tmp_pose[:3, :3] @ vec
                # print(trans)
                tmp_pose[0:3, 3] = trans
                self.poses_l1.append(nerf_matrix_to_ngp(tmp_pose))
            return

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        results = {
            "pose": self.poses[index],
            "intrinsic": self.intrinsics[index],
            "index": index,
            "exp": self.exps[index],
            "image_name": self.images_list[index].split('/')[-1]
        }
        results["H"] = str(self.H)
        results["W"] = str(self.W)

        if self.to_mem == False and self.type!='normal_test':
            if self.rects is not None:
                results["rects"] = self.rects[index]
                results["rects_mouth"] = self.rects_mouth[index]
            # else:
            #     results["rects"] = None
            #     results["rects_mouth"] = None
            f_path = self.images_list[index]
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255
            results["image"] = image

            parsing_path = self.parsings[index]
            seg = cv2.imread(parsing_path, cv2.IMREAD_UNCHANGED)
            if seg.shape[-1] == 3:
                seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
            else:
                seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2RGBA)
            seg = cv2.resize(seg, (self.W, self.H), interpolation=cv2.INTER_AREA)
            mask = (seg[:, :, 0] == 0) * (seg[:, :, 1] == 0) * (seg[:, :, 2] == 255)
            results["mask"] = mask
            return results
        elif self.to_mem == True and self.type!='normal_test':
            if self.rects is not None:
                results["rects"] = self.rects[index]
                results["rects_mouth"] = self.rects_mouth[index]
            # else:
            #     results["rects"] = None
            #     results["rects_mouth"] = None

            image = self.mem_images[index]
            results["image"] = image.astype(np.float32) / 255

            mask = self.mem_masks[index]
            results["mask"] = mask
            return results
        elif self.type == "normal_test":
            results["pose_l1"] = self.poses_l1[index]

            if self.rects is not None:
                results["rects"] = self.rects[index]
                results["rects_mouth"] = self.rects_mouth[index]
                results["rects_eyes"] = self.rects_eyes[index]
            # else:
            #     results["rects"] = None
            #     results["rects_mouth"] = None
            #     results["rects_eyes"] = None
            f_path = self.images_list[index]
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255
            results["image"] = image
            return results
        else:
            return results

    def get_rect(self, lms, W, H):
        max_w = np.max(lms[:, :, 0], axis=1)
        min_w = np.min(lms[:, :, 0], axis=1)
        max_h = np.max(lms[:, :, 1], axis=1)
        min_h = np.min(lms[:, :, 1], axis=1)
        w0 = (max_w + min_w) / 2
        h0 = (max_h + min_h) / 2
        radius = 1.2 * np.sqrt((max_w - min_w) ** 2 + (max_h - min_h) ** 2) / 2  # 1.1
        w1 = w0 - radius
        h1 = h0 - radius * 1.2
        w2 = w0 + radius
        h2 = h0 + radius * 0.8
        rect = np.stack([w1, h1, w2, h2], axis=1).astype(np.int32)
        rect[:, [0, 2]] = np.clip(rect[:, [0, 2]], 1, self.W - 1)
        rect[:, [1, 3]] = np.clip(rect[:, [1, 3]], 1, self.H - 1)

        mouth_l, mouth_r, mouth_t, mouth_b = (
            lms[:, 57, 0],
            lms[:, 287, 0],
            lms[:, 0, 1],
            lms[:, 17, 1],
        )
        eye_l, eye_r, eye_t, eye_b = (
            lms[:, 130, 0],
            lms[:, 359, 0],
            np.minimum(lms[:, 27, 1], lms[:, 257, 1]),
            np.maximum(lms[:, 23, 1], lms[:, 253, 1]),
        )

        rect_mouth = np.stack([mouth_l, mouth_t, mouth_r, mouth_b], axis=1).astype(
            np.int32
        )
        rect_eye = np.stack([eye_l, eye_t, eye_r, eye_b], axis=1).astype(np.int32)

        rect_mouth[:, [0, 2]] = np.clip(rect_mouth[:, [0, 2]], 1, self.W - 1)
        rect_mouth[:, [1, 3]] = np.clip(rect_mouth[:, [1, 3]], 1, self.H - 1)

        rect_eye[:, [0, 2]] = np.clip(rect_eye[:, [0, 2]], 1, self.W - 1)
        rect_eye[:, [1, 3]] = np.clip(rect_eye[:, [1, 3]], 1, self.H - 1)

        rect_sp = rect_mouth
        rect_sp2 = rect_eye

        return rect, rect_sp, rect_sp2

    def get_rect_test(self, lms, W, H, scale=1.2):
        max_w = np.max(lms[:, :, 0], axis=1)
        min_w = np.min(lms[:, :, 0], axis=1)
        max_h = np.max(lms[:, :, 1], axis=1)
        min_h = np.min(lms[:, :, 1], axis=1)
        w0 = (max_w + min_w) / 2
        h0 = (max_h + min_h) / 2
        radius = scale * np.sqrt((max_w - min_w) ** 2 + (max_h - min_h) ** 2) / 2
        w1 = w0 - radius
        h1 = h0 - radius * 1.2
        w2 = w0 + radius
        h2 = h0 + radius * 0.8
        rect = np.stack([w1, h1, w2, h2], axis=1).astype(np.int32)
        rect[:, [0, 2]] = np.clip(rect[:, [0, 2]], 1, self.W - 1)
        rect[:, [1, 3]] = np.clip(rect[:, [1, 3]], 1, self.H - 1)

        mouth_l, mouth_r, mouth_t, mouth_b = (
            lms[:, 57, 0],
            lms[:, 287, 0],
            lms[:, 0, 1],
            lms[:, 17, 1],
        )
        eye_l, eye_r, eye_t, eye_b = (
            lms[:, 130, 0],
            lms[:, 359, 0],
            np.minimum(lms[:, 27, 1], lms[:, 257, 1]),
            np.maximum(lms[:, 23, 1], lms[:, 253, 1]),
        )

        radius_mouth = (
            0.5 * np.sqrt((mouth_r - mouth_l) ** 2 + (mouth_b - mouth_t) ** 2) / 2
        )
        radius_eye = 0.3 * np.sqrt((eye_r - eye_l) ** 2 + (eye_b - eye_t) ** 2) / 2
        mouth_l = mouth_l - radius_mouth
        mouth_t = mouth_t - radius_mouth
        mouth_r = mouth_r + radius_mouth
        mouth_b = mouth_b + radius_mouth
        eye_l = eye_l - radius_eye
        eye_t = eye_t - radius_eye
        eye_r = eye_r + radius_eye
        eye_b = eye_b + radius_eye

        rect_mouth = np.stack([mouth_l, mouth_t, mouth_r, mouth_b], axis=1).astype(
            np.int32
        )
        rect_eye = np.stack([eye_l, eye_t, eye_r, eye_b], axis=1).astype(np.int32)
        rect_mouth[:, [0, 2]] = np.clip(rect_mouth[:, [0, 2]], 1, self.W - 1)
        rect_mouth[:, [1, 3]] = np.clip(rect_mouth[:, [1, 3]], 1, self.H - 1)
        rect_eye[:, [0, 2]] = np.clip(rect_eye[:, [0, 2]], 1, self.W - 1)
        rect_eye[:, [1, 3]] = np.clip(rect_eye[:, [1, 3]], 1, self.H - 1)
        rect_sp = rect_mouth
        rect_sp2 = rect_eye

        return rect, rect_sp, rect_sp2

    def load_max(self, max_path, min_path, extra_max=None, extra_min=None):
        maxper = np.loadtxt(max_path)
        if extra_max is not None:
            maxper = np.concatenate([extra_max, maxper], axis=0)
        self.max_per = (
            torch.from_numpy(maxper)[: self.basis_num].to(dtype=torch.float16).cuda()
        )
        # self.load_max(max_path, min_path, extra_max=extra_max, extra_min=extra_min)
        if self.add_mean:
            self.max_per = torch.cat([torch.ones([1]).cuda(), self.max_per], dim=0)

        print(f"load max_per successfully (shape = {self.max_per.shape}):")
        print(self.max_per)

        minper = np.loadtxt(min_path)
        if extra_min is not None:
            minper = np.concatenate([extra_min, minper], axis=0)
        self.min_per = (
            torch.from_numpy(minper)[: self.basis_num].to(dtype=torch.float16).cuda()
        )
        if self.add_mean:
            self.min_per = torch.cat([torch.ones([1]).cuda(), self.min_per], dim=0)
        print(f"load min_per successfully (shape = {self.min_per.shape}):")
        print(self.min_per)
