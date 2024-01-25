# Code heavily inspired by https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py.
# Please consider citing their work if you find this code useful. The code is subject to the license available via
# https://github.com/vchoutas/smplx/edit/master/LICENSE

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import pathlib
import pickle

import numpy as np
import torch
import torch.nn as nn

from .lbs import blend_shapes, lbs, vertices2joints, vertices2landmarks
from .obj_io import load_obj

FLAME_PATH = os.environ.get("FLAME_PATH")
s = "FLAME_PATH must contain flame2023.pkl, head_template_mesh.obj and landmark_embedding_with_eyes.npy"
assert FLAME_PATH is not None, "Must set env var FLAME_PATH !\n" + s
FLAME_PATH = pathlib.Path(FLAME_PATH)
assert FLAME_PATH.exists(), "FLAME_PATH: %s does not exist!\n" % FLAME_PATH + s
FLAME_MODEL_PATH = FLAME_PATH / "flame2023.pkl"
FLAME_MESH_PATH = FLAME_PATH / "head_template_mesh.obj"
FLAME_LMK_PATH = FLAME_PATH / "landmark_embedding_with_eyes.npy"


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FlameHead(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(
        self,
        shape_params,
        expr_params,
        flame_model_path=FLAME_MODEL_PATH,
        flame_lmk_embedding_path=FLAME_LMK_PATH,
        flame_template_mesh_path=FLAME_MESH_PATH,
    ):
        super().__init__()

        self.n_shape_params = shape_params
        self.n_expr_params = expr_params

        with open(flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:, :, :shape_params], shapedirs[:, :, 300 : 300 + expr_params]],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        #
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        # Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            flame_lmk_embedding_path, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        # add faces and uvs
        verts, faces, aux = load_obj(flame_template_mesh_path, load_textures=False)

        vertex_uvs = aux.verts_uvs
        face_uvs_idx = faces.textures_idx  # index into verts_uvs

        # create uvcoords per face --> this is what you can use for uv map rendering
        # range from -1 to 1 (-1, -1) = left top; (+1, +1) = right bottom
        # pad 1 to the end
        pad = torch.ones(vertex_uvs.shape[0], 1)
        vertex_uvs = torch.cat([vertex_uvs, pad], dim=-1)
        vertex_uvs = vertex_uvs * 2 - 1
        vertex_uvs[..., 1] = -vertex_uvs[..., 1]

        # face_uv_coords = face_vertices(vertex_uvs[None], face_uvs_idx[None])[0]
        # self.register_buffer("face_uvcoords", face_uv_coords, persistent=False)
        self.register_buffer("faces", faces.verts_idx, persistent=False)

        self.register_buffer("verts_uvs", aux.verts_uvs, persistent=False)
        self.register_buffer("textures_idx", faces.textures_idx, persistent=False)

    def forward(
        self,
        shape,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        translation,
        zero_centered_at_root_node=False,  # otherwise, zero centered at the face
        return_landmarks=True,
        return_verts_cano=False,
        static_offset=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape.shape[0]

        betas = torch.cat([shape.expand(expr.shape[0], -1), expr], dim=1)
        full_pose = torch.cat([rotation, neck, jaw, eyes], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Add shape contribution
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)

        # Add personal offsets
        if static_offset is not None:
            v_shaped += static_offset

        vertices, J, mat_rot = lbs(
            full_pose,
            v_shaped,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )

        if zero_centered_at_root_node:
            vertices = vertices - J[:, [0]]
            J = J - J[:, [0]]

        vertices = vertices + translation[:, None, :]
        J = J + translation[:, None, :]

        ret_vals = [vertices, J]

        if return_verts_cano:
            ret_vals.append(v_shaped)

        # compute landmarks if desired
        if return_landmarks:
            bz = vertices.shape[0]
            landmarks = vertices2landmarks(
                vertices,
                self.faces,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1),
            )
            ret_vals.append(landmarks)

        if len(ret_vals) > 1:
            return ret_vals
        else:
            return ret_vals[0]
