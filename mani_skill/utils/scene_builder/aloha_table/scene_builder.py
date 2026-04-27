"""Scene builder for the ALOHA bimanual workstation.

Recreates the table + aluminum-extrusion T-slot frame from
``mujoco_playground/_src/manipulation/aloha/xmls/mjx_scene.xml``.

The original MuJoCo scene places the table top at ``z ~= 0`` and decorates the
workspace with 31 visual-only frame geoms (``contype=0 conaffinity=0``) that
form the overhead cage, side cameras, and angle braces. We mirror that layout
here as static SAPIEN actors so any tabletop task can swap in the ALOHA agent
and get the recognizable workstation around it.

Quaternions in the source XML are written un-normalized (e.g. ``quat="0 1 0 1"``);
SAPIEN requires unit quats, so each entry below is normalized at construction
time. Both MuJoCo and SAPIEN use ``(w, x, y, z)`` ordering, so values transfer
directly.
"""

import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder


_ASSETS_DIR = Path(osp.dirname(__file__)) / "assets"
_STD_TABLE_GLB = (
    Path(osp.dirname(__file__)).parent / "table" / "assets" / "table.glb"
)

# ALOHA workstation table dimensions. The original mjx_scene.xml has the
# collision plane at size 0.61 x 0.37 x 0.1, but the front 1220 extrusion
# (which holds the wormseye_mount at y=-0.391) and the back 1220 extrusion
# (y=+0.369) bracket a slightly wider footprint. We size the table to span
# exactly between those two rails so each 1220 sits flush on the table edge:
#   front edge at y = -0.391  (front 1220 / wormseye mount)
#   back  edge at y = +0.369  (back 1220)
_ALOHA_TABLE_HALF_X = 0.61   # long axis (matches outer X of the frame)
_ALOHA_TABLE_HALF_Y = 0.38   # span = 0.76m (front 1220 to back 1220)
_ALOHA_TABLE_Y_CENTER = -0.011  # (-0.391 + 0.369) / 2
_ALOHA_TABLE_HEIGHT = 0.75
_ALOHA_TABLE_HALF_Z = _ALOHA_TABLE_HEIGHT / 2

# Robot articulation root sits at the world origin; arm bases live in the MJCF.
ALOHA_RIG_OFFSET = np.array([0.0, 0.0, 0.0])


def _norm_q(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


# (mesh_filename, pos, quat) extracted from xmls/mjx_scene.xml lines 66-96.
# Quats are stored un-normalized exactly as in the source; _norm_q normalizes at build time.
_FRAME_GEOMS = [
    ("extrusion_2040_880.stl", [0.44, -0.361, 1.03], [0, 1, 0, 1]),
    ("extrusion_150.stl", [0.44, -0.371, 0.61], [1, 0, -1, 0]),
    ("d405_solid.stl", [0, -0.303794, 1.02524], [0, 0, 0.976296, 0.21644]),
    ("corner_bracket.stl", [0.44, -0.383, 1.04], [0, 0, -1, 1]),
    ("extrusion_1220.stl", [-0.61, -0.391, -0.01], [0, -1, 0, 1]),
    ("extrusion_150.stl", [-0.59, -0.371, 0.61], [0, -1, 0, 1]),
    ("corner_bracket.stl", [0.42, -0.383, 0.62], [1, 1, 1, -1]),
    ("d405_solid.stl", [0, -0.377167, 0.0316055], [0, 0, -0.672367, -0.740218]),
    ("corner_bracket.stl", [0.61, -0.383, 0.62], [0, 0, 1, -1]),
    ("extrusion_2040_1000.stl", [-0.43, -0.361, 1.02], [0, 0, 0, 1]),
    ("corner_bracket.stl", [-0.61, -0.383, 0.62], [1, 1, 1, -1]),
    ("angled_extrusion.stl", [-0.43, -0.24, 0.12], [0.923, 0.382, 0, 0]),
    ("extrusion_150.stl", [-0.59, -0.066, 0.01], [0, 1, 0, -1]),
    ("extrusion_600.stl", [-0.6, -0.371, 0.62], [0, 0, 0, -1]),
    ("extrusion_150.stl", [0.44, -0.631, 0.01], [1, 0, -1, 0]),
    ("overhead_mount.stl", [0, -0.351, 1.03], [0, 0, 1, 1]),
    ("extrusion_1000.stl", [-0.43, -0.641, 0.01], [1, 1, -1, 1]),
    ("angled_extrusion.stl", [0.6, -0.26, 0.12], [0.923, 0.382, 0, 0]),
    ("extrusion_150.stl", [0.44, -0.066, 0.01], [1, 0, -1, 0]),
    ("corner_bracket.stl", [-0.44, -0.383, 1.04], [1, 1, 1, -1]),
    ("extrusion_1220.stl", [-0.61, 0.369, 0.01], [0, 1, 0, -1]),
    ("extrusion_1000.stl", [0.43, -0.641, 0.01], [0, 0, -1, 1]),
    ("extrusion_1000.stl", [0.6, -0.641, 0.01], [0, 0, -1, 1]),
    ("extrusion_150.stl", [-0.59, -0.631, 0.01], [0, 1, 0, -1]),
    ("corner_bracket.stl", [-0.42, -0.383, 0.62], [0, 0, -1, 1]),
    ("extrusion_1000.stl", [-0.6, -0.641, 0.01], [0, 0, -1, 1]),
    ("extrusion_600.stl", [0.6, -0.371, 0.62], [1, 0, 0, 1]),
    ("angled_extrusion.stl", [0.43, -0.24, 0.12], [0.923, 0.382, 0, 0]),
    ("angled_extrusion.stl", [-0.6, -0.26, 0.12], [0.923, 0.382, 0, 0]),
    ("extrusion_2040_1000.stl", [0.43, -0.361, 1.02], [0, 0, 0, 1]),
    ("wormseye_mount.stl", [0, -0.391, -0.01], [0, 0, 0, 1]),
]


class AlohaTableSceneBuilder(TableSceneBuilder):
    """Adds the ALOHA aluminum-extrusion frame on top of the standard ManiSkill table.

    The robot itself is the ALOHA bimanual setup; the table + ground come from
    :class:`TableSceneBuilder` (same furniture used by Panda/etc. in PickCube).
    Only the overhead T-slot frame, brackets, and side cameras from
    ``mjx_scene.xml`` are added here as static visual actors. The ``aloha`` robot
    sits at the world origin since its MJCF places each arm base at
    ``(+/-0.469, -0.019, 0.02)`` relative to the articulation root, which puts
    them on the table top (z=0).
    """

    def build(self):
        # Custom kinematic table sized to wrap the ALOHA frame footprint
        # (1.22m x 0.76m x 0.75m), reusing the standard ManiSkill wood
        # table.glb visual scaled non-uniformly to those dimensions. Top
        # surface sits at world z=0 to match the rest of ManiSkill's
        # tabletop conventions.
        from transforms3d.euler import euler2quat

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            pose=sapien.Pose(p=[0.0, 0.0, _ALOHA_TABLE_HALF_Z]),
            half_size=(_ALOHA_TABLE_HALF_X, _ALOHA_TABLE_HALF_Y, _ALOHA_TABLE_HALF_Z),
        )
        # Standard ManiSkill table.glb has its long axis along glb-local Y
        # (native dimensions 0.691 x 1.382 x 0.526m, derived from the parent's
        # AABB at scale=1.75). We apply a 90-deg-about-Z visual rotation so the
        # long axis ends up along world X. Scale is applied BEFORE the rotation,
        # so the scale components index glb-local axes:
        #   glb local X (native 0.691) -> world Y after rotation -> 2*half_y
        #   glb local Y (native 1.382) -> world X after rotation -> 2*half_x
        #   glb local Z (native 0.526) -> world Z                -> 2*half_z
        scale_x_local = 2 * _ALOHA_TABLE_HALF_Y / 0.691
        scale_y_local = 2 * _ALOHA_TABLE_HALF_X / 1.382
        scale_z_local = 2 * _ALOHA_TABLE_HALF_Z / 0.526
        builder.add_visual_from_file(
            filename=str(_STD_TABLE_GLB),
            scale=[scale_x_local, scale_y_local, scale_z_local],
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.initial_pose = sapien.Pose(
            p=[0.0, _ALOHA_TABLE_Y_CENTER, -_ALOHA_TABLE_HEIGHT]
        )
        self.table = builder.build_kinematic(name="aloha-table-workspace")

        # Ground at z = -table_height, same convention as TableSceneBuilder.
        from mani_skill.utils.building.ground import build_ground
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-_ALOHA_TABLE_HEIGHT
        )

        self.table_length = 2 * _ALOHA_TABLE_HALF_X
        self.table_width = 2 * _ALOHA_TABLE_HALF_Y
        self.table_height = _ALOHA_TABLE_HEIGHT
        self.scene_objects = [self.table, self.ground]

        # Frame extrusions (visual only; contype=0 conaffinity=0 in source MJCF).
        # Each pose is shifted by ALOHA_RIG_OFFSET so the rig moves with the
        # robot articulation root.
        # Materials match xmls/mjx_scene.xml + xmls/mjx_aloha.xml:
        #   "black" (rgba=0.15 0.15 0.15)  - default for all frame geoms
        #   "metal" (rgba=0.517 0.529 0.537) - only the angled_extrusion supports
        import sapien.render as sr
        mat_black = sr.RenderMaterial(
            base_color=[0.15, 0.15, 0.15, 1.0], roughness=0.6, specular=0.3
        )
        mat_metal = sr.RenderMaterial(
            base_color=[0.517, 0.529, 0.537, 1.0], roughness=0.4, specular=0.6, metallic=0.8
        )

        self.frame_actors = []
        for i, (mesh, pos, quat) in enumerate(_FRAME_GEOMS):
            shifted_pos = (np.asarray(pos, dtype=np.float64) + ALOHA_RIG_OFFSET).tolist()
            material = mat_metal if mesh.startswith("angled_extrusion") else mat_black
            fb = self.scene.create_actor_builder()
            fb.add_visual_from_file(
                filename=str(_ASSETS_DIR / mesh),
                pose=sapien.Pose(p=shifted_pos, q=_norm_q(quat)),
                material=material,
            )
            fb.initial_pose = sapien.Pose(p=[0, 0, 0])
            self.frame_actors.append(
                fb.build_static(name=f"aloha-frame-{i:02d}-{mesh.split('.')[0]}")
            )

        self.scene_objects.extend(self.frame_actors)

    def initialize(self, env_idx: torch.Tensor):
        # Re-anchor the table at its canonical pose (parent's initialize sets
        # this for the standard table; we replicate for our custom one).
        self.table.set_pose(
            sapien.Pose(p=[0.0, _ALOHA_TABLE_Y_CENTER, -_ALOHA_TABLE_HEIGHT])
        )

        if self.env.robot_uids == "aloha":
            qpos = self._aloha_rest_qpos(len(env_idx))
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose(ALOHA_RIG_OFFSET.tolist()))
        else:
            super().initialize(env_idx)

    def _aloha_rest_qpos(self, b: int) -> np.ndarray:
        """Sensible rest qpos for the ALOHA arms.

        Both arms point upward and slightly forward, fingers half-open. Matches
        the spirit of the mujoco_playground 'home' keyframe without copying it
        verbatim (we don't reproduce equality-coupled finger values exactly).
        """
        agent = self.env.agent
        n = len(agent.robot.active_joints)
        qpos = np.zeros((b, n), dtype=np.float32)
        rest_per_joint = {
            "left/waist": 0.0,
            "left/shoulder": -0.96,
            "left/elbow": 1.16,
            "left/forearm_roll": 0.0,
            "left/wrist_angle": -0.3,
            "left/wrist_rotate": 0.0,
            "right/waist": 0.0,
            "right/shoulder": -0.96,
            "right/elbow": 1.16,
            "right/forearm_roll": 0.0,
            "right/wrist_angle": -0.3,
            "right/wrist_rotate": 0.0,
            "left/left_finger": 0.02,
            "left/right_finger": 0.02,
            "right/left_finger": 0.02,
            "right/right_finger": 0.02,
        }
        for jname, val in rest_per_joint.items():
            j = agent.robot.active_joints_map.get(jname)
            if j is None or j.active_index is None:
                continue
            idx = j.active_index[0].item()
            qpos[:, idx] = val
        if self.robot_init_qpos_noise > 0:
            noise = self.env._episode_rng.normal(
                0, self.robot_init_qpos_noise, qpos.shape
            ).astype(np.float32)
            # Don't add noise to fingers (they'd push past the [0.002, 0.037] ctrl range).
            for fname in ("left/left_finger", "left/right_finger",
                          "right/left_finger", "right/right_finger"):
                j = agent.robot.active_joints_map.get(fname)
                if j is not None and j.active_index is not None:
                    noise[:, j.active_index[0].item()] = 0.0
            qpos = qpos + noise
        return qpos
