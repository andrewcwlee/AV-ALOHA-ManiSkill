"""ALOHA bimanual peg insertion ported from mujoco_playground.

Source: ``mujoco_playground/_src/manipulation/aloha/single_peg_insertion.py``
plus ``xmls/mjx_single_peg_insertion.xml``.

Bimanual task: the left arm picks up a hollow socket, the right arm picks up
a red peg, and the right arm inserts the peg into the socket through the
socket's open +X face.
"""

from typing import Any

import numpy as np
import sapien
import sapien.render as sr
import torch

from mani_skill.agents.robots import Aloha
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.aloha_table.scene_builder import (
    AlohaTableSceneBuilder,
)
from mani_skill.utils.structs.pose import Pose


# Reward scales transcribed from default_config().reward_config.scales.
_REWARD_SCALES = dict(
    left_reward=1.0,
    right_reward=1.0,
    left_target_qpos=0.3,
    right_target_qpos=0.3,
    no_table_collision=0.3,
    socket_z_up=0.5,
    peg_z_up=0.5,
    socket_entrance_reward=4.0,
    peg_end2_reward=4.0,
    peg_insertion_reward=8.0,
)


# Initial qpos for both arms taken from the "home" keyframe in the source MJCF
# (xmls/mjx_single_peg_insertion.xml). 12 arm joints + 4 fingers, in the joint
# order used by the Aloha agent class (left_arm, right_arm, then finger pairs).
_HOME_LEFT_ARM = np.array(
    [0.083383, -0.122008, 0.950168, 0.108187, -0.869224, -0.0731298]
)
_HOME_RIGHT_ARM = np.array(
    [-0.0862348, -0.109522, 0.949474, -0.113041, -0.887378, 0.0754333]
)
_HOME_LEFT_GRIP = 0.0305
_HOME_RIGHT_GRIP = 0.0186


def _build_socket(scene, name="socket"):
    """5-piece hollow box. Open face at +X (entrance), closed back at -X."""
    builder = scene.create_actor_builder()
    blue = sr.RenderMaterial(base_color=[0.2, 0.4, 0.85, 1.0])
    red = sr.RenderMaterial(base_color=[0.9, 0.1, 0.1, 1.0])

    # Each tuple: (local_pos, half_size, material).
    parts = [
        ((0.0,  0.0, -0.020), (0.048, 0.022, 0.002), blue),  # bottom
        ((0.0,  0.0, +0.020), (0.048, 0.022, 0.002), blue),  # top
        ((0.0, -0.020, 0.0), (0.048, 0.002, 0.018), blue),   # left wall
        ((0.0, +0.020, 0.0), (0.048, 0.002, 0.018), blue),   # right wall
        ((-0.044, 0.0, 0.0), (0.004, 0.018, 0.018), red),    # closed back wall
    ]
    for pos, half, mat in parts:
        local_pose = sapien.Pose(p=pos)
        builder.add_box_collision(pose=local_pose, half_size=half)
        builder.add_box_visual(pose=local_pose, half_size=half, material=mat)
    builder.initial_pose = sapien.Pose(p=[-0.147, 0.0, 0.023])
    return builder.build(name=name)


def _build_peg(scene, name="peg"):
    builder = scene.create_actor_builder()
    half = (0.048, 0.01, 0.01)
    red = sr.RenderMaterial(base_color=[0.9, 0.1, 0.1, 1.0])
    builder.add_box_collision(half_size=half)
    builder.add_box_visual(half_size=half, material=red)
    builder.initial_pose = sapien.Pose(p=[0.136, 0.0, 0.011])
    return builder.build(name=name)


@register_env("PegInsertionAloha-v1", max_episode_steps=200)
class PegInsertionAlohaEnv(BaseEnv):
    """Bimanual peg-insertion task on the ALOHA workstation.

    **Randomizations:**
    - The peg's xy is offset from its home pose by ~U[-0.1, 0.1]^2.
    - The socket's xy is offset from its home pose by ~U[-0.1, 0.1]^2.

    **Success:**
    The peg's insertion end (peg_end2) is within ``insertion_thresh`` of the
    socket interior (socket_rear site).
    """

    SUPPORTED_ROBOTS = ["aloha"]
    agent: Aloha

    insertion_thresh = 0.01            # peg_end2 within 10mm of socket interior
    insertion_align_thresh = 0.005     # peg_end2 within 5mm of socket axis to gate the reward
    randomization_range = 0.1          # +/- this in XY for peg/socket reset

    # Goal positions ("lift" targets) used in the reward shaping.
    socket_goal_pos = np.array([-0.05, 0.0, 0.15])
    peg_goal_pos = np.array([+0.05, 0.0, 0.15])

    # Local-frame offsets of the named sites from the source MJCF.
    SOCKET_ENTRANCE_LOCAL = np.array([+0.048, 0.0, 0.0])
    SOCKET_REAR_LOCAL = np.array([0.0, 0.0, 0.0])
    PEG_END2_LOCAL = np.array([-0.048, 0.0, 0.0])

    def __init__(self, *args, robot_uids="aloha", robot_init_qpos_noise=0.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # The agent already exposes overhead/worms-eye/wrist cams; just reuse
        # an over-the-shoulder render camera as the task's primary sensor view.
        pose = sapien_utils.look_at(eye=[0.0, -0.45, 0.55], target=[0.0, 0.0, 0.10])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.55], target=[0.0, 0.0, 0.10])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # MJCF places each arm base at (+/-0.469, -0.019, 0.02); articulation
        # root sits at the world origin.
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = AlohaTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.socket = _build_socket(self.scene)
        self.peg = _build_peg(self.scene)

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Place table + arms at home keyframe.
            self.table_scene.initialize(env_idx)
            qpos = self._home_qpos(b)
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([0, 0, 0]))

            # Randomize socket/peg XY around their home positions.
            socket_xy_off = (torch.rand(b, 2) - 0.5) * 2 * self.randomization_range
            peg_xy_off = (torch.rand(b, 2) - 0.5) * 2 * self.randomization_range

            socket_pos = torch.zeros(b, 3)
            socket_pos[:, 0] = -0.147 + socket_xy_off[:, 0]
            socket_pos[:, 1] = 0.0 + socket_xy_off[:, 1]
            socket_pos[:, 2] = 0.023
            self.socket.set_pose(Pose.create_from_pq(p=socket_pos))

            peg_pos = torch.zeros(b, 3)
            peg_pos[:, 0] = 0.136 + peg_xy_off[:, 0]
            peg_pos[:, 1] = 0.0 + peg_xy_off[:, 1]
            peg_pos[:, 2] = 0.011
            self.peg.set_pose(Pose.create_from_pq(p=peg_pos))

    def _home_qpos(self, b: int) -> np.ndarray:
        """Construct the home-keyframe qpos for a batch of envs."""
        agent = self.agent
        n = len(agent.robot.active_joints)
        qpos = np.zeros((b, n), dtype=np.float32)
        per_joint = {
            **{n: v for n, v in zip(agent.left_arm_joint_names, _HOME_LEFT_ARM)},
            **{n: v for n, v in zip(agent.right_arm_joint_names, _HOME_RIGHT_ARM)},
            "left/left_finger": _HOME_LEFT_GRIP,
            "left/right_finger": _HOME_LEFT_GRIP,
            "right/left_finger": _HOME_RIGHT_GRIP,
            "right/right_finger": _HOME_RIGHT_GRIP,
        }
        for jname, val in per_joint.items():
            j = agent.robot.active_joints_map.get(jname)
            if j is None or j.active_index is None:
                continue
            idx = j.active_index[0].item()
            qpos[:, idx] = float(val)
        if self.robot_init_qpos_noise > 0:
            qpos = qpos + self._episode_rng.normal(
                0, self.robot_init_qpos_noise, qpos.shape
            ).astype(np.float32)
        return qpos

    # ------------------------------------------------------------------ #
    # Pose helpers
    # ------------------------------------------------------------------ #
    def _site_pos(self, actor, local_offset_np: np.ndarray) -> torch.Tensor:
        """Return the world position of a site defined as a local offset."""
        local = torch.tensor(local_offset_np, dtype=torch.float32, device=self.device)
        # actor.pose.to_transformation_matrix() is (B, 4, 4)
        T = actor.pose.to_transformation_matrix()
        local_h = torch.cat([local, torch.ones(1, device=self.device)])
        # broadcast: T (B,4,4) @ local (4,) -> (B, 4)
        world = (T @ local_h).squeeze(-1) if T.dim() == 2 else (T @ local_h)
        return world[..., :3]

    def _z_axis(self, actor) -> torch.Tensor:
        """World-frame Z-axis of an actor (3rd column of its rotation matrix)."""
        T = actor.pose.to_transformation_matrix()
        return T[..., :3, 2]

    # ------------------------------------------------------------------ #
    # Observations / evaluation / reward
    # ------------------------------------------------------------------ #
    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_tcp_pose=self.agent.left_tcp.pose.raw_pose,
            right_tcp_pose=self.agent.right_tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                socket_pose=self.socket.pose.raw_pose,
                peg_pose=self.peg.pose.raw_pose,
                socket_entrance_pos=self._site_pos(self.socket, self.SOCKET_ENTRANCE_LOCAL),
                peg_end2_pos=self._site_pos(self.peg, self.PEG_END2_LOCAL),
            )
        return obs

    def evaluate(self):
        peg_end2 = self._site_pos(self.peg, self.PEG_END2_LOCAL)
        socket_rear = self._site_pos(self.socket, self.SOCKET_REAR_LOCAL)
        peg_insertion_dist = torch.linalg.norm(peg_end2 - socket_rear, axis=-1)
        peg_dist_to_axis = self._peg_dist_to_socket_axis(peg_end2)
        is_inserted = (peg_insertion_dist <= self.insertion_thresh) & (
            peg_dist_to_axis <= self.insertion_align_thresh
        )
        return {
            "success": is_inserted,
            "peg_insertion_dist": peg_insertion_dist,
            "peg_dist_to_axis": peg_dist_to_axis,
        }

    def _peg_dist_to_socket_axis(self, peg_end2: torch.Tensor) -> torch.Tensor:
        """Closest distance from peg_end2 to the socket's interior axis (rear->entrance)."""
        socket_entrance = self._site_pos(self.socket, self.SOCKET_ENTRANCE_LOCAL)
        socket_rear = self._site_pos(self.socket, self.SOCKET_REAR_LOCAL)
        ab = socket_entrance - socket_rear
        t = ((peg_end2 - socket_rear) * ab).sum(dim=-1)
        t = t / ((ab * ab).sum(dim=-1) + 1e-6)
        nearest = socket_rear + t.unsqueeze(-1) * ab
        return torch.linalg.norm(peg_end2 - nearest, axis=-1)

    @staticmethod
    def _tolerance(x: torch.Tensor, lo: float, hi: float, margin: float) -> torch.Tensor:
        """Linear-falloff tolerance kernel matching mujoco_playground's reward_util.tolerance."""
        in_band = (x >= lo) & (x <= hi)
        below = (lo - x).clamp(min=0.0)
        above = (x - hi).clamp(min=0.0)
        outside = below + above
        falloff = (1.0 - (outside / max(margin, 1e-6))).clamp(min=0.0, max=1.0)
        return torch.where(in_band, torch.ones_like(x), falloff)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Reproduce the per-component reward stack from the source.
        b = self.num_envs
        left_tcp = self.agent.left_tcp.pose.p
        right_tcp = self.agent.right_tcp.pose.p
        socket_pos = self.socket.pose.p
        peg_pos = self.peg.pose.p

        left_dist = torch.linalg.norm(socket_pos - left_tcp, axis=-1)
        right_dist = torch.linalg.norm(peg_pos - right_tcp, axis=-1)
        left_reward = self._tolerance(left_dist, 0.0, 0.001, margin=0.3)
        right_reward = self._tolerance(right_dist, 0.0, 0.001, margin=0.3)

        # Stay close to the home arm pose.
        qpos = self.agent.robot.get_qpos()
        home_qpos_t = torch.tensor(self._home_qpos(b), device=self.device)
        diff = qpos - home_qpos_t
        # Slice off finger joints (indices for arm joints only)
        arm_idx = [
            self.agent.robot.active_joints_map[n].active_index[0].item()
            for n in self.agent.left_arm_joint_names + self.agent.right_arm_joint_names
        ]
        arm_diff = diff[..., arm_idx]
        left_pose_err = torch.linalg.norm(arm_diff[..., :6], axis=-1)
        right_pose_err = torch.linalg.norm(arm_diff[..., 6:], axis=-1)
        left_pose_r = self._tolerance(left_pose_err, 0.0, 0.01, margin=2.0)
        right_pose_r = self._tolerance(right_pose_err, 0.0, 0.01, margin=2.0)

        socket_lift_dist = torch.linalg.norm(
            torch.tensor(self.socket_goal_pos, device=self.device) - socket_pos, axis=-1
        )
        peg_lift_dist = torch.linalg.norm(
            torch.tensor(self.peg_goal_pos, device=self.device) - peg_pos, axis=-1
        )
        socket_lift = self._tolerance(socket_lift_dist, 0.0, 0.01, margin=0.15)
        peg_lift = self._tolerance(peg_lift_dist, 0.0, 0.01, margin=0.15)

        z_world = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        socket_orient = (self._z_axis(self.socket) * z_world).sum(dim=-1)
        peg_orient = (self._z_axis(self.peg) * z_world).sum(dim=-1)
        socket_orient_r = self._tolerance(socket_orient, 0.99, 1.0, margin=0.03)
        peg_orient_r = self._tolerance(peg_orient, 0.99, 1.0, margin=0.03)

        # Hand-table collision: query the AlohaTableSceneBuilder's table.
        table_collision = self._finger_table_collision_flag()
        no_table_collision = 1.0 - table_collision

        # Insertion reward gated on alignment with the socket axis.
        peg_end2 = self._site_pos(self.peg, self.PEG_END2_LOCAL)
        socket_rear = self._site_pos(self.socket, self.SOCKET_REAR_LOCAL)
        insertion_dist = torch.linalg.norm(peg_end2 - socket_rear, axis=-1)
        gate = (self._peg_dist_to_socket_axis(peg_end2) < self.insertion_align_thresh).float()
        insertion_r = self._tolerance(insertion_dist, 0.0, 0.001, margin=0.1) * gate

        components = {
            "left_reward": left_reward,
            "right_reward": right_reward,
            "left_target_qpos": left_pose_r * left_reward * right_reward,
            "right_target_qpos": right_pose_r * left_reward * right_reward,
            "no_table_collision": no_table_collision,
            "socket_entrance_reward": socket_lift,
            "peg_end2_reward": peg_lift,
            "socket_z_up": socket_orient_r * socket_lift,
            "peg_z_up": peg_orient_r * peg_lift,
            "peg_insertion_reward": insertion_r,
        }
        reward = sum(_REWARD_SCALES[k] * v for k, v in components.items())
        reward = reward / sum(_REWARD_SCALES.values())
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        # Already in [0, 1] by construction (sum-of-scales normalization).
        return self.compute_dense_reward(obs, action, info)

    def _finger_table_collision_flag(self) -> torch.Tensor:
        """Return 1.0 if any finger link is in contact with the table, else 0.0."""
        table = self.table_scene.table
        flags = []
        for link in (
            self.agent.left_finger1_link,
            self.agent.left_finger2_link,
            self.agent.right_finger1_link,
            self.agent.right_finger2_link,
        ):
            forces = self.scene.get_pairwise_contact_forces(link, table)
            flags.append(torch.linalg.norm(forces, axis=-1) > 1e-3)
        any_contact = torch.stack(flags, dim=-1).any(dim=-1)
        return any_contact.float()
