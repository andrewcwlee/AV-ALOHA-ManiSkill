import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class Aloha(BaseAgent):
    """ALOHA bimanual robot (two Interbotix vx300s arms) ported from mujoco_playground.

    The MJCF wraps both arms under a single ``aloha_root`` body so SAPIEN parses
    one articulation. The base poses inside the MJCF place the left arm at
    ``(-0.469, -0.019, 0.02)`` and the right arm at ``(+0.469, -0.019, 0.02)``
    with the right arm rotated 180 degrees about Z, mirroring the real ALOHA rig.

    Finger coupling: MuJoCo ``<equality>`` joint constraints are not preserved by
    the SAPIEN MJCF loader, so the right finger of each gripper mimics the left
    finger via :class:`PDJointPosMimicControllerConfig`.
    """

    uid = "aloha"
    mjcf_path = f"{PACKAGE_ASSET_DIR}/robots/aloha/aloha.xml"
    urdf_config = dict()
    fix_root_link = True
    # Source MJCF uses <contact><exclude>; SAPIEN's MJCF loader drops those.
    # Disable self-collisions wholesale to avoid spurious base<->shoulder contacts.
    disable_self_collisions = True

    left_arm_joint_names = [
        "left/waist",
        "left/shoulder",
        "left/elbow",
        "left/forearm_roll",
        "left/wrist_angle",
        "left/wrist_rotate",
    ]
    right_arm_joint_names = [n.replace("left/", "right/") for n in left_arm_joint_names]
    # Order: control joints first, mimic joints last (matches mimic dict below).
    gripper_joint_names = [
        "left/left_finger",
        "right/left_finger",
        "left/right_finger",
        "right/right_finger",
    ]

    # Per-joint dynamics transcribed from xmls/mjx_aloha.xml <default> classes.
    # Order matches *_arm_joint_names: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate.
    arm_stiffness = np.array([43.0, 265.0, 227.0, 78.0, 37.0, 10.4])
    arm_damping = np.array([5.76, 20.0, 18.49, 6.78, 6.28, 1.2])
    arm_force_limit = np.array([35.0, 144.0, 59.0, 22.0, 35.0, 35.0])

    gripper_stiffness = 365.0
    gripper_damping = 40.0
    gripper_force_limit = 35.0

    # 16 DOF: 6 left arm + 6 right arm + 2 left fingers + 2 right fingers.
    # Active joint order in SAPIEN follows the body traversal order in the MJCF,
    # which interleaves arm joints with fingers; we set per-joint qpos via the
    # joints_map in the scene builder.
    keyframes = dict(
        rest=Keyframe(
            qpos=np.zeros(16),
            pose=sapien.Pose(),
        )
    )

    @property
    def _sensor_configs(self):
        # All four ALOHA cameras (overhead, worms-eye, left wrist, right wrist)
        # use Intel D405 intrinsics: focal=1.93mm, sensor=3.896x2.140mm.
        # Vertical FOV = 2*atan(2.140 / (2*1.93)) ~= 1.0123 rad.
        # MuJoCo cameras look down -Z (local) with +Y up; SAPIEN cameras look
        # down +X (local) with +Z up. We sidestep this convention conversion
        # by precomputing equivalent (eye, target, up) triples and using
        # sapien_utils.look_at - which works in any frame, including a link's
        # local frame for mounted cameras.
        d405_fov_y = 1.0123

        # ---- Workstation cameras (world-fixed, from xmls/mjx_scene.xml) ----
        # overhead_cam: pos=(0, -0.303794, 1.02524), quat about X by 25 deg.
        #   forward = R_x(0.4366) * (0,0,-1) = (0, 0.4226, -0.9063)
        overhead_pose = sapien_utils.look_at(
            eye=[0.0, -0.303794, 1.02524],
            target=[0.0, 0.118806, 0.119240],
        )
        # worms_eye_cam: pos=(0, -0.377167, 0.0316055), quat about X by 95.5 deg.
        #   forward = R_x(1.6664) * (0,0,-1) = (0, 0.9954, 0.0959)
        worms_eye_pose = sapien_utils.look_at(
            eye=[0.0, -0.377167, 0.0316055],
            target=[0.0, 0.618233, 0.127506],
        )

        # ---- Wrist cameras (mounted to gripper_base link of each arm) ----
        # Source MJCF: pos=(0, -0.0824748, -0.0095955), euler=(2.70525955359, 0, 0)
        # in the gripper_base body frame. The euler is a single rotation about
        # X by ~155 deg. Computing the resulting MuJoCo camera axes in the
        # link's local frame:
        #   look (-Z_mj after rotation) = (0, sin155, -(-cos155)) = (0, 0.4226, 0.9063)
        #   up   (+Y_mj after rotation) = (0, cos155,    sin155 ) = (0, -0.9063, 0.4226)
        # eye + look*1.0 gives a target 1m forward of the camera; SAPIEN's
        # look_at handles the camera-frame conversion.
        wrist_eye = [0.0, -0.0824748, -0.0095955]
        wrist_target = [0.0, -0.0824748 + 0.4226, -0.0095955 + 0.9063]
        wrist_up = (0.0, -0.9063, 0.4226)
        wrist_local_pose = sapien_utils.look_at(
            eye=wrist_eye, target=wrist_target, up=wrist_up
        )

        return [
            CameraConfig(
                "overhead_cam", overhead_pose, 128, 128, d405_fov_y, 0.01, 100,
            ),
            CameraConfig(
                "worms_eye_cam", worms_eye_pose, 128, 128, d405_fov_y, 0.01, 100,
            ),
            CameraConfig(
                "wrist_cam_left", wrist_local_pose, 128, 128, d405_fov_y, 0.01, 100,
                mount=self.robot.links_map["left/gripper_base"],
            ),
            CameraConfig(
                "wrist_cam_right", wrist_local_pose, 128, 128, d405_fov_y, 0.01, 100,
                mount=self.robot.links_map["right/gripper_base"],
            ),
        ]

    @property
    def _controller_configs(self):
        left_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        right_arm_pd_joint_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        left_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.left_arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        right_arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.right_arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        # Two control joints (left/left_finger, right/left_finger) drive the two
        # mimic joints (left/right_finger, right/right_finger). lower goes
        # negative (below the physical 0.002 limit) so the PD controller has
        # extra "squeeze force" when contact stops the joint from reaching
        # target -- matches the trick used in Panda.
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={
                "left/right_finger": {"joint": "left/left_finger"},
                "right/right_finger": {"joint": "right/left_finger"},
            },
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                left_arm=left_arm_pd_joint_pos,
                right_arm=right_arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                left_arm=left_arm_pd_joint_delta_pos,
                right_arm=right_arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
        )
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.left_tcp = self.robot.links_map["left/tcp"]
        self.right_tcp = self.robot.links_map["right/tcp"]
        # Tabletop tasks like PickCube assume agent.tcp; default to the right arm.
        self.tcp = self.right_tcp

        self.left_finger1_link = self.robot.links_map["left/left_finger_link"]
        self.left_finger2_link = self.robot.links_map["left/right_finger_link"]
        self.right_finger1_link = self.robot.links_map["right/left_finger_link"]
        self.right_finger2_link = self.robot.links_map["right/right_finger_link"]

    def is_static(self, threshold: float = 0.2):
        # Check only the 12 arm joints; finger joints are excluded since gripper
        # closure motion shouldn't count as "moving" for tabletop success criteria.
        qvel = self.robot.get_qvel()
        arm_joint_indices = [
            self.robot.active_joints_map[n].active_index[0].item()
            for n in self.left_arm_joint_names + self.right_arm_joint_names
        ]
        arm_qvel = qvel[..., arm_joint_indices]
        return torch.max(torch.abs(arm_qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    def is_grasping(self, object: Actor, arm: str = "right", min_force=0.5, max_angle=85):
        """Per-arm grasp check ported from WidowX250S.is_grasping."""
        if arm == "left":
            f1, f2 = self.left_finger1_link, self.left_finger2_link
        elif arm == "right":
            f1, f2 = self.right_finger1_link, self.right_finger2_link
        else:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")

        l_contact_forces = self.scene.get_pairwise_contact_forces(f1, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(f2, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # Open direction is +y in the finger link frame.
        ldirection = f1.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -f2.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)
