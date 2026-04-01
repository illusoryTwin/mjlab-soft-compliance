"""Unitree G1 custom v2: phased joint-pos tracking schedule.

Phase 1 (0 → 120k steps): pure joint position tracking ramps up.
Phase 2 (120k → 240k steps): crossfade to compliant tracking.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g1_custom.env_cfgs import (
    unitree_g1_custom_env_cfg,
)
from mjlab.tasks.velocity.config.g1_custom.events import (
    apply_compliance_forces,
)
from mjlab.tasks.velocity.mdp.compliance_command import ComplianceCommandCfg
from mjlab.tasks.velocity.mdp.stiffness_command import StiffnessCommandCfg

# Bodies that receive external perturbations AND are monitored by compliance.
PERTURBED_BODIES = (
    # Left arm.
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    # # Right arm.
    # "right_shoulder_pitch_link",
    # "right_shoulder_roll_link",
    # "right_shoulder_yaw_link",
    # "right_elbow_link",
    # "right_wrist_roll_link",
    # "right_wrist_pitch_link",
    # "right_wrist_yaw_link",
)


def unitree_g1_custom_v2_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Unitree G1 with phased joint-pos tracking schedule."""
    cfg = unitree_g1_custom_env_cfg(play=play)

    cfg.rewards["compliant_joint_pos_tracking"].weight = 0.0

    # Variable stiffness command (resampled every 10 s).
    cfg.commands["stiffness"] = StiffnessCommandCfg(
        resampling_time_range=(10.0, 10.0),
        stiffness_range=(10.0, 10.0), # 30.0),
        initial_stiffness=10.0,
    )


    # ── Apply forces to one of the arm bodies ──
    cfg.events["compliance_push"] = EventTermCfg(
        func=apply_compliance_forces,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=PERTURBED_BODIES,
            ),
            "force_amplitude": 50.0,
            "min_hold_duration": 2.0,
            "max_hold_duration": 6.0,
            "on_duration": 3.0,
            "off_duration": 3.0,
            "random_bodies": True,
            "max_random_bodies": 1,
        },
    )

    cfg.curriculum["compliant_joint_pos_tracking_weight"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "compliant_joint_pos_tracking",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 60_000, "weight": 0.1},
                {"step": 90_000, "weight": 0.3},
                {"step": 120_000, "weight": 0.5},
                {"step": 180_000, "weight": 1.0},
            ],
        },
    )

    return cfg
