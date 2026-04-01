"""Custom Unitree G1 velocity environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1_custom.events import (
  apply_compliance_forces,
  apply_compliance_torques,
  apply_constant_torque,
)
from mjlab.tasks.velocity.mdp import JointPositionCommandCfg
from mjlab.tasks.velocity.mdp.compliance_command import ComplianceCommandCfg
from mjlab.tasks.velocity.mdp.compliance_manager import ComplianceManagerCfg
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


def unitree_g1_custom_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree G1 with compliance push event."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  #standing only
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  # Move base_lin_vel to critic-only
  del cfg.observations["actor"].terms["base_lin_vel"]

  # New observations
  cfg.observations["actor"].terms["joint_pos_command"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "joint_pos"},
  )
  cfg.observations["critic"].terms["compliance_deformations"] = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "compliance"},
  )

  # New commands
  cfg.commands["joint_pos"] = JointPositionCommandCfg(
    resampling_time_range=(10.0, 10.0),
    entity_name="robot",
    joint_names=[
      # Left arm.
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      # Right arm.
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ],
    ranges={
      # # ========
      # # Left arm.
      "left_shoulder_pitch_joint": (-0.5, 0.5),
      "left_shoulder_roll_joint": (-0.0, 0.7),
      "left_shoulder_yaw_joint": (-0.0, 0.0),
      "left_elbow_joint": (-0.3, 0.7),
      "left_wrist_roll_joint": (-0.3, 0.3),
      "left_wrist_pitch_joint": (-0.3, 0.3),
      "left_wrist_yaw_joint": (-0.3, 0.3),
      # # Right arm.
      "right_shoulder_pitch_joint": (-0.5, 0.5),
      "right_shoulder_roll_joint": (-0.7, 0.0),
      "right_shoulder_yaw_joint": (-0.0, 0.0),
      "right_elbow_joint": (-0.3, 0.7),
      "right_wrist_roll_joint": (-0.3, 0.3),
      "right_wrist_pitch_joint": (-0.3, 0.3),
      "right_wrist_yaw_joint": (-0.3, 0.3),
    },
    smooth_sampling=False,
    sampling_frequency_range=(0.0, 0.4),
    debug_vis=True,
    compliance_command_name="compliance", # only for visualization 
  )

  # # New rewards
  # cfg.rewards["compliant_joint_pos_tracking"] = RewardTermCfg(
  #   func=mdp.track_compliant_joint_position_command_l1,
  #   weight=0.1, # 1.0,
  #   params={
  #     "command_name": "joint_pos",
  #     "compliance_command_name": "compliance",
  #     "asset_cfg": SceneEntityCfg(
  #       "robot",
  #       joint_names=(
  #         # Left arm.
  #         "left_shoulder_pitch_joint",
  #         "left_shoulder_roll_joint",
  #         "left_shoulder_yaw_joint",
  #         "left_elbow_joint",
  #         "left_wrist_roll_joint",
  #         "left_wrist_pitch_joint",
  #         "left_wrist_yaw_joint",
  #         # Right arm.
  #         "right_shoulder_pitch_joint",
  #         "right_shoulder_roll_joint",
  #         "right_shoulder_yaw_joint",
  #         "right_elbow_joint",
  #         "right_wrist_roll_joint",
  #         "right_wrist_pitch_joint",
  #         "right_wrist_yaw_joint",
  #       ),
  #     ),
  #   },
  # )

  cfg.rewards["joint_pos_tracking"] = RewardTermCfg(
    func=mdp.track_joint_position_command_l1,
    weight=0.1, # 1.0,
    params={
      "command_name": "joint_pos",
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          # Left arm.
          "left_shoulder_pitch_joint",
          "left_shoulder_roll_joint",
          "left_shoulder_yaw_joint",
          "left_elbow_joint",
          "left_wrist_roll_joint",
          "left_wrist_pitch_joint",
          "left_wrist_yaw_joint",
          # Right arm.
          "right_shoulder_pitch_joint",
          "right_shoulder_roll_joint",
          "right_shoulder_yaw_joint",
          "right_elbow_joint",
          "right_wrist_roll_joint",
          "right_wrist_pitch_joint",
          "right_wrist_yaw_joint",
        ),
      ),
    },
  )

  # Variable stiffness command (resampled every 10 s).
  cfg.commands["stiffness"] = StiffnessCommandCfg(
    resampling_time_range=(10.0, 10.0),
    stiffness_range=(30.0, 30.0),
    initial_stiffness=10.0,
  )

  # # Compliance command (MSD deformations from external forces).
  # cfg.commands["compliance"] = ComplianceCommandCfg(
  #   resampling_time_range=(1e9, 1e9),  # never resample
  #   stiffness_command_name="stiffness",
  #   compliance=ComplianceManagerCfg(
  #     monitored_bodies=list(PERTURBED_BODIES),
  #   ),
  # )

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

  # cfg.curriculum["compliant_joint_pos_tracking_weight"] = CurriculumTermCfg(
  #     func=mdp.reward_weight,
  #     params={
  #         "reward_name": "compliant_joint_pos_tracking",
  #         "weight_stages": [
  #             {"step": 0, "weight": 0.0},
  #             {"step": 60_000, "weight": 0.1},
  #             {"step": 90_000, "weight": 0.3},
  #             {"step": 120_000, "weight": 0.5},
  #             {"step": 180_000, "weight": 1.0},
  #         ],
  #     },
  # )
    
  cfg.curriculum["joint_pos_tracking_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
        "reward_name": "joint_pos_tracking",
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
