"""Unitree G1 compliance (MSD) parameters — tune coefficients here."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from mjlab.tasks.velocity.mdp.compliance.manager import ComplianceManagerCfg


# Those links are where external disturbances show up as body wrenches, 
# and where those wrenches are turned into joint torques via the Jacobian to drive the MSD
UNITREE_G1_DEFAULT_MONITORED_BODIES: List[str] = [
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]

# Per-joint stiffness scales
UNITREE_G1_STIFFNESS_SCALES: Dict[str, float] = {
  "waist_yaw_joint": 2.5,
  "left_shoulder_pitch_joint": 0.1,
  "right_shoulder_pitch_joint": 0.1,
  "left_shoulder_roll_joint": 0.1,
  "right_shoulder_roll_joint": 0.1,
  "left_shoulder_yaw_joint": 0.1,
  "right_shoulder_yaw_joint": 0.1,
  "left_elbow_joint": 0.075,
  "right_elbow_joint": 0.075,
  "left_wrist_roll_joint": 0.075,
  "right_wrist_roll_joint": 0.075,
  "left_wrist_pitch_joint": 0.075,
  "right_wrist_pitch_joint": 0.075,
  "left_wrist_yaw_joint": 0.075,
  "right_wrist_yaw_joint": 0.075,
}


def unitree_g1_compliance_manager_cfg(
  *,
  monitored_bodies: Sequence[str] | None = None,
  stiffness_config: Dict[str, float] | None = None,
  **kwargs: Any,
) -> ComplianceManagerCfg:
  """Instantiate `ComplianceManagerCfg` with G1 default parameters.

  For small updates from an existing cfg, prefer using `replace` method:

      cfg = unitree_g1_compliance_manager_cfg()
      cfg_softer = replace(cfg, base_stiffness=8.0)
  """
  return ComplianceManagerCfg(
    monitored_bodies=(
      list(monitored_bodies)
      if monitored_bodies is not None
      else list(UNITREE_G1_DEFAULT_MONITORED_BODIES)
    ),
    stiffness_config=(
      dict(stiffness_config)
      if stiffness_config is not None
      else dict(UNITREE_G1_STIFFNESS_SCALES)
    ),
    **kwargs,
  )
