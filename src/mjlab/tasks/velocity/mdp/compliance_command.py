"""Command term that computes compliance deformations from external forces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.compliance_manager import (
  ComplianceManager,
  ComplianceManagerCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ComplianceCommand(CommandTerm):

  cfg: ComplianceCommandCfg

  def __init__(self, cfg: ComplianceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    entity: Entity = env.scene[cfg.entity_name]
    self._manager = ComplianceManager(cfg.compliance, entity, env)

    # Joint names for the active MSD DOFs (used in metric keys).
    active_idx = self._manager._msd_system.active_idx
    self._active_joint_names = [entity.joint_names[i] for i in active_idx]

  @property
  def command(self) -> torch.Tensor:
    return self._manager.deformations

  def _update_metrics(self) -> None:
    msd = self._manager._msd_system
    q_def = msd.state["q_def"][0]  # env 0, (n_active,)
    qd_def = msd.state["qd_def"][0]
    ext_tau = self._manager.last_joint_torques[0]  # (num_joints,)
    active_idx = msd.active_idx_torch

    log = self._env.extras.get("log")
    if log is None:
      return
    for i, name in enumerate(self._active_joint_names):
      log[f"Compliance/deformation/{name}"] = q_def[i]
      log[f"Compliance/deformation_vel/{name}"] = qd_def[i]
      log[f"Compliance/ext_torque/{name}"] = ext_tau[active_idx[i]]

    # Log external body forces on monitored bodies for env 0.
    entity = self._manager._entity
    body_force = entity.data.body_external_force[0]  # (num_bodies, 3)
    body_torque = entity.data.body_external_torque[0]  # (num_bodies, 3)
    for body_id, body_name in zip(
      self._manager._body_ids, self._manager._body_names, strict=False
    ):
      f = body_force[body_id]
      log[f"Forces/{body_name}/fx"] = f[0]
      log[f"Forces/{body_name}/fy"] = f[1]
      log[f"Forces/{body_name}/fz"] = f[2]
      log[f"Forces/{body_name}/norm"] = torch.linalg.norm(f)
      t = body_torque[body_id]
      log[f"Forces/{body_name}/tx"] = t[0]
      log[f"Forces/{body_name}/ty"] = t[1]
      log[f"Forces/{body_name}/tz"] = t[2]

    # Log computed compliance joint torques for all joints for env 0.
    for j, jname in enumerate(entity.joint_names):
      log[f"Compliance/joint_torque/{jname}"] = ext_tau[j]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._manager.reset(env_ids)

  def _update_command(self) -> None:
    if self.cfg.stiffness_command_name is not None:
      stiffness = self._env.command_manager.get_command(
        self.cfg.stiffness_command_name
      )
      self._manager._msd_system.set_stiffness(stiffness[:, 0])
    self._manager.compute()

@dataclass(kw_only=True)
class ComplianceCommandCfg(CommandTermCfg):
  """Configuration for the compliance command."""

  entity_name: str = "robot"
  compliance: ComplianceManagerCfg = field(default_factory=ComplianceManagerCfg)
  stiffness_command_name: str | None = None
  """Name of the stiffness command to read variable stiffness from.
  When set, the MSD stiffness is updated each step from that command."""

  def build(self, env: ManagerBasedRlEnv) -> ComplianceCommand:
    return ComplianceCommand(self, env)
