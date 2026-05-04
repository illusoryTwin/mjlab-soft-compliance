"""Command term that computes compliance deformations from external forces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.compliance.manager import (
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

    active_idx = self.active_joint_indices()
    self._active_joint_names = [entity.joint_names[i] for i in active_idx]

  def deformations(self) -> torch.Tensor:
    return self._manager.deformations

  def deformation_velocities(self) -> torch.Tensor:
    return self._manager.deformation_velocities

  def active_joint_indices(self) -> tuple[int, ...]:
    return self._manager.active_joint_entity_indices

  def active_joint_indices_torch(self) -> torch.Tensor:
    return self._manager.active_joint_entity_indices_torch

  def joint_torques_from_disturbance(self) -> torch.Tensor:
    return self._manager.joint_torques_from_disturbance()

  @property
  def command(self) -> torch.Tensor:
    return self.deformations()

  def _update_metrics(self) -> None:
    msd = self._manager._msd_system
    q_def = self.deformations()[0]
    qd_def = self.deformation_velocities()[0]
    ext_tau = self.joint_torques_from_disturbance()[0]
    active_idx = self.active_joint_indices_torch()

    log = self._env.extras.get("log")
    if log is None:
      return
    for i, name in enumerate(self._active_joint_names):
      log[f"Compliance/deformation/{name}"] = q_def[i]
      log[f"Compliance/deformation_vel/{name}"] = qd_def[i]
      log[f"Compliance/ext_torque/{name}"] = ext_tau[active_idx[i]]
      log[f"Compliance/msd/M/{name}"] = msd.M_active[i]
      log[f"Compliance/msd/K/{name}"] = msd.K[0, i]
      log[f"Compliance/msd/D/{name}"] = msd.D[0, i]
      log[f"Compliance/msd/q/{name}"] = q_def[i]
      log[f"Compliance/msd/qd/{name}"] = qd_def[i]
      log[f"Compliance/msd/qdd/{name}"] = msd._last_qdd[0, i]
      log[f"Compliance/msd/tau/{name}"] = msd._last_tau_active[0, i]

    entity = self._manager._entity
    body_force = entity.data.body_external_force[0]
    body_torque = entity.data.body_external_torque[0]
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

    for j, jname in enumerate(entity.joint_names):
      log[f"Compliance/joint_torque/{jname}"] = ext_tau[j]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._manager.reset(env_ids)

  def _update_command(self) -> None:
    if self.cfg.stiffness_command_name is not None:
      stiffness = self._env.command_manager.get_command(
        self.cfg.stiffness_command_name
      )
      self._manager.set_msd_base_stiffness(stiffness[:, 0])
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
