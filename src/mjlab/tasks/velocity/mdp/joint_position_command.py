from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  import mujoco

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class JointPositionCommand(CommandTerm):
  cfg: JointPositionCommandCfg

  def __init__(self, cfg: JointPositionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.joint_ids, self.joint_names = self.robot.find_joints(
      cfg.joint_names, preserve_order=True
    )
    self.num_joints = len(self.joint_ids)

    # Command buffer: target joint positions
    self.joint_positions_b = torch.zeros(
      self.num_envs, self.num_joints, device=self.device
    )
    self.mid_positions = torch.zeros(self.num_envs, self.num_joints, device=self.device)
    for i, name in enumerate(self.joint_names):
      if name in cfg.ranges:
        joint_range = cfg.ranges[name]
        self.mid_positions[:, i] = (joint_range[0] + joint_range[1]) / 2.0
    self.joint_positions_b = self.mid_positions.clone()

    if cfg.smooth_sampling:
      # Random phase offsets for each environment and joint.
      self.phase_offsets = torch.rand(
        self.num_envs, self.num_joints, device=self.device
      )
      # Random frequency multipliers for each environment and joint.
      self.frequency_multiplier = torch.empty(
        self.num_envs, self.num_joints, device=self.device
      ).uniform_(
        cfg.sampling_frequency_range[0],
        cfg.sampling_frequency_range[1],
      )

      self.amplitudes = torch.zeros(self.num_envs, self.num_joints, device=self.device)
      for i, name in enumerate(self.joint_names):
        if name in cfg.ranges:
          joint_range = cfg.ranges[name]
          self.amplitudes[:, i] = (joint_range[1] - joint_range[0]) / 2.0

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    for name in self.joint_names:
      self.metrics[f"error_{name}"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost models created lazily on first visualization.
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._compliant_ghost_model: mujoco.MjModel | None = None
    self._compliant_ghost_color = np.array(
      cfg.viz.compliant_ghost_color, dtype=np.float32
    )

  @property
  def command(self) -> torch.Tensor:
    return self.joint_positions_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    joint_error = self.joint_positions_b - self.robot.data.joint_pos[:, self.joint_ids]
    self.metrics["error_joint_pos"] += (
      torch.norm(joint_error, dim=-1) / max_command_step
    )
    for i, name in enumerate(self.joint_names):
      self.metrics[f"error_{name}"] += torch.abs(joint_error[:, i]) / max_command_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if not self.cfg.smooth_sampling:
      # Discrete random sampling (joints without ranges keep mid-position).
      r = torch.empty(len(env_ids), device=self.device)
      for i, name in enumerate(self.joint_names):
        if name not in self.cfg.ranges:
          continue
        joint_range = self.cfg.ranges[name]
        self.joint_positions_b[env_ids, i] = r.uniform_(joint_range[0], joint_range[1])
    else:
      # Smooth sampling: re-randomize phase offsets and frequencies.
      r = torch.empty(len(env_ids), device=self.device)
      for i in range(self.num_joints):
        self.phase_offsets[env_ids, i] = torch.empty(
          len(env_ids), device=self.device
        ).uniform_(0.0, 1.0)
        self.frequency_multiplier[env_ids, i] = r.uniform_(
          self.cfg.sampling_frequency_range[0], self.cfg.sampling_frequency_range[1]
        )

  def _update_command(self) -> None:
    if self.cfg.smooth_sampling:
      max_time = self.cfg.resampling_time_range[1]
      elapsed = max_time - self.time_left
      smooth_variation = torch.cos(
        2.0 * math.pi * self.frequency_multiplier * elapsed.unsqueeze(-1)
        + self.phase_offsets
      )
      self.joint_positions_b = self.mid_positions + self.amplitudes * smooth_variation

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robots: pure command (blue) and compliant (red)."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self._ghost_model is None:
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      self._ghost_model.geom_rgba[:] = self._ghost_color

    has_compliance = self.cfg.compliance_command_name is not None
    if has_compliance and self._compliant_ghost_model is None:
      self._compliant_ghost_model = copy.deepcopy(self._env.sim.mj_model)
      self._compliant_ghost_model.geom_rgba[:] = self._compliant_ghost_color

    indexing = self.robot.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()
    cmd_joint_q_adr = joint_q_adr[self.joint_ids]

    for batch in env_indices:
      # Base qpos from current robot state.
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = (
        self.robot.data.body_link_pos_w[batch, 0].cpu().numpy()
      )
      qpos[free_joint_q_adr[3:7]] = (
        self.robot.data.body_link_quat_w[batch, 0].cpu().numpy()
      )
      qpos[joint_q_adr] = self.robot.data.joint_pos[batch].cpu().numpy()

      # Ghost 1: pure command (blue).
      pure_target = self.joint_positions_b[batch].cpu().numpy()
      qpos_cmd = qpos.copy()
      qpos_cmd[cmd_joint_q_adr] = pure_target
      visualizer.add_ghost_mesh(
        qpos_cmd, model=self._ghost_model, label=f"ghost_cmd_{batch}"
      )

      # Ghost 2: command + compliance deformations (red).
      if has_compliance:
        comp = self._env.command_manager.get_term(self.cfg.compliance_command_name)
        msd = comp._manager._msd_system
        defs = comp._manager.deformations[batch].cpu().numpy()
        compliant_target = pure_target.copy()
        for msd_col, entity_idx in enumerate(msd.active_idx):
          if entity_idx in self.joint_ids:
            cmd_col = self.joint_ids.index(entity_idx)
            compliant_target[cmd_col] += defs[msd_col]
        qpos_comp = qpos.copy()
        qpos_comp[cmd_joint_q_adr] = compliant_target
        visualizer.add_ghost_mesh(
          qpos_comp, model=self._compliant_ghost_model,
          label=f"ghost_comp_{batch}",
        )


@dataclass(kw_only=True)
class JointPositionCommandCfg(CommandTermCfg):
  entity_name: str
  joint_names: list[str]
  ranges: dict[str, tuple[float, float]]
  smooth_sampling: bool = False
  sampling_frequency_range: tuple[float, float] = (0.1, 0.5)
  compliance_command_name: str | None = None

  @dataclass
  class VizCfg:
    ghost_color: tuple[float, float, float, float] = (0.5, 0.5, 0.8, 0.4)
    compliant_ghost_color: tuple[float, float, float, float] = (0.8, 0.3, 0.3, 0.4)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> JointPositionCommand:
    return JointPositionCommand(self, env)
