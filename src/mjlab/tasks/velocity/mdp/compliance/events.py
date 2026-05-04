"""Event callables for compliance-related external wrenches (forces/torques)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class apply_constant_torque:
  """Apply a constant torque to specified bodies every step.

  The torque vector is specified in the world frame and written to
  ``xfrc_applied`` each step. Useful for testing joint-level disturbance
  rejection (e.g. a constant 30 Nm load on a shoulder pitch).

  Use with ``mode="step"``.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )

    torque = cfg.params["torque"]
    self._torque = (
      torch.tensor(torque, device=self._device, dtype=torch.float32)
      .unsqueeze(0)
      .unsqueeze(0)
      .expand(self._num_envs, self._num_bodies, 3)
      .clone()
    )
    self._zero_forces = torch.zeros_like(self._torque)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
  ) -> None:
    del env, env_ids, asset_cfg, torque
    self._asset.write_external_wrench_to_sim(
      self._zero_forces, self._torque, body_ids=self._body_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    pass


class apply_compliance_torques:
  """Apply piecewise-constant random torques on compliance-monitored bodies.

  Each environment/body/axis gets a random torque in
  ``[-amplitude, +amplitude]`` held for a random duration sampled from
  ``[min_hold_duration, max_hold_duration]``.  When the hold timer
  expires the value is resampled.

  The duty-cycle (on/off) mask is applied on top: during the "off"
  phase torques are zeroed.  When both ``on_duration`` and
  ``off_duration`` are 0, torques are applied continuously.

  Use with ``mode="step"``.
  """

  @dataclass
  class VizCfg:
    """Arrow visualization settings for compliance torques."""

    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 0.9)
    """Arrow color (RGBA)."""
    scale: float = 0.005
    """Arrow length in meters per Nm of torque."""
    width: float = 0.015
    """Arrow shaft width in meters."""
    min_torque: float = 1.0
    """Minimum torque magnitude (Nm) below which arrows are hidden."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )
    self._viz_cfg: apply_compliance_torques.VizCfg = cfg.params.get(
      "viz_cfg", apply_compliance_torques.VizCfg()
    )
    self._random_bodies: bool = cfg.params.get("random_bodies", False)
    self._min_random_bodies: int = cfg.params.get(
      "min_random_bodies", 1
    )
    self._max_random_bodies: int = cfg.params.get(
      "max_random_bodies", self._num_bodies
    )
    self._min_hold: float = cfg.params.get("min_hold_duration", 2.0)
    self._max_hold: float = cfg.params.get("max_hold_duration", 6.0)

    self._current_directions = torch.zeros(
      (self._num_envs, self._num_bodies, 3), device=self._device
    )
    self._hold_timers = torch.zeros(
      self._num_envs, device=self._device
    )
    self._resample_values(slice(None), self._num_envs)

    self._body_mask = torch.ones(
      (self._num_envs, self._num_bodies, 1), device=self._device
    )
    if self._random_bodies:
      self._randomize_body_mask(slice(None), self._num_envs)

    self._duty_offsets = torch.zeros(self._num_envs, device=self._device)

    self._step_count = 0

  def _resample_values(
    self, env_ids: torch.Tensor | slice, n: int
  ) -> None:
    """Sample new random constant directions and hold durations."""
    self._current_directions[env_ids] = (
      torch.rand((n, self._num_bodies, 3), device=self._device) * 2 - 1
    )
    self._hold_timers[env_ids] = (
      torch.rand(n, device=self._device)
      * (self._max_hold - self._min_hold)
      + self._min_hold
    )

  def _randomize_body_mask(
    self, env_ids: torch.Tensor | slice, n: int
  ) -> None:
    """Pick a random subset of bodies for each environment."""
    counts = torch.randint(
      self._min_random_bodies,
      self._max_random_bodies + 1,
      (n,),
      device=self._device,
    )
    mask = torch.zeros(
      (n, self._num_bodies), device=self._device
    )
    for i in range(n):
      perm = torch.randperm(self._num_bodies, device=self._device)
      mask[i, perm[: counts[i]]] = 1.0
    self._body_mask[env_ids] = mask.unsqueeze(-1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    torque_amplitude: float = 50.0,
    min_hold_duration: float = 2.0,
    max_hold_duration: float = 6.0,
    on_duration: float = 0.0,
    off_duration: float = 0.0,
    random_bodies: bool = False,
    min_random_bodies: int = 1,
    max_random_bodies: int = 0,
  ) -> None:
    del env, env_ids, asset_cfg
    del random_bodies, min_random_bodies, max_random_bodies
    del min_hold_duration, max_hold_duration

    t = self._step_count * self._step_dt
    self._step_count += 1

    self._hold_timers -= self._step_dt
    expired = self._hold_timers <= 0
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      self._resample_values(expired_ids, len(expired_ids))
      if self._random_bodies:
        self._randomize_body_mask(expired_ids, len(expired_ids))

    torques = torque_amplitude * self._current_directions

    use_duty_cycle = on_duration > 0 and off_duration > 0
    if use_duty_cycle:
      cycle_period = on_duration + off_duration
      t_in_cycle = (t + self._duty_offsets) % cycle_period
      active_mask = (t_in_cycle < on_duration).float()
      torques = torques * active_mask[:, None, None]

    torques = torques * self._body_mask

    forces = torch.zeros_like(torques)

    self._asset.write_external_wrench_to_sim(
      forces, torques, body_ids=self._body_ids
    )

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw arrows for active compliance torques."""
    viz = self._viz_cfg
    min_sq = viz.min_torque * viz.min_torque
    wrench = self._asset.data.body_external_wrench
    com_pos = self._asset.data.body_com_pos_w
    for env_idx in visualizer.get_env_indices(self._num_envs):
      body_ids = (
        self._body_ids
        if isinstance(self._body_ids, list)
        else range(wrench.shape[1])
      )
      for i in body_ids:
        torque = wrench[env_idx, i, 3:]
        if (torque * torque).sum().item() < min_sq:
          continue
        torque_np = torque.cpu().numpy()
        start_np = com_pos[env_idx, i].cpu().numpy()
        end_np = start_np + torque_np * viz.scale
        visualizer.add_arrow(
          start=start_np,
          end=end_np,
          color=viz.rgba,
          width=viz.width,
        )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    if isinstance(env_ids, slice):
      n = self._num_envs
    else:
      n = len(env_ids)

    self._resample_values(env_ids, n)
    self._duty_offsets[env_ids] = torch.rand(n, device=self._device)

    if self._random_bodies:
      self._randomize_body_mask(env_ids, n)

    zeros = torch.zeros((n, self._num_bodies, 3), device=self._device)
    self._asset.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
    )


class apply_compliance_forces:
  """Apply piecewise-constant random forces on compliance-monitored bodies.

  Each environment/body/axis gets a random force in
  ``[-amplitude, +amplitude]`` held for a random duration sampled from
  ``[min_hold_duration, max_hold_duration]``.  When the hold timer
  expires the value is resampled.

  The duty-cycle (on/off) mask is applied on top: during the "off"
  phase forces are zeroed.  When both ``on_duration`` and
  ``off_duration`` are 0, forces are applied continuously.

  Use with ``mode="step"``.
  """

  @dataclass
  class VizCfg:
    """Arrow visualization settings for compliance forces."""

    rgba: tuple[float, float, float, float] = (0.2, 0.8, 0.2, 0.9)
    """Arrow color (RGBA)."""
    scale: float = 0.005
    """Arrow length in meters per Newton of force."""
    width: float = 0.015
    """Arrow shaft width in meters."""
    min_force: float = 1.0
    """Minimum force magnitude (N) below which arrows are hidden."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )
    self._viz_cfg: apply_compliance_forces.VizCfg = cfg.params.get(
      "viz_cfg", apply_compliance_forces.VizCfg()
    )
    self._random_bodies: bool = cfg.params.get("random_bodies", False)
    self._min_random_bodies: int = cfg.params.get(
      "min_random_bodies", 1
    )
    self._max_random_bodies: int = cfg.params.get(
      "max_random_bodies", self._num_bodies
    )
    self._min_hold: float = cfg.params.get("min_hold_duration", 2.0)
    self._max_hold: float = cfg.params.get("max_hold_duration", 6.0)

    self._current_directions = torch.zeros(
      (self._num_envs, self._num_bodies, 3), device=self._device
    )
    self._hold_timers = torch.zeros(
      self._num_envs, device=self._device
    )
    self._resample_values(slice(None), self._num_envs)

    self._body_mask = torch.ones(
      (self._num_envs, self._num_bodies, 1), device=self._device
    )
    if self._random_bodies:
      self._randomize_body_mask(slice(None), self._num_envs)

    self._duty_offsets = torch.zeros(self._num_envs, device=self._device)

    self._step_count = 0

  def _resample_values(
    self, env_ids: torch.Tensor | slice, n: int
  ) -> None:
    """Sample new random constant directions and hold durations."""
    self._current_directions[env_ids] = (
      torch.rand((n, self._num_bodies, 3), device=self._device) * 2 - 1
    )
    self._hold_timers[env_ids] = (
      torch.rand(n, device=self._device)
      * (self._max_hold - self._min_hold)
      + self._min_hold
    )

  def _randomize_body_mask(
    self, env_ids: torch.Tensor | slice, n: int
  ) -> None:
    """Pick a random subset of bodies for each environment."""
    counts = torch.randint(
      self._min_random_bodies,
      self._max_random_bodies + 1,
      (n,),
      device=self._device,
    )
    mask = torch.zeros(
      (n, self._num_bodies), device=self._device
    )
    for i in range(n):
      perm = torch.randperm(self._num_bodies, device=self._device)
      mask[i, perm[: counts[i]]] = 1.0
    self._body_mask[env_ids] = mask.unsqueeze(-1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    force_amplitude: float = 50.0,
    min_hold_duration: float = 2.0,
    max_hold_duration: float = 6.0,
    on_duration: float = 0.0,
    off_duration: float = 0.0,
    random_bodies: bool = False,
    min_random_bodies: int = 1,
    max_random_bodies: int = 0,
  ) -> None:
    del env, env_ids, asset_cfg
    del random_bodies, min_random_bodies, max_random_bodies
    del min_hold_duration, max_hold_duration

    t = self._step_count * self._step_dt
    self._step_count += 1

    self._hold_timers -= self._step_dt
    expired = self._hold_timers <= 0
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      self._resample_values(expired_ids, len(expired_ids))
      if self._random_bodies:
        self._randomize_body_mask(expired_ids, len(expired_ids))

    forces = force_amplitude * self._current_directions

    use_duty_cycle = on_duration > 0 and off_duration > 0
    if use_duty_cycle:
      cycle_period = on_duration + off_duration
      t_in_cycle = (t + self._duty_offsets) % cycle_period
      active_mask = (t_in_cycle < on_duration).float()
      forces = forces * active_mask[:, None, None]

    forces = forces * self._body_mask

    torques = torch.zeros_like(forces)

    self._asset.write_external_wrench_to_sim(
      forces, torques, body_ids=self._body_ids
    )

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw arrows for active compliance forces."""
    viz = self._viz_cfg
    min_sq = viz.min_force * viz.min_force
    wrench = self._asset.data.body_external_wrench
    com_pos = self._asset.data.body_com_pos_w
    for env_idx in visualizer.get_env_indices(self._num_envs):
      body_ids = (
        self._body_ids
        if isinstance(self._body_ids, list)
        else range(wrench.shape[1])
      )
      for i in body_ids:
        force = wrench[env_idx, i, :3]
        if (force * force).sum().item() < min_sq:
          continue
        force_np = force.cpu().numpy()
        start_np = com_pos[env_idx, i].cpu().numpy()
        end_np = start_np + force_np * viz.scale
        visualizer.add_arrow(
          start=start_np,
          end=end_np,
          color=viz.rgba,
          width=viz.width,
        )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    if isinstance(env_ids, slice):
      n = self._num_envs
    else:
      n = len(env_ids)

    self._resample_values(env_ids, n)
    self._duty_offsets[env_ids] = torch.rand(n, device=self._device)

    if self._random_bodies:
      self._randomize_body_mask(env_ids, n)

    zeros = torch.zeros((n, self._num_bodies, 3), device=self._device)
    self._asset.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
    )
