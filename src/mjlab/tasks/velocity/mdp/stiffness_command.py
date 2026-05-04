"""Command term that generates variable stiffness for the compliance MSD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class StiffnessCommand(CommandTerm):

    cfg: StiffnessCommandCfg

    def __init__(self, cfg: StiffnessCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._discrete = len(cfg.stiffness_multipliers) > 0
        if self._discrete:
            self._levels = torch.tensor(
                [cfg.base_stiffness * float(m) for m in cfg.stiffness_multipliers],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self._levels = None
        self.stiffness = torch.full(
            (self.num_envs, 1),
            float(cfg.initial_stiffness),
            device=self.device,
            dtype=torch.float32,
        )

    @property
    def command(self) -> torch.Tensor:
        return self.stiffness

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.shape[0])
        if n == 0:
            return
        if self._discrete:
            k = int(self._levels.shape[0])
            idx = torch.randint(0, k, (n,), device=self.device)
            self.stiffness[env_ids, 0] = self._levels[idx]
        else:
            self.stiffness[env_ids, 0] = torch.empty(
                n, device=self.device, dtype=torch.float32
            ).uniform_(float(self.cfg.stiffness_range[0]), float(self.cfg.stiffness_range[1]))

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class StiffnessCommandCfg(CommandTermCfg):
    """Configuration for the variable stiffness command."""

    base_stiffness: float = 1.0
    """Scales ``stiffness_multipliers`` when discrete sampling is enabled."""

    stiffness_multipliers: tuple[float, ...] = ()
    """When non-empty, each resample picks a multiplier ``m`` uniformly and sets
    stiffness to ``base_stiffness * m``. When empty, ``stiffness_range`` is used instead."""

    stiffness_range: tuple[float, float] = (10.0, 60.0)
    """Used only when ``stiffness_multipliers`` is empty: uniform ``[min, max]``."""

    initial_stiffness: float = 10.0
    """Stiffness value used in :meth:`StiffnessCommand.__init__` before the first resample."""

    def build(self, env: ManagerBasedRlEnv) -> StiffnessCommand:
        return StiffnessCommand(self, env)
