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
        self.stiffness = torch.full(
            (self.num_envs, 1),
            cfg.initial_stiffness,
            device=self.device,
        )

    @property
    def command(self) -> torch.Tensor:
        return self.stiffness

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.stiffness[env_ids, 0] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(*self.cfg.stiffness_range)

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class StiffnessCommandCfg(CommandTermCfg):
    """Configuration for the variable stiffness command."""

    stiffness_range: tuple[float, float] = (5.0, 20.0)
    """Range ``[min, max]`` for uniform stiffness sampling."""

    initial_stiffness: float = 10.0
    """Stiffness value used before the first resample."""

    def build(self, env: ManagerBasedRlEnv) -> StiffnessCommand:
        return StiffnessCommand(self, env)
