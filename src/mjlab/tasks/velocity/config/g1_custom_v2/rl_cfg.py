"""RL configuration for Unitree G1 custom v2 task."""

from mjlab.tasks.velocity.config.g1_custom.rl_cfg import (
    unitree_g1_custom_ppo_runner_cfg,
)


def unitree_g1_custom_v2_ppo_runner_cfg():
    """Reuse g1_custom RL config with a separate experiment name."""
    cfg = unitree_g1_custom_ppo_runner_cfg()
    cfg.experiment_name = "g1_custom_v2"
    return cfg
