"""Export a trained policy for deployment on real hardware.

Produces a deployment package with:
  - policy.pt: JIT-scripted actor + normalizer
  - config.yaml: observation layout, action scales, PD gains, joint config
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportConfig:
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  output: str = "./exported"
  device: str = "cpu"


def generate_deploy_config(
  env: ManagerBasedRlEnv, resume_path: Path, output_dir: Path
) -> None:
  """Generate config.yaml from training environment metadata."""
  metadata = get_base_metadata(env, run_path=str(resume_path))
  obs_manager = env.observation_manager
  actor_terms = obs_manager.active_terms["actor"]

  # Get per-term observation dimensions from the manager.
  term_shapes = obs_manager.group_obs_term_dim["actor"]
  obs_dims = {}
  for name, shape in zip(actor_terms, term_shapes):
    obs_dims[name] = int(shape[0]) if len(shape) == 1 else list(shape)

  deploy_config = {
    "num_joints": len(metadata["joint_names"]),
    "joint_names": metadata["joint_names"],
    "control": {
      "control_dt": env.step_dt,
      "action_scale": metadata["action_scale"],
      "stiffness": metadata["joint_stiffness"],
      "damping": metadata["joint_damping"],
    },
    "init_state": {
      "default_joint_pos": dict(
        zip(metadata["joint_names"], metadata["default_joint_pos"])
      ),
    },
    "observation": {
      "order": list(actor_terms),
      "dims": obs_dims,
    },
    "commands": {
      "names": metadata["command_names"],
    },
  }

  config_path = output_dir / "config.yaml"
  dump_yaml(config_path, deploy_config)
  print(f"[INFO] Saved: {config_path}")


def run_export(task_id: str, cfg: ExportConfig) -> None:
  configure_torch_backends()

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # Use 1 env for metadata extraction.
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)

  # Resolve checkpoint path.
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
  elif cfg.wandb_run_path is not None:
    log_root_path = (
      Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    ).resolve()
    resume_path, _ = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
  else:
    # Try to find the latest local checkpoint.
    log_root_path = (
      Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    ).resolve()
    resume_path = get_checkpoint_path(log_root_path)

  print(f"[INFO] Loading checkpoint: {resume_path}")

  # Create runner and load weights.
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=cfg.device
  )

  output_dir = Path(cfg.output)
  output_dir.mkdir(parents=True, exist_ok=True)

  # 1. Export JIT model (actor + normalizer).
  runner.export_policy_to_jit(str(output_dir), filename="policy.pt")
  print(f"[INFO] Saved: {output_dir / 'policy.pt'}")

  # 2. Generate config.yaml from training environment metadata.
  generate_deploy_config(env, resume_path, output_dir)

  print(f"[INFO] Deploy package ready at: {output_dir}")
  env.close()


def main():
  import mjlab.tasks  # noqa: F401 — populate registry

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    ExportConfig,
    args=remaining_args,
    default=ExportConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  run_export(task_id=chosen_task, cfg=args)


if __name__ == "__main__":
  main()
