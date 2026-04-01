"""Tests for MassSpringDamperModel."""

import pytest
import torch
from conftest import get_test_device

from mjlab.tasks.velocity.mdp.mass_spring_damper_model import (
  MassSpringDamperModel,
)


@pytest.fixture
def device():
  return get_test_device()


# -- active_idx construction --------------------------------------------------


def test_active_idx_is_sorted_list(device):
  """active_idx must be a sorted list, not a dict_keys view."""
  scales = {3: 1.0, 1: 0.5, 5: 2.0}
  msd = MassSpringDamperModel(
    n_dofs=6, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=1, device=device,
  )
  assert isinstance(msd.active_idx, list)
  assert msd.active_idx == [1, 3, 5]


def test_n_active_matches_stiffness_scales(device):
  scales = {0: 1.0, 2: 1.0, 4: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=6, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=4, device=device,
  )
  assert msd.n_active == 3


def test_active_idx_torch_matches_active_idx(device):
  scales = {2: 0.8, 0: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=4, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=1, device=device,
  )
  expected = torch.tensor([0, 2], dtype=torch.long, device=device)
  assert torch.equal(msd.active_idx_torch, expected)


def test_empty_stiffness_scales(device):
  """No active DOFs should produce None matrices."""
  msd = MassSpringDamperModel(
    n_dofs=4, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales={}, num_envs=2, device=device,
  )
  assert msd.n_active == 0
  assert msd.Ad is None
  assert msd.Bd is None


# -- K, D computation ---------------------------------------------------------


def test_stiffness_K_shape_and_values(device):
  scales = {0: 1.0, 1: 2.0}
  msd = MassSpringDamperModel(
    n_dofs=3, dt=0.02, base_inertia=0.5, base_stiffness=10.0,
    stiffness_scales=scales, num_envs=2, device=device,
  )
  # K[env, dof] = base_stiffness * scale
  assert msd.K.shape == (2, 2)
  assert torch.allclose(msd.K[0], torch.tensor([10.0, 20.0], device=device))
  # Same for all envs when scalar base_stiffness
  assert torch.allclose(msd.K[0], msd.K[1])


def test_critical_damping(device):
  """D = 2 * sqrt(M * K) for critical damping."""
  scales = {0: 1.0}
  M, K_base = 0.5, 60.0
  msd = MassSpringDamperModel(
    n_dofs=1, dt=0.02, base_inertia=M, base_stiffness=K_base,
    stiffness_scales=scales, num_envs=1, device=device,
  )
  expected_D = 2.0 * (M * K_base * 1.0) ** 0.5
  assert torch.allclose(
    msd.D, torch.tensor([[expected_D]], device=device), atol=1e-5
  )


# -- discrete matrices --------------------------------------------------------


def test_discrete_matrix_shapes(device):
  n_active = 3
  num_envs = 4
  scales = {0: 1.0, 2: 1.0, 4: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=6, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=num_envs, device=device,
  )
  assert msd.Ad.shape == (num_envs, 2 * n_active, 2 * n_active)
  assert msd.Bd.shape == (num_envs, 2 * n_active, n_active)


def test_Ad_is_finite(device):
  scales = {0: 1.0, 1: 0.5}
  msd = MassSpringDamperModel(
    n_dofs=2, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=2, device=device,
  )
  assert torch.isfinite(msd.Ad).all()
  assert torch.isfinite(msd.Bd).all()


# -- state creation and reset -------------------------------------------------


def test_initial_state_is_zero(device):
  scales = {0: 1.0, 1: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=3, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=2, device=device,
  )
  assert torch.equal(msd.state["q_def"], torch.zeros(2, 2, device=device))
  assert torch.equal(msd.state["qd_def"], torch.zeros(2, 2, device=device))


def test_reset_all(device):
  scales = {0: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=1, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=2, device=device,
  )
  msd.state["q_def"][:] = 99.0
  msd.state["qd_def"][:] = 99.0
  msd.reset()
  assert (msd.state["q_def"] == 0.0).all()
  assert (msd.state["qd_def"] == 0.0).all()


def test_reset_specific_envs(device):
  scales = {0: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=1, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=4, device=device,
  )
  msd.state["q_def"][:] = 5.0
  msd.reset(torch.tensor([0, 2], device=device))
  assert msd.state["q_def"][0].item() == 0.0
  assert msd.state["q_def"][1].item() == 5.0
  assert msd.state["q_def"][2].item() == 0.0
  assert msd.state["q_def"][3].item() == 5.0


# -- update (discrete step) ---------------------------------------------------


def test_update_with_zero_torque_stays_at_zero(device):
  """From zero state + zero input, state should remain zero."""
  scales = {0: 1.0, 1: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=3, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=2, device=device,
  )
  torques = torch.zeros(2, 3, device=device)
  msd.update(torques)
  assert torch.allclose(msd.state["q_def"], torch.zeros(2, 2, device=device))
  assert torch.allclose(msd.state["qd_def"], torch.zeros(2, 2, device=device))


def test_update_nonzero_torque_produces_deformation(device):
  """A constant torque should produce nonzero deformation."""
  scales = {0: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=2, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=1, device=device,
  )
  torques = torch.tensor([[10.0, 0.0]], device=device)
  for _ in range(100):
    msd.update(torques)
  # After many steps, q_def should settle near tau/K = 10/60
  assert msd.state["q_def"][0, 0].item() > 0.0


def test_update_only_affects_active_dofs(device):
  """Torque on inactive DOF 1 should not affect active DOF 0."""
  scales = {0: 1.0}  # only DOF 0 is active
  msd = MassSpringDamperModel(
    n_dofs=3, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=1, device=device,
  )
  # Torque only on DOF 1 (inactive)
  torques = torch.tensor([[0.0, 100.0, 0.0]], device=device)
  msd.update(torques)
  assert msd.state["q_def"][0, 0].item() == 0.0


def test_update_converges_to_steady_state(device):
  """Under constant torque, MSD should converge to q_ss = tau / K."""
  K_base = 60.0
  scale = 1.0
  tau = 10.0
  scales = {0: scale}
  msd = MassSpringDamperModel(
    n_dofs=1, dt=0.02, base_inertia=0.5, base_stiffness=K_base,
    stiffness_scales=scales, num_envs=1, device=device,
  )
  torques = torch.tensor([[tau]], device=device)
  for _ in range(1000):
    msd.update(torques)
  expected_ss = tau / (K_base * scale)
  assert abs(msd.state["q_def"][0, 0].item() - expected_ss) < 1e-3


# -- set_stiffness -------------------------------------------------------------


def test_set_stiffness_updates_matrices(device):
  scales = {0: 1.0}
  msd = MassSpringDamperModel(
    n_dofs=1, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=1, device=device,
  )
  Ad_old = msd.Ad.clone()
  msd.set_stiffness(120.0)
  # Matrices should change
  assert not torch.equal(msd.Ad, Ad_old)


def test_set_stiffness_per_env(device):
  """Per-env stiffness should give different K per environment."""
  scales = {0: 1.0}
  num_envs = 3
  msd = MassSpringDamperModel(
    n_dofs=1, dt=0.02, base_inertia=0.5, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=num_envs, device=device,
  )
  per_env = torch.tensor([10.0, 20.0, 30.0], device=device)
  msd.set_stiffness(per_env)
  assert msd.K[0, 0].item() == pytest.approx(10.0)
  assert msd.K[1, 0].item() == pytest.approx(20.0)
  assert msd.K[2, 0].item() == pytest.approx(30.0)


# -- M_active ------------------------------------------------------------------


def test_M_active_shape_and_value(device):
  scales = {1: 0.5, 3: 2.0}
  inertia = 0.7
  msd = MassSpringDamperModel(
    n_dofs=5, dt=0.02, base_inertia=inertia, base_stiffness=60.0,
    stiffness_scales=scales, num_envs=1, device=device,
  )
  assert msd.M_active.shape == (2,)
  assert torch.allclose(
    msd.M_active,
    torch.tensor([inertia, inertia], device=device),
  )


# -- ComplianceManager._setup_msd_system via _build_stiffness_scales ----------


G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def _make_g1_msd(device, num_envs=2):
  """Build MSD from G1 stiffness config, same as ComplianceManager would."""
  from mjlab.tasks.velocity.mdp.compliance_manager import (
    ComplianceManager,
    ComplianceManagerCfg,
  )

  cfg = ComplianceManagerCfg()
  scales = ComplianceManager._build_stiffness_scales(
    cfg.stiffness_config, G1_JOINT_NAMES
  )
  msd = MassSpringDamperModel(
    n_dofs=len(G1_JOINT_NAMES),
    dt=cfg.dt,
    base_inertia=cfg.base_inertia,
    base_stiffness=cfg.base_stiffness,
    stiffness_scales=scales,
    num_envs=num_envs,
    device=device,
  )
  return msd, scales, cfg


def test_g1_stiffness_scales_match_config(device):
  """Every configured joint should appear in scales with correct value."""
  from mjlab.tasks.velocity.mdp.compliance_manager import (
    ComplianceManager,
    ComplianceManagerCfg,
  )

  cfg = ComplianceManagerCfg()
  scales = ComplianceManager._build_stiffness_scales(
    cfg.stiffness_config, G1_JOINT_NAMES
  )
  assert len(scales) == len(cfg.stiffness_config)
  for pattern, expected_scale in cfg.stiffness_config.items():
    idx = G1_JOINT_NAMES.index(pattern)
    assert scales[idx] == expected_scale


def test_g1_msd_construction(device):
  """MSD built from G1 config should have correct dimensions."""
  msd, scales, cfg = _make_g1_msd(device, num_envs=2)
  n_active = len(scales)
  assert msd.n_active == n_active
  assert msd.Ad.shape == (2, 2 * n_active, 2 * n_active)
  assert msd.Bd.shape == (2, 2 * n_active, n_active)
  assert torch.isfinite(msd.Ad).all()
  assert torch.isfinite(msd.Bd).all()


def test_g1_msd_step_converges(device):
  """Constant torque on first active joint converges to tau/K."""
  msd, scales, cfg = _make_g1_msd(device, num_envs=1)
  n_dofs = len(G1_JOINT_NAMES)
  first_active = msd.active_idx[0]
  tau = 10.0

  torques = torch.zeros(1, n_dofs, device=device)
  torques[:, first_active] = tau
  for _ in range(1000):
    msd.update(torques)

  expected_ss = tau / (cfg.base_stiffness * scales[first_active])
  assert abs(msd.state["q_def"][0, 0].item() - expected_ss) < 1e-3


def test_setup_msd_system_via_compliance_manager(device):
  """Test _setup_msd_system is called correctly through ComplianceManager."""
  from types import SimpleNamespace

  from mjlab.tasks.velocity.mdp.compliance_manager import (
    ComplianceManager,
    ComplianceManagerCfg,
  )

  cfg = ComplianceManagerCfg()
  num_envs = 2
  fake_env = SimpleNamespace(
    device=device, num_envs=num_envs,
  )
  fake_entity = SimpleNamespace(
    joint_names=G1_JOINT_NAMES,
    num_joints=len(G1_JOINT_NAMES),
    find_bodies=lambda names: (
      list(range(len(names))),
      list(names),
    ),
  )

  manager = ComplianceManager(cfg, fake_entity, fake_env)
  msd = manager._msd_system

  assert msd.n_dofs == len(G1_JOINT_NAMES)
  assert msd.dt == cfg.dt
  assert msd.num_envs == num_envs
  assert msd.n_active == len(cfg.stiffness_config)
  assert msd.Ad.shape == (num_envs, 2 * msd.n_active, 2 * msd.n_active)
  assert torch.isfinite(msd.Ad).all()
