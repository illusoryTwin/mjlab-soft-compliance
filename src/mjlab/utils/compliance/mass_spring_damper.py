import torch


class MassSpringDamperModel:
  def __init__(
    self,
    n_dofs: int,
    dt: float,
    base_inertia: float,
    base_stiffness: float,
    stiffness_scales: dict[int, float],
    num_envs=1,
    device="cuda",
  ):
    self.n_dofs = n_dofs
    self.dt = dt
    self.device = device
    self.num_envs = num_envs
    self.active_idx = sorted(stiffness_scales.keys())
    self.active_idx_torch = torch.tensor(
      self.active_idx, dtype=torch.long, device=device
    )
    self.n_active = len(self.active_idx)
    self.M_active = torch.full(
      (self.n_active,), base_inertia, dtype=torch.float32, device=device
    )
    scales_list = [stiffness_scales[idx] for idx in self.active_idx]
    self.scales_tensor = torch.tensor(
      scales_list, dtype=torch.float32, device=device
    )
    self._set_stiffness(base_stiffness)
    self.state = self._create_msd_state()
    z = (self.num_envs, self.n_active)
    self._last_qdd = torch.zeros(z, dtype=torch.float32, device=device)
    self._last_tau_active = torch.zeros(z, dtype=torch.float32, device=device)

  def set_stiffness(self, base_stiffness: float | torch.Tensor):
    """Update base stiffness and recompute MSD matrices."""
    self._set_stiffness(base_stiffness)

  def _set_stiffness(self, base_stiffness: float | torch.Tensor):
    if isinstance(base_stiffness, (int, float)):
      base_stiffness = torch.full(
        (self.num_envs,),
        float(base_stiffness),
        dtype=torch.float32,
        device=self.device,
      )
    else:
      base_stiffness = base_stiffness.to(dtype=torch.float32, device=self.device)
      if base_stiffness.dim() == 0:
        base_stiffness = base_stiffness.expand(self.num_envs)
    self.K = base_stiffness.unsqueeze(1) * self.scales_tensor.unsqueeze(0)
    self.D = 2.0 * torch.sqrt(self.M_active.unsqueeze(0) * self.K)
    self.Ad, self.Bd = self._compute_discrete_matrices()

  def _create_msd_state(self):
    return {
      "q_def": torch.zeros(
        (self.num_envs, self.n_active), dtype=torch.float32, device=self.device
      ),
      "qd_def": torch.zeros(
        (self.num_envs, self.n_active), dtype=torch.float32, device=self.device
      ),
    }

  def _compute_discrete_matrices(
    self,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if self.n_active == 0:
      return None, None
    n = self.n_active
    n_envs = self.num_envs
    A = torch.zeros(
      n_envs, n * 2, n * 2, dtype=torch.float32, device=self.device
    )
    eye_n = torch.eye(n, dtype=torch.float32, device=self.device)
    A[:, :n, n:] = eye_n
    diag_idx = torch.arange(n, device=self.device)
    A[:, n + diag_idx, diag_idx] = -self.K / self.M_active
    A[:, n + diag_idx, n + diag_idx] = -self.D / self.M_active
    B = torch.zeros(
      n_envs, n * 2, n, dtype=torch.float32, device=self.device
    )
    B[:, n + diag_idx, diag_idx] = 1.0 / self.M_active
    Ad = torch.linalg.matrix_exp(A * self.dt)
    I_2n = torch.eye(
      2 * n, dtype=torch.float32, device=self.device
    ).unsqueeze(0)
    A_inv = torch.linalg.inv(A)
    Bd = A_inv @ (Ad - I_2n) @ B
    return Ad, Bd

  def update(self, external_torques: torch.Tensor) -> None:
    if self.n_active == 0:
      return
    external_torques = external_torques.to(device=self.device)
    tau_active = external_torques[:, self.active_idx_torch]
    M_b = self.M_active.unsqueeze(0)
    self._last_qdd[:] = (
      tau_active - self.D * self.state["qd_def"] - self.K * self.state["q_def"]
    ) / M_b
    self._last_tau_active[:] = tau_active
    x = torch.cat(
      [self.state["q_def"], self.state["qd_def"]], dim=1
    )
    x_next = (
      torch.bmm(self.Ad, x.unsqueeze(-1))
      + torch.bmm(self.Bd, tau_active.unsqueeze(-1))
    ).squeeze(-1)
    self.state["q_def"][:] = x_next[:, : self.n_active]
    self.state["qd_def"][:] = x_next[:, self.n_active :]

  def reset(self, env_ids: torch.Tensor = None):
    """Reset MSD state to zero for specified environments."""
    if env_ids is None:
      self.state["q_def"][:] = 0.0
      self.state["qd_def"][:] = 0.0
      if self.n_active > 0:
        self._last_qdd[:] = 0.0
        self._last_tau_active[:] = 0.0
    else:
      self.state["q_def"][env_ids] = 0.0
      self.state["qd_def"][env_ids] = 0.0
      if self.n_active > 0:
        self._last_qdd[env_ids] = 0.0
        self._last_tau_active[env_ids] = 0.0
