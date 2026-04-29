import mujoco_warp as mjwarp 
import warp as wp

from dataclasses import dataclass, field
from typing import Dict, List
import re
import torch


def _g1_default_monitored_bodies() -> List[str]:
    """Lazy import so G1 parameters stay in `config/g1/compliance_params`."""
    from mjlab.tasks.velocity.config.g1.compliance_params import (
        UNITREE_G1_DEFAULT_MONITORED_BODIES,
    )

    return list(UNITREE_G1_DEFAULT_MONITORED_BODIES)


def _g1_default_stiffness_scales() -> Dict[str, float]:
    from mjlab.tasks.velocity.config.g1.compliance_params import (
        UNITREE_G1_STIFFNESS_SCALES,
    )

    return dict(UNITREE_G1_STIFFNESS_SCALES)


@dataclass 
class ComplianceManagerCfg:
    enabled: bool = True 
    robot_name: str = "robot" 
    monitored_bodies: List[str] = field(default_factory=_g1_default_monitored_bodies)

    stiffness_config: Dict[str, float] = field(
        default_factory=_g1_default_stiffness_scales,
    )
    dt: float = 0.02 
    base_stiffness: float = 10.0 # 60.0
    base_inertia: float = 0.5
    damping_ratio: float = 1.0


class ComplianceManager:
    def __init__(self, cfg: ComplianceManagerCfg, entity: "Entity", env):
        self.cfg = cfg
        self._env = env
        self._entity = entity
        self._device = env.device
        self._num_envs = env.num_envs

        self._body_ids, self._body_names = entity.find_bodies(cfg.monitored_bodies)
        self._joint_dof_ids = entity.indexing.joint_v_adr
        self._stiffness_scales = self._build_stiffness_scales(
            cfg.stiffness_config, entity.joint_names
        )
        # Warp Jacobian buffers
        nv = env.sim.mj_model.nv
        with wp.ScopedDevice(env.sim.wp_device):
            self._jacp_wp = wp.zeros((self._num_envs, 3, nv), dtype=float)
            self._jacr_wp = wp.zeros((self._num_envs, 3, nv), dtype=float)
            self._point_wp = wp.zeros(self._num_envs, dtype=wp.vec3)
            self._body_wp = wp.zeros(self._num_envs, dtype=wp.int32)

        self._jacp_torch = wp.to_torch(self._jacp_wp)
        self._jacr_torch = wp.to_torch(self._jacr_wp)
        self._point_torch = wp.to_torch(self._point_wp).view(self._num_envs, 3)

        self.last_joint_torques = torch.zeros(
            self._num_envs, entity.num_joints, device=self._device
        )
        self._msd_system = self._setup_msd_system()

    def _setup_msd_system(self):
        from mjlab.tasks.velocity.mdp.mass_spring_damper_model import (
            MassSpringDamperModel,
        )

        return MassSpringDamperModel(
            n_dofs=self._entity.num_joints,
            dt=self.cfg.dt,
            base_inertia=self.cfg.base_inertia,
            base_stiffness=self.cfg.base_stiffness,
            stiffness_scales=self._stiffness_scales,
            num_envs=self._num_envs,
            device=self._device,
        )

    @staticmethod
    def _build_stiffness_scales(
        stiffness_config: dict, joint_names: tuple[str, ...]) -> dict[int, float]:
        """Build stiffness scales dict mapping joint index -> scale."""
        scales = {}
        for pattern, scale in stiffness_config.items():
            regex = re.compile(pattern)
            for i, name in enumerate(joint_names):
                if regex.fullmatch(name):
                    scales[i] = scale
        return scales


    def _compute_joint_torques(self) -> torch.Tensor:
        joint_dof_ids = self._joint_dof_ids
        total_tau = torch.zeros(
            self._num_envs, self._entity.num_joints, device=self._device
        )

        body_pos = self._entity.data.body_com_pos_w
        body_force = self._entity.data.body_external_force   # world frame
        body_torque = self._entity.data.body_external_torque  # world frame
        global_body_ids = self._entity.indexing.body_ids[self._body_ids]

        for i, local_body_id in enumerate(self._body_ids):
            global_bid = global_body_ids[i].item()

            # Set the point for Jacobian computation (body COM position)
            self._point_torch[:] = body_pos[:, local_body_id]
            self._body_wp.fill_(global_bid)


            # Compute J_p (3×nv) and J_r (3×nv) for this body
            with wp.ScopedDevice(self._env.sim.wp_device):
                mjwarp.jac(
                    self._env.sim.wp_model,
                    self._env.sim.wp_data,
                    self._jacp_wp,
                    self._jacr_wp,
                    self._point_wp,
                    self._body_wp,
                )

            jacp = self._jacp_torch[:, :, joint_dof_ids] 
            jacr = self._jacr_torch[:, :, joint_dof_ids]

            forces = body_force[:, local_body_id]
            torques = body_torque[:, local_body_id]

            total_tau += (
                torch.bmm(jacp.transpose(1, 2), forces.unsqueeze(-1)).squeeze(-1)
                + torch.bmm(jacr.transpose(1, 2), torques.unsqueeze(-1)).squeeze(-1)
            )

        return total_tau

    @property
    def deformations(self) -> torch.Tensor:
        return self._msd_system.state["q_def"]

    def reset(self, env_ids: torch.Tensor | None = None):
        self._msd_system.reset(env_ids)

    def compute(self):
        joint_torques = self._compute_joint_torques()
        self.last_joint_torques = joint_torques
        self._msd_system.update(joint_torques)
        return self.deformations

