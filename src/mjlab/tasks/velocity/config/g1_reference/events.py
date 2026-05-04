"""Backward-compatible re-exports. Prefer ``mjlab.tasks.velocity.mdp.compliance.events``."""

from mjlab.tasks.velocity.mdp.compliance.events import (
  apply_compliance_forces,
  apply_compliance_torques,
  apply_constant_torque,
)

__all__ = [
  "apply_compliance_forces",
  "apply_compliance_torques",
  "apply_constant_torque",
]
