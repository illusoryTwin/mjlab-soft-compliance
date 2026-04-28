"""Backward-compatible re-exports. Prefer ``mjlab.tasks.velocity.mdp.compliance_events``."""

from mjlab.tasks.velocity.mdp.compliance_events import (
  apply_compliance_forces,
  apply_compliance_torques,
  apply_constant_torque,
)

__all__ = [
  "apply_compliance_forces",
  "apply_compliance_torques",
  "apply_constant_torque",
]
