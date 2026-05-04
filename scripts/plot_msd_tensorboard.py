#!/usr/bin/env python3
"""Load mjlab MSD scalars from TensorBoard event files and plot q, qd, qdd and dynamic terms.

Expects tags written by ComplianceCommand, e.g. Compliance/msd/q/<joint_name>.

Dependencies: matplotlib, tensorboard (for event_accumulator).

Example:
  python plot_msd_tensorboard.py --logdir logs/rsl_rl/g1_soft/2026-04-03_08-26-48 \\
    --joint left_elbow_joint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def event_files(logdir: Path) -> list[Path]:
    files = sorted(logdir.rglob("events.out.tfevents.*"))
    return [p for p in files if p.is_file()]


def load_scalar(
    acc: event_accumulator.EventAccumulator, tag: str
) -> tuple[np.ndarray, np.ndarray]:
    scalars = acc.Scalars(tag)
    steps = np.array([s.step for s in scalars], dtype=np.float64)
    values = np.array([s.value for s in scalars], dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], values[order]


def interp_on_master(
    master: np.ndarray, steps: np.ndarray, values: np.ndarray
) -> np.ndarray:
    if len(steps) == 0:
        return np.full_like(master, np.nan)
    return np.interp(master, steps, values, left=np.nan, right=np.nan)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="Run directory containing events.out.tfevents.* (searched recursively)",
    )
    p.add_argument(
        "--joint",
        type=str,
        required=True,
        help="Joint name as in MJCF / logging tags (e.g. left_elbow_joint)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="Compliance/msd",
        help="Scalar prefix before /q/, /qd/, ...",
    )
    p.add_argument(
        "--event",
        type=Path,
        default=None,
        help="Explicit event file (overrides --logdir discovery)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, save fig to this path instead of showing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.event is not None:
        ev_paths = [args.event]
    else:
        ev_paths = event_files(args.logdir)
    if not ev_paths:
        raise SystemExit(f"No TensorBoard event files under {args.logdir}")
    ev_path = ev_paths[-1]
    prefix = args.prefix.rstrip("/")
    joint = args.joint
    tags = {
        "q": f"{prefix}/q/{joint}",
        "qd": f"{prefix}/qd/{joint}",
        "qdd": f"{prefix}/qdd/{joint}",
        "M": f"{prefix}/M/{joint}",
        "K": f"{prefix}/K/{joint}",
        "D": f"{prefix}/D/{joint}",
    }
    acc = event_accumulator.EventAccumulator(str(ev_path))
    acc.Reload()
    available = set(acc.Tags()["scalars"])
    missing = [t for k, t in tags.items() if t not in available]
    if missing:
        sample = [x for x in sorted(available) if joint in x][:15]
        hint = "\n  ".join(sample) if sample else "(no tags containing joint name)"
        raise SystemExit(
            "Missing scalar tags:\n  "
            + "\n  ".join(missing)
            + f"\nUsing file: {ev_path}\nExample tags for this joint:\n  {hint}"
        )
    series = {k: load_scalar(acc, t) for k, t in tags.items()}
    master_steps, _ = series["q"]
    if len(master_steps) == 0:
        raise SystemExit("Series q is empty")
    aligned = {
        k: interp_on_master(master_steps, series[k][0], series[k][1])
        for k in tags
    }
    q = aligned["q"]
    qd = aligned["qd"]
    qdd = aligned["qdd"]
    m = aligned["M"]
    k = aligned["K"]
    d = aligned["D"]
    m_qdd = m * qdd
    k_q = k * q
    d_qd = d * qd
    tau_tag = f"{prefix}/tau/{joint}"
    ext_tag = f"Compliance/ext_torque/{joint}"
    force_tag: str | None = None
    if tau_tag in available:
        force_tag = tau_tag
    elif ext_tag in available:
        force_tag = ext_tag
    if force_tag is None:
        print(
            "Warning: no scalar for external joint torque "
            f"({tau_tag} or {ext_tag}); second plot omits force (τ)",
            file=sys.stderr,
        )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax0 = axes[0]
    ax0.plot(master_steps, q, label="q")
    ax0.plot(master_steps, qd, label="dot q")
    ax0.plot(master_steps, qdd, label="ddot q")
    ax0.set_ylabel("state")
    ax0.set_title(f"MSD state ({joint})")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax1 = axes[1]
    ax1.plot(master_steps, m_qdd, label="M ddot q")
    ax1.plot(master_steps, k_q, label="K q")
    ax1.plot(master_steps, d_qd, label="D dot q")
    if force_tag is not None:
        st, tv = load_scalar(acc, force_tag)
        f_on_master = interp_on_master(master_steps, st, tv)
        ax1.plot(
            master_steps,
            f_on_master,
            color="k",
            linewidth=1.8,
            alpha=0.85,
            label="force (τ ext, Nm)",
        )
    ax1.set_xlabel("logged step")
    ax1.set_ylabel("torque (Nm)")
    ax1.set_title(
        f"MSD torque balance ({joint}); τ is joint torque from external wrench"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
