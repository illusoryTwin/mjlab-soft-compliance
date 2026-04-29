# Mass–spring–damper: how the continuous-time matrix \(A\) is built

This note documents the state-space layout used in [`mass_spring_damper_model.py`](../src/mjlab/tasks/velocity/mdp/mass_spring_damper_model.py) (`MassSpringDamperModel._compute_discrete_matrices`).

## State ordering

For \(n =\) `n_active` joints, the state vector is stacked as **all positions, then all velocities**:

\[
x = [q_1,\, \ldots,\, q_n,\, \dot q_1,\, \ldots,\, \dot q_n]^\top \in \mathbb{R}^{2n}.
\]

So \(q_i\) is at **index** \(i\) (0-based) and \(\dot q_i\) is at **index** \(n + i\).

## Physics per joint

For each active joint \(i\):

\[
M_i \ddot q_i = -K_i q_i - D_i \dot q_i + u_i
\qquad\Rightarrow\qquad
\ddot q_i = -\frac{K_i}{M_i} q_i - \frac{D_i}{M_i} \dot q_i + \frac{1}{M_i} u_i.
\]

We write \(\dot x = A x + B u\). The input \(u\) appears in \(B\); **\(A\)** only encodes the homogeneous part (drop the \(u\) term).

## First block of rows (indices \(0 .. n-1\))

By definition,

\[
\frac{d}{dt} q_i = \dot q_i.
\]

So the derivative of component \(i\) equals component \(n+i\) with coefficient 1, and nothing else. In matrix form:

- **Top-left** \(n \times n\) block: zeros.
- **Top-right** \(n \times n\) block: identity \(I\).

That matches `A[:, :n, n:] = eye_n` in code.

## Second block of rows (indices \(n+i\))

Now the time derivative of \(\dot q_i\) is \(\ddot q_i\). Ignoring \(u\) for \(A\):

\[
\frac{d}{dt}(\dot q_i)
= \ddot q_i
= -\frac{K_i}{M_i} q_i - \frac{D_i}{M_i} \dot q_i.
\]

In the big vector \(x\):

- \(\dot q_i\) is state component **\(n+i\)**.
- The rate of change of that component depends on:
  - \(q_i\) at index **\(i\)** with factor **\(-K_i/M_i\)**,
  - \(\dot q_i\) at index **\(n+i\)** with factor **\(-D_i/M_i\)**.

So **row \(n+i\)** of \(A\) has:

- column **\(i\)**: \(-K_i/M_i\)
- column **\(n+i\)**: \(-D_i/M_i\)
- all other columns: **0**

Joints are **uncoupled**: joint \(i\) does not use \(q_j\) or \(\dot q_j\) for \(j \neq i\). So in the lower half you only see **diagonals** in two places:

- **Lower-left** \(n \times n\): entries \((n+i,\, i)\)
- **Lower-right** \(n \times n\): entries \((n+i,\, n+i)\)

Code uses `diag_idx = torch.arange(n)`:

```text
A[n + i, i]         = -K_i / M_i
A[n + i, n + i]     = -D_i / M_i
```

(`K` and `D` can be per-environment tensors; `M_active` broadcasts per DOF.)

## Tiny numeric picture (\(n = 2\))

State order: \(x = [q_1,\, q_2,\, \dot q_1,\, \dot q_2]^\top\).

With \(k_i = K_i/M_i\) and \(d_i = D_i/M_i\), \(A\) is:

```text
           q1   q2   dq1  dq2
q1'    [    0    0    1    0  ]
q2'    [    0    0    0    1  ]
dq1'   [   -k1   0   -d1   0  ]
dq2'   [    0  -k2   0   -d2  ]
```

- Rows 1–2: position dynamics = copy the matching velocity.
- Rows 3–4: velocity dynamics = spring and damper on the **same** joint only.

## Discrete matrices (short pointer)

The simulator steps with interval `dt`. With piecewise-constant \(u\) over each step, the usual discretization is:

\[
x_{k+1} = A_d x_k + B_d u_k,
\quad
A_d = e^{A\, dt},
\quad
B_d = A^{-1}(A_d - I)\,B
\]

as implemented with `torch.linalg.matrix_exp` and the `Bd` computation in code. A batch of per-environment \(A\) matrices yields per-environment \(A_d\) and \(B_d\).

## Plain-language summary

- **Top of \(A\)**: “where each bend is heading” = its current speed.
- **Bottom of \(A\)**: “how each speed changes” = push back from **its own** bend (spring) and **its own** speed (damper).
- **Zeros off the joint’s own columns**: joints do not interact in this diagonal MSD model.
