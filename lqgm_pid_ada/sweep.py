"""
sweep.py — Full backward/forward sweeps over all K PWC intervals.

Public API
----------
  backward_sweep(protocol, *, bc_eps) → List[CoeffState]   length K+1
  forward_sweep(protocol, x0, *, bc_eps) → List[CoeffState] length K+1

Storage convention
------------------
Both functions return a list of K+1 CoeffState objects indexed by grid
position j = 0, …, K, where j corresponds to t = breaks[j].

  backward_sweep[j] = backward Green-function coefficients at t = breaks[j]
  forward_sweep[j]  = forward  Green-function coefficients at t = breaks[j]

The δ-BC is placed at the *terminal* end:
  backward_sweep[K] ≈ δ(x − ·) at t = 1   (approximated by (1/ε)I)
  forward_sweep[0]  ≈ δ(· − x₀) at t = 0  (B set from x₀; see below)

Forward BC with Gaussian initial condition
------------------------------------------
When x₀ is a fixed point (delta initial condition), the forward coefficients
at t=0 should be the same δ-BC used in the backward sweep.

When x₀ ~ N(0, Σ₀) (Gaussian initial), the forward BC is:
  A⁺(0) = Σ₀⁻¹,  B⁺(0) = Σ₀⁻¹,  C⁺(0) = Σ₀⁻¹,
  θ_x(0) = Σ₀⁻¹ m₀,  θ_y(0) = Σ₀⁻¹ m₀

where m₀ = mean of x₀.  For a delta initial condition (Σ₀ → 0), this
again collapses to the δ-BC with the linear terms encoding x₀.

In practice, `forward_sweep` accepts an optional `x0_mean` and `x0_cov`
to specify the forward BC; if both are None, the δ-BC is used (equivalent
to x₀ = 0 deterministically).

Isotropic validation
--------------------
For sigma=0, beta=b*I, nu uniform, the sweep reproduces the per-interval
a_minus/b_minus/c_minus values from guided_continuous.ContinuousCoeffs
to machine precision across all K+1 grid points.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .core import CoeffState, MatrixPWCProtocol
from .coeff_propagator import backward_interval, forward_interval, delta_bc

Tensor = torch.Tensor

# Default δ-BC precision (relative to the smallest interval length)
_DEFAULT_BC_EPS = 1e-6


# ---------------------------------------------------------------------------
# Backward sweep  (t = 1 → 0, storing CoeffState at each breakpoint)
# ---------------------------------------------------------------------------

def backward_sweep(
    protocol: MatrixPWCProtocol,
    *,
    bc_eps: float = _DEFAULT_BC_EPS,
) -> List[CoeffState]:
    """Propagate backward Green-function coefficients from t=1 to t=0.

    Parameters
    ----------
    protocol : MatrixPWCProtocol
    bc_eps   : precision of the δ-BC approximation at t=1

    Returns
    -------
    states : list of K+1 CoeffState objects
        states[j] = backward coefficients at t = breaks[j]
        states[K] = δ-BC at t = breaks[K] = 1  (starting point)
        states[0] = coefficients propagated all the way to t = breaks[0] = 0
    """
    K   = protocol.K
    d   = protocol.d
    dev = protocol.device
    dty = protocol.dtype

    states: List[Optional[CoeffState]] = [None] * (K + 1)

    # Terminal boundary condition at t = 1
    states[K] = delta_bc(d, dev, dty, eps=bc_eps)

    # Sweep backward: interval k runs from breaks[k] to breaks[k+1]
    # We propagate from states[k+1] → states[k]
    for k in range(K - 1, -1, -1):
        tau   = float((protocol.breaks[k + 1] - protocol.breaks[k]).item())
        sigma = protocol.sigma[k]   # (d, d)
        beta  = protocol.beta[k]    # (d, d)
        nu    = protocol.nu[k]      # (d,)

        states[k] = backward_interval(sigma, beta, nu, tau, states[k + 1])

    return states  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Forward sweep  (t = 0 → 1, storing CoeffState at each breakpoint)
# ---------------------------------------------------------------------------

def forward_sweep(
    protocol: MatrixPWCProtocol,
    *,
    bc_eps:  float           = _DEFAULT_BC_EPS,
    x0_mean: Optional[Tensor] = None,   # (d,)  initial mean; None → 0
    x0_cov:  Optional[Tensor] = None,   # (d,d) initial covariance; None → δ-BC
) -> List[CoeffState]:
    """Propagate forward Green-function coefficients from t=0 to t=1.

    Parameters
    ----------
    protocol : MatrixPWCProtocol
    bc_eps   : precision of the δ-BC approximation at t=0
    x0_mean  : (d,) mean of the initial distribution; default zeros
    x0_cov   : (d,d) covariance of the initial distribution.
               If None, a δ-function BC is used (x₀ deterministic at x0_mean).

    Returns
    -------
    states : list of K+1 CoeffState objects
        states[0] = forward BC at t = breaks[0] = 0  (starting point)
        states[K] = coefficients propagated all the way to t = breaks[K] = 1
    """
    K   = protocol.K
    d   = protocol.d
    dev = protocol.device
    dty = protocol.dtype

    states: List[Optional[CoeffState]] = [None] * (K + 1)

    # Initial boundary condition at t = 0
    states[0] = _forward_bc(d, dev, dty, bc_eps=bc_eps,
                             x0_mean=x0_mean, x0_cov=x0_cov)

    # Sweep forward: interval k runs from breaks[k] to breaks[k+1]
    for k in range(K):
        tau   = float((protocol.breaks[k + 1] - protocol.breaks[k]).item())
        sigma = protocol.sigma[k]
        beta  = protocol.beta[k]
        nu    = protocol.nu[k]

        states[k + 1] = forward_interval(sigma, beta, nu, tau, states[k])

    return states  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sweep pair  (convenience: run both sweeps in one call)
# ---------------------------------------------------------------------------

def full_sweep(
    protocol: MatrixPWCProtocol,
    *,
    bc_eps:  float           = _DEFAULT_BC_EPS,
    x0_mean: Optional[Tensor] = None,
    x0_cov:  Optional[Tensor] = None,
) -> tuple[List[CoeffState], List[CoeffState]]:
    """Run both backward and forward sweeps.

    Returns
    -------
    (bwd_states, fwd_states) : each a list of K+1 CoeffState objects
    """
    bwd = backward_sweep(protocol, bc_eps=bc_eps)
    fwd = forward_sweep(protocol, bc_eps=bc_eps, x0_mean=x0_mean, x0_cov=x0_cov)
    return bwd, fwd


# ---------------------------------------------------------------------------
# Internal: build forward BC from initial distribution
# ---------------------------------------------------------------------------

def _forward_bc(
    d:       int,
    device:  torch.device,
    dtype:   torch.dtype,
    *,
    bc_eps:  float,
    x0_mean: Optional[Tensor],
    x0_cov:  Optional[Tensor],
) -> CoeffState:
    """Build the forward boundary condition at t = 0.

    Delta initial condition (x0_cov is None)
    -----------------------------------------
    A⁺(0) = B⁺(0) = C⁺(0) = (1/ε) I
    θ_x(0) = θ_y(0) = (1/ε) x0_mean   (encodes the fixed starting point)

    Gaussian initial condition (x0_cov provided)
    ---------------------------------------------
    A⁺(0) = Σ₀⁻¹
    B⁺(0) = Σ₀⁻¹         (matching backward convention)
    C⁺(0) = Σ₀⁻¹
    θ_x(0) = Σ₀⁻¹ m₀
    θ_y(0) = Σ₀⁻¹ m₀
    """
    bc = delta_bc(d, device, dtype, eps=bc_eps)

    if x0_mean is None and x0_cov is None:
        return bc   # pure δ at origin

    if x0_cov is None:
        # Delta at x0_mean: shift the linear terms
        m = x0_mean.to(device=device, dtype=dtype)
        scale = torch.tensor(1.0 / bc_eps, device=device, dtype=dtype)
        return CoeffState(
            A       = bc.A,
            B       = bc.B,
            C       = bc.C,
            theta_x = scale * m,
            theta_y = scale * m,
        )

    # Gaussian initial condition
    x0_cov  = x0_cov.to(device=device, dtype=dtype)
    x0_mean = (x0_mean if x0_mean is not None
               else torch.zeros(d, device=device, dtype=dtype))
    x0_mean = x0_mean.to(device=device, dtype=dtype)
    P0  = torch.linalg.inv(x0_cov)          # (d, d)  precision
    th0 = P0 @ x0_mean                       # (d,)
    return CoeffState(
        A       = P0,
        B       = P0,
        C       = P0,
        theta_x = th0,
        theta_y = th0,
    )
