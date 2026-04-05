"""
hamiltonian.py — Hamiltonian matrix constructors for LQ-GM-PID.

Each PWC interval I_k has three Hamiltonians per branch (backward / forward):

  1. H2d   (2d × 2d)     — drives quadratic coefficients A, B     (eqs 35, 50)
  2. H_C   (3d × 3d)     — augmented system for C                 (eq 41 + fwd analogue)
  3. H_lin (2d+1 × 2d+1) — augmented system for θ_x, θ_y         (eq 45 + fwd analogue)

Public API
----------
  backward_H2d(sigma, beta)          → (2d, 2d)
  forward_H2d(sigma, beta)           → (2d, 2d)
  backward_H_C(sigma, beta, B_end)   → (3d, 3d)
  forward_H_C(sigma, beta, B_init)   → (3d, 3d)
  backward_H_lin(sigma, beta, g)     → (2d+1, 2d+1)
  forward_H_lin(sigma, beta, g)      → (2d+1, 2d+1)
  phi_blocks(H, tau)                 → (Phi, Phi11, Phi12, Phi21, Phi22)
  all_hamiltonians(...)              → (H2d, H_C, H_lin)

Sign convention — backward branch
----------------------------------
The notes (eq 35) define H^(-)_k for the *forward-in-t* linear flow:

    d/dt [p; q] = H^(-) [p; q],   H^(-) = [[ σ,  -I],
                                             [-β,  -σᵀ]]

so that A^(-) = q p^{-1} satisfies the backward Riccati ODE (eq 18).

For the PWC propagation we instead need exp(H * τ) with τ = t_{k+1} - t
*increasing* as we go backward from t_{k+1}. Since t = t_{k+1} - τ:

    d/dτ [p; q] = -H^(-) [p; q]

so the correct matrix in the exponent is **−H^(-)**:

    Φ^(-)(τ) = exp(−H^(-) · τ)

In terms of explicit blocks:

    −H^(-) = [[-σ,  I ],
               [ β,  σᵀ]]                             (backward_H2d)

Verified numerically: for σ=0, β=b·I, −H^(-) = [[0, I],[b·I, 0]],
and exp([[0,1],[b,0]]·τ) = diag_d( [[cosh wτ,  sinh wτ/w],
                                     [w sinh wτ, cosh wτ]] )
which recovers the a_minus / b_minus / c_minus coth/csch formulas in
guided_continuous.ContinuousCoeffs exactly.

Sign convention — forward branch
---------------------------------
The forward branch uses τ = t − t_k (increasing forward in time), so:

    Φ^(+)(τ) = exp(H^(+) · τ),   H^(+) = [[σ,   I ],
                                             [β,  −σᵀ]]  (forward_H2d, eq 50)

No sign flip is needed for the forward branch.
"""
from __future__ import annotations
from typing import Tuple
import torch

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dev_dty(t: Tensor) -> Tuple[torch.device, torch.dtype]:
    return t.device, t.dtype

def _I(d: int, dev, dty) -> Tensor:
    return torch.eye(d, device=dev, dtype=dty)

def _Z(n: int, m: int, dev, dty) -> Tensor:
    return torch.zeros(n, m, device=dev, dtype=dty)


# ---------------------------------------------------------------------------
# 1.  2d × 2d  Hamiltonians
# ---------------------------------------------------------------------------

def backward_H2d(sigma: Tensor, beta: Tensor) -> Tensor:
    """
    −H^(−)_k = [[ −σ_k,   I  ],
                 [  β_k,   σ_kᵀ]]   shape (2d, 2d).

    Use as:  Φ^(−)(τ) = exp(backward_H2d(σ,β) · τ)
    to propagate coefficients backward by τ = t_{k+1} − t.
    """
    d = sigma.shape[0]
    dev, dty = _dev_dty(sigma)
    H = _Z(2*d, 2*d, dev, dty)
    H[:d,  :d]  = -sigma
    H[:d,  d:]  =  _I(d, dev, dty)
    H[d:, :d]   =  beta
    H[d:, d:]   =  sigma.T
    return H


def forward_H2d(sigma: Tensor, beta: Tensor) -> Tensor:
    """
    H^(+)_k = [[ σ_k,   I  ],
                [ β_k,  −σ_kᵀ]]   shape (2d, 2d).   Eq (50).

    Use as:  Φ^(+)(τ) = exp(forward_H2d(σ,β) · τ)
    to propagate coefficients forward by τ = t − t_k.
    """
    d = sigma.shape[0]
    dev, dty = _dev_dty(sigma)
    H = _Z(2*d, 2*d, dev, dty)
    H[:d,  :d]  =  sigma
    H[:d,  d:]  =  _I(d, dev, dty)
    H[d:, :d]   =  beta
    H[d:, d:]   = -sigma.T
    return H


# ---------------------------------------------------------------------------
# 2.  3d × 3d  augmented Hamiltonians for C
# ---------------------------------------------------------------------------

def backward_H_C(sigma: Tensor, beta: Tensor, B_end: Tensor) -> Tensor:
    """
    Augmented backward system for C, shape (3d, 3d).

    Structure:
        [[ −σ,   I,   0 ],
         [  β,   σᵀ,  0 ],
         [  0,  +B^T, 0 ]]

    The bottom row tracks −C (so that C *decreases* in τ = t_{k+1}−t,
    consistent with dC/dt = B^T B > 0). The positive +B^T sign is required
    here even though this is the backward (negated-H) branch; the extra sign
    flip is absorbed into the formula C_t = C_ref − Γ₂ X⁻¹ B_ref (eq 43).

    B_end is the *terminal* (right-endpoint) value of B^(−) on this interval.
    """
    d = sigma.shape[0]
    dev, dty = _dev_dty(sigma)
    H = _Z(3*d, 3*d, dev, dty)
    H[:d,    :d]    = -sigma
    H[:d,    d:2*d] =  _I(d, dev, dty)
    H[d:2*d, :d]    =  beta
    H[d:2*d, d:2*d] =  sigma.T
    H[2*d:,  d:2*d] =  B_end.T          # +B^T (verified numerically)
    return H


def forward_H_C(sigma: Tensor, beta: Tensor, B_init: Tensor) -> Tensor:
    """
    Augmented forward system for C, shape (3d, 3d).

    Forward analogue of eq (41) (no sign flip needed):
        [[ σ,   I,    0   ],
         [ β,  −σᵀ,   0   ],
         [ 0,  B^T,   0   ]]

    B_init is the *left-endpoint* value of B^(+) on this interval.
    """
    d = sigma.shape[0]
    dev, dty = _dev_dty(sigma)
    H = _Z(3*d, 3*d, dev, dty)
    H[:d,    :d]    =  sigma
    H[:d,    d:2*d] =  _I(d, dev, dty)
    H[d:2*d, :d]    =  beta
    H[d:2*d, d:2*d] = -sigma.T
    H[2*d:,  d:2*d] =  B_init.T
    return H


# ---------------------------------------------------------------------------
# 3.  (2d+1) × (2d+1)  augmented Hamiltonians for θ
# ---------------------------------------------------------------------------

def backward_H_lin(sigma: Tensor, beta: Tensor, g: Tensor) -> Tensor:
    """
    Augmented backward system for θ_x, shape (2d+1, 2d+1).

    Tracks the state [p_v; q_v; r_x] where r_x = p_v * θ_x (so θ_x = r_x/p_v).
    ODEs:
        dp_v/dτ = −σ p_v + q_v
        dq_v/dτ =  β p_v + σᵀ q_v
        dr_x/dτ =  gᵀ p_v          ← g in BOTTOM-LEFT block

    Matrix form:
        [[ −σ,   I,   0 ],
         [  β,   σᵀ,  0 ],
         [ gᵀ,   0,   0 ]]

    θ_x recovery: θ_x(τ) = r_x(τ) / p_v(τ)   (scalar ratio for each column).

    g = β_k ν_k  (eq 44).
    """
    d = sigma.shape[0]
    dev, dty = _dev_dty(sigma)
    H = _Z(2*d+1, 2*d+1, dev, dty)
    H[:d,    :d]    = -sigma
    H[:d,    d:2*d] =  _I(d, dev, dty)
    H[d:2*d, :d]    =  beta
    H[d:2*d, d:2*d] =  sigma.T
    H[2*d,   :d]    =  g           # g^T in bottom-left row
    return H


def forward_H_lin(sigma: Tensor, beta: Tensor, g: Tensor) -> Tensor:
    """
    Augmented forward system for θ_x, shape (2d+1, 2d+1).

    Same structure as backward_H_lin but uses H^(+) blocks:
        [[ σ,   I,   0 ],
         [ β,  −σᵀ,  0 ],
         [ gᵀ,  0,   0 ]]

    g = β_k ν_k.
    """
    d = sigma.shape[0]
    dev, dty = _dev_dty(sigma)
    H = _Z(2*d+1, 2*d+1, dev, dty)
    H[:d,    :d]    =  sigma
    H[:d,    d:2*d] =  _I(d, dev, dty)
    H[d:2*d, :d]    =  beta
    H[d:2*d, d:2*d] = -sigma.T
    H[2*d,   :d]    =  g           # g^T in bottom-left row
    return H


# ---------------------------------------------------------------------------
# 4.  Matrix-exponential block extractor
# ---------------------------------------------------------------------------

def phi_blocks(
    H: Tensor, tau: float | Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute Φ(τ) = exp(H τ) and return the four (d × d) corner blocks.

    H must be (2d × 2d); d is inferred as H.shape[0] // 2.

    Returns
    -------
    Phi   : (2d, 2d)   full matrix exponential
    Phi11 : (d, d)     upper-left
    Phi12 : (d, d)     upper-right
    Phi21 : (d, d)     lower-left
    Phi22 : (d, d)     lower-right
    """
    n = H.shape[0]
    assert n % 2 == 0, "H must be square with even dimension"
    d = n // 2
    if not isinstance(tau, Tensor):
        tau_h = H * tau
    else:
        tau_h = H * tau.to(dtype=H.dtype, device=H.device)
    Phi = torch.linalg.matrix_exp(tau_h)
    return Phi, Phi[:d, :d], Phi[:d, d:], Phi[d:, :d], Phi[d:, d:]


# ---------------------------------------------------------------------------
# 5.  Convenience: all six Hamiltonians for one interval
# ---------------------------------------------------------------------------

def all_hamiltonians(
    sigma: Tensor,
    beta:  Tensor,
    nu:    Tensor,
    B_ref: Tensor,
    *,
    branch: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Return (H2d, H_C, H_lin) for the specified branch in one call.

    B_ref  : right-endpoint B^(−)_{k+1} for 'backward',
             left-endpoint  B^(+)_k     for 'forward'.
    """
    g = beta @ nu
    if branch == "backward":
        return (backward_H2d(sigma, beta),
                backward_H_C(sigma, beta, B_ref),
                backward_H_lin(sigma, beta, g))
    elif branch == "forward":
        return (forward_H2d(sigma, beta),
                forward_H_C(sigma, beta, B_ref),
                forward_H_lin(sigma, beta, g))
    else:
        raise ValueError(f"branch must be 'backward' or 'forward', got {branch!r}")
