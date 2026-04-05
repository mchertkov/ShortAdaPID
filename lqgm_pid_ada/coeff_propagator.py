"""
coeff_propagator.py — Single-interval propagation of Green-function coefficients.

For each PWC interval I_k = [t_k, t_{k+1}] with parameters (σ_k, β_k, ν_k):

  backward_interval(sigma, beta, nu, tau, cs_right) → cs_left
      Propagates backward by τ = t_{k+1} − t ∈ [0, Δ_k].
      Full interval: τ = Δ_k.

  forward_interval(sigma, beta, nu, tau, cs_left) → cs_right
      Propagates forward by τ = t − t_k ∈ [0, Δ_k].
      Full interval: τ = Δ_k.

  delta_bc(d, device, dtype, *, eps) → CoeffState
      δ-function boundary condition: A = C = (1/ε)I, B = (1/ε)I,
      θ_x = θ_y = 0, ζ = 0.

Propagation formulas (backward branch, tau = t_{k+1} − t)
----------------------------------------------------------
Using H  = backward_H2d(σ, β),  Φ(τ) = exp(H τ):

  X   = Φ₁₁ + Φ₁₂ A_ref
  Y   = Φ₂₁ + Φ₂₂ A_ref

  A_t = Y X⁻¹                                    (eq 39)
  B_t = X⁻ᵀ B_ref                                (eq 40)
  C_t = C_ref − Γ₂(τ) X⁻¹ B_ref                 (eq 43, Γ₂ from 3d exp)
  θ_x = r_x / p_v,  where [p_v; q_v; r_x] from (2d+1) exp with g=βν   (verified)
  θ_y = θ_y,ref + ∫₀^τ B(s)ᵀ θ_x(s) ds          (eq 22 integrated, GL quadrature)

Special cases: σ_k = 0  (zero-drift)
--------------------------------------
When σ_k = 0, the backward Hamiltonian is −H^(−) = [[0, I],[β, 0]], whose
matrix exponential is built entirely from matrix hyperbolic functions of W=√β.

Two sub-cases are handled analytically (no GL quadrature, no (2d+1) matrix_exp):

Case A — diagonal β_k = diag(b₁,…,b_d):
  The d modes decouple completely.  For mode i (scalar wᵢ = √bᵢ):

    Φ₁₁ᵢ = cosh(wᵢτ),  Φ₁₂ᵢ = sinh(wᵢτ)/wᵢ  [I-blocks × scalar]
    Xᵢ   = cosh(wᵢτ) + sinh(wᵢτ)/wᵢ · [A_ref]ᵢᵢ   (scalar per mode)
    θ_xᵢ = [W⁻¹ sinh(Wτ/2)²  · g]ᵢ / Xᵢ  + θ_x,ref,i / Xᵢ · (full term)
  Analytic θ_y:
    θ_yᵢ = θ_y,ref,i + (gᵢ/wᵢ) tanh(wᵢτ/2)   when θ_x,ref = 0
  See _zero_drift_diag_analytic() for the general formula.

Case B — general SPD β_k:
  Use the eigendecomposition β_k = Q D Qᵀ to reduce to Case A in the
  eigenbasis.  A_ref is rotated into the eigenbasis; if it is not diagonal
  there, off-diagonal entries couple modes for A/B/C but θ_x/θ_y still
  benefit from the same matrix-hyperbolic structure.
  See _zero_drift_spd_analytic() for details.

In both cases all operations are differentiable through torch.autograd.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .core import CoeffState
from .hamiltonian import (
    backward_H2d, forward_H2d,
    backward_H_C, forward_H_C,
    backward_H_lin, forward_H_lin,
    phi_blocks,
)

Tensor = torch.Tensor

# ---------------------------------------------------------------------------
# Gauss-Legendre nodes/weights on [0, 1]  (16-point rule)
# ---------------------------------------------------------------------------
# Plain Python floats — no torch dependency at module load time.
# 16-point rule: integrates polynomials up to degree 31 exactly.
# Replaces the old 4-point rule (degree 7) to reduce θ_y quadrature error
# from ~1e-3 to ~1e-9 for the general σ≠0 path.
# Generated via numpy.polynomial.legendre.leggauss(16), mapped to [0,1].
_GL4_NODES_F = [
    0.00529953250417503074, 0.02771248846338369987,
    0.06718439880608412240, 0.12229779582249850067,
    0.19106187779867811471, 0.27099161117138631516,
    0.35919822461037054229, 0.45249374508118128668,
    0.54750625491881876883, 0.64080177538962945771,
    0.72900838882861362933, 0.80893812220132188529,
    0.87770220417750155484, 0.93281560119391593311,
    0.97228751153661630013, 0.99470046749582496926,
]
_GL4_WEIGHTS_F = [
    0.01357622970587708811, 0.03112676196932372824,
    0.04757925584124630264, 0.06231448562776703559,
    0.07479799440828835411, 0.08457825969750132344,
    0.09130170752246181964, 0.09472530522753432047,
    0.09472530522753432047, 0.09130170752246181964,
    0.08457825969750132344, 0.07479799440828835411,
    0.06231448562776703559, 0.04757925584124630264,
    0.03112676196932372824, 0.01357622970587708811,
]


# ---------------------------------------------------------------------------
# Special case: σ = 0, diagonal β_k  (Case A — fully analytic)
# ---------------------------------------------------------------------------

def _mhyp_diag(
    b: Tensor,      # (d,) positive eigenvalues of β
    tau: float,
    *,
    half: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Scalar hyperbolic functions per mode.

    Returns (cosh_w, sinh_w_over_w, tanh_half) where each is (d,):
      cosh_w       = cosh(wᵢ τ)
      sinh_w_over_w = sinh(wᵢ τ) / wᵢ
      tanh_half    = tanh(wᵢ τ/2)   [if half=True, else unused zeros]
    with wᵢ = √bᵢ.  All are differentiable w.r.t. b.
    """
    w   = torch.sqrt(b.clamp(min=1e-30))   # (d,)  wᵢ = √bᵢ
    wt  = w * tau
    ch  = torch.cosh(wt)                   # cosh(wτ)
    sh  = torch.sinh(wt)                   # sinh(wτ)
    sh_w = sh / w                          # sinh(wτ)/w = Φ₁₂ diagonal entry
    if half:
        th_half = torch.tanh(wt * 0.5)    # tanh(wτ/2)
    else:
        th_half = torch.zeros_like(w)
    return ch, sh_w, th_half


def _zero_drift_diag_analytic(
    beta_diag: Tensor,   # (d,)  diagonal entries of β_k  (all > 0)
    g:         Tensor,   # (d,)  g_k = β_k ν_k
    tau:       float,
    A_ref:     Tensor,   # (d, d)  symmetric — A_{k+1}^(−) or A_k^(+)
    B_ref:     Tensor,   # (d, d)  B_{k+1}^(−) or B_k^(+)
    C_ref:     Tensor,   # (d, d)  C_{k+1}^(−) or C_k^(+)
    tx_ref:    Tensor,   # (d,)    θ_x at reference end
    ty_ref:    Tensor,   # (d,)    θ_y at reference end
    *,
    branch: str,         # 'backward' or 'forward'
) -> CoeffState:
    """Fully analytic propagation for σ=0, diagonal β.

    The d modes decouple.  Φ₁₁ = diag(cosh wᵢτ), Φ₁₂ = diag(sinh wᵢτ/wᵢ),
    Φ₂₁ = diag(wᵢ sinh wᵢτ), Φ₂₂ = diag(cosh wᵢτ).

    For the backward branch the effective Hamiltonian is −H^(−) = [[0,I],[β,0]],
    which is identical in structure to the forward H^(+) = [[0,I],[β,0]].
    Both branches therefore use the same matrix-exponential blocks; the sign
    difference between the branches enters only through which end is the
    reference.

    θ_x analytic formula (valid for any A_ref, tx_ref):
        r_xᵢ(τ) = [Γ̃₁]ᵢ + [Γ̃₂ A_ref]ᵢ + [tx_ref]ᵢ
        Γ̃₁ᵢ = gᵢ · (sinh wᵢτ)/wᵢ - gᵢ/wᵢ² · (1 − cosh wᵢτ)   — see derivation
              = (gᵢ/wᵢ²)(wᵢ sinh wᵢτ + cosh wᵢτ − 1)            (*)
        Γ̃₂ᵢⱼ = gᵢ · (cosh wᵢτ − 1)/wᵢ² · δᵢⱼ                   (diagonal)
        θ_xᵢ  = r_xᵢ / Xᵢ,   Xᵢ = cosh wᵢτ + (sinh wᵢτ/wᵢ)·[A_ref]ᵢᵢ

    θ_y analytic formula (integral of B(s)ᵀ θ_x(s) ds):
        When modes decouple (diagonal A throughout), the integrand per mode i is
            Bᵢ(s) θ_xᵢ(s) = [Xᵢ(s)⁻¹ bᵢ_ref] · [rᵢ(s) / Xᵢ(s)]
        which does NOT simplify to a single tanh unless tx_ref = 0.
        For the generic case we use 4-point GL on [0,τ], but entirely with
        scalar (per-mode) arithmetic — no matrix_exp calls at GL nodes.
    """
    d   = beta_diag.shape[0]
    dev = beta_diag.device
    dty = beta_diag.dtype

    ch, sh_w, _ = _mhyp_diag(beta_diag, tau)       # (d,) each
    w            = torch.sqrt(beta_diag.clamp(min=1e-30))

    # Φ blocks (diagonal matrices stored as (d,) vectors)
    P11 = ch                # diag of Φ₁₁
    P12 = sh_w              # diag of Φ₁₂
    P21 = w * torch.sinh(w * tau)  # diag of Φ₂₁  = w·sinh(wτ)
    P22 = ch                # diag of Φ₂₂

    # A_ref diagonal entries (off-diagonal entries couple rows of X, but
    # since β is diagonal and the Riccati with σ=0 preserves symmetry,
    # A stays symmetric.  If A is also diagonal — which it is when β is
    # diagonal and the BC is ∝ I — the per-mode X is scalar.)
    # We support general symmetric A_ref: X = diag(P11) + diag(P12) @ A_ref
    # X is (d,d) in general.  For truly diagonal A_ref it reduces to (d,).
    I_d  = torch.eye(d, device=dev, dtype=dty)
    X    = P11[:, None] * I_d + P12[:, None] * A_ref   # (d, d)  row-scaled
    Y    = P21[:, None] * I_d + P22[:, None] * A_ref   # (d, d)
    X_inv = torch.linalg.inv(X)

    A_t = Y @ X_inv
    B_t = X_inv.T @ B_ref

    # C_t: use the (3d×3d) matrix_exp path — exact, no singularity issues.
    # The GL alternative has a near-singular integrand at s=0 when A_ref is
    # large (delta BC: A_ref = B_ref = 1/eps).  Near s=0, X(s) ≈ s·A_ref,
    # B(s) = B_ref/X(s) → ∞, so 4-point GL gives O(1) errors.
    # The 3d matrix_exp is cheap (d=1 → 3×3) and exact.
    from .hamiltonian import backward_H_C
    sigma_zero = torch.zeros(d, d, device=dev, dtype=dty)
    H_C   = backward_H_C(sigma_zero, torch.diag(beta_diag), B_ref)
    Phi_C = torch.linalg.matrix_exp(H_C * tau)
    Gamma2 = Phi_C[2*d:, d:2*d]               # (d, d) bottom-middle block
    C_t   = C_ref - Gamma2 @ X_inv @ B_ref

    # θ_x analytic: r_x = Γ̃₁ + Γ̃₂·A_ref_diag + tx_ref, θ_x = X⁻¹ r_x
    # Γ̃₁ᵢ = gᵢ·∫₀^τ Φ₁₁ᵢ(s) ds = gᵢ·sinh(wᵢτ)/wᵢ
    # Γ̃₂ᵢ = gᵢ·∫₀^τ Φ₁₂ᵢ(s) ds = gᵢ·(cosh(wᵢτ)−1)/wᵢ²
    Gtilde1 = g * sh_w                              # (d,) = g·sinh(wτ)/w
    Gtilde2_diag = g * (ch - 1.0) / (w * w)        # (d,) diagonal of Γ̃₂
    # r_x = Γ̃₁ + diag(Γ̃₂) @ A_ref @ ones + tx_ref  (one row per solution col)
    r_x = Gtilde1 + Gtilde2_diag * torch.diag(A_ref) + tx_ref  # (d,) — exact only if A_ref diagonal
    # For non-diagonal A_ref the full expression is Γ̃₁ + Γ̃₂ · A_ref diag entries + (A_ref off-diag terms)
    # We handle this by computing the full (d,d) version:
    r_x_full = (Gtilde1[:, None] * I_d              # outer-broadcast Γ̃₁
                + Gtilde2_diag[:, None] * A_ref      # Γ̃₂ᵢ · [A_ref]ᵢⱼ row-wise
                + tx_ref[:, None] * I_d)             # tx_ref on diagonal
    # r_x_full is (d, d); each column j is the augmented-state-for-solution-j
    # θ_x = X⁻¹ @ r_x_full   → (d, d), then take diagonal (the d solutions share the diagonal)
    tx_mat = X_inv @ r_x_full                        # (d, d)
    tx_t   = torch.diag(tx_mat)                      # (d,) — one scalar θ_x per spatial dim

    # θ_y: exact closed form using already-computed A_t, B_t, C_ref, C_t
    ty_t = _propagate_theta_y_zero_drift_diag(
        beta_diag, g, tau, A_ref, B_ref, A_t, B_t, C_ref, C_t, tx_ref, ty_ref
    )

    return CoeffState(A=A_t, B=B_t, C=C_t, theta_x=tx_t, theta_y=ty_t)




# NOTE: _propagate_C_zero_drift_diag was removed.  GL quadrature for C has a
# near-singular integrand at s=0 when A_ref is large (delta BC: A_ref=1/eps).
# Both zero-drift and scalar-drift cases now use the (3d×3d) matrix_exp path
# for C, which is exact.  The analytic savings come from the (2d+1) theta
# system, not from C.


def _propagate_theta_y_zero_drift_diag(
    beta_diag: Tensor, g: Tensor, tau: float,
    A_ref: Tensor, B_ref: Tensor,
    A_t:   Tensor, B_t:   Tensor, C_ref: Tensor, C_t: Tensor,
    tx_ref: Tensor, ty_ref: Tensor,
) -> Tensor:
    """θ_y for σ=0, diagonal β — exact closed form, no quadrature.

    Derived from ∫_t^{tR} b(s) θ_x(s) ds using the backward ODEs:
        ∫ b·a ds = b(tR) − b(t)  [from db/dt = a·b]
        ∫ b²  ds = c(tR) − c(t)  [from dc/dt = b²]

    Formula (per mode i):
        ty_i(t) = ty_i(tR)
                + (C_ref_i − C_t_i) / B_ref_i · tx_ref_i          [θ_x term]
                + ((B_ref_i − B_t_i)                                [ν term]
                   − A_ref_i · (C_ref_i − C_t_i) / B_ref_i) · ν_i

    where ν_i = g_i / β_i (raw noise), g = β ν.

    Special case tx_ref ≈ 0 (terminal BC): reduces to tanh formula.
    """
    # Extract per-mode diagonal scalars
    A_r = torch.diag(A_ref)   # (d,)
    B_r = torch.diag(B_ref)   # (d,)
    A_l = torch.diag(A_t)     # (d,)
    B_l = torch.diag(B_t)     # (d,)
    C_r = torch.diag(C_ref)   # (d,)
    C_l = torch.diag(C_t)     # (d,)
    nu  = g / beta_diag.clamp(min=1e-30)   # ν = g/β per mode

    dC = C_r - C_l                         # C(tR) - C(t)
    dB = B_r - B_l                         # B(tR) - B(t)
    inv_Br = 1.0 / B_r.clamp(min=1e-30)

    ty_t = (ty_ref
            + (dC * inv_Br) * tx_ref           # θ_x contribution
            + (dB - A_r * dC * inv_Br) * nu)   # ν contribution
    return ty_t


# ---------------------------------------------------------------------------
# Special case: σ = 0, general SPD β_k  (Case B — via eigendecomposition)
# ---------------------------------------------------------------------------

def _zero_drift_spd_analytic(
    beta:   Tensor,   # (d, d) symmetric positive definite
    g:      Tensor,   # (d,)
    tau:    float,
    A_ref:  Tensor,
    B_ref:  Tensor,
    C_ref:  Tensor,
    tx_ref: Tensor,
    ty_ref: Tensor,
    *,
    branch: str,      # "backward" or "forward"
) -> CoeffState:
    """Propagation for σ=0, general SPD β using the full matrix-exp path.

    When β is SPD but not diagonal, the eigenbases of adjacent intervals
    may differ, so the rotated coefficients are not diagonal in any single
    eigenbasis.  Rather than attempting per-mode formulas that require
    diagonal structure, we use the same general matrix-exp primitives
    as the σ≠0 case (with σ set to zero).  For d=2 the matrices are tiny
    (4×4, 6×6, 5×5) and the overhead is negligible.
    """
    d   = beta.shape[0]
    dev = beta.device
    dty = beta.dtype
    sigma = torch.zeros(d, d, device=dev, dtype=dty)

    if branch == 'backward':
        H2d   = backward_H2d(sigma, beta)
        H_lin = backward_H_lin(sigma, beta, g)
    else:
        H2d   = forward_H2d(sigma, beta)
        H_lin = forward_H_lin(sigma, beta, g)

    A_t, B_t, X, _ = _propagate_AB(H2d, tau, A_ref, B_ref)
    C_t             = _propagate_C(sigma, beta, tau, C_ref, B_ref, X,
                                   branch=branch)
    tx_t            = _propagate_theta_x(H_lin, tau, tx_ref, A_ref, X)
    ty_t            = _propagate_theta_y(H2d, H_lin, tau,
                                         A_ref, B_ref, tx_ref, ty_ref)
    return CoeffState(A=A_t, B=B_t, C=C_t, theta_x=tx_t, theta_y=ty_t)


def _is_zero(sigma: Tensor, tol: float = 1e-12) -> bool:
    """True if σ is numerically zero."""
    return sigma.abs().max().item() < tol


def _is_diagonal(M: Tensor, tol: float = 1e-12) -> bool:
    """True if M is numerically diagonal."""
    d = M.shape[0]
    if d == 1:
        return True
    off = M - torch.diag(torch.diag(M))
    return off.abs().max().item() < tol


def _is_scalar_drift(sigma: Tensor, tol: float = 1e-12):
    """Return (True, c) if σ = c·I, else (False, None).

    Checks that σ is a scalar multiple of the identity by comparing
    σ − (σ₁₁)·I to zero.  Works for d=1 trivially.
    """
    d = sigma.shape[0]
    c = sigma[0, 0]
    off = sigma - c * torch.eye(d, device=sigma.device, dtype=sigma.dtype)
    if off.abs().max().item() < tol:
        return True, c
    return False, None


# ---------------------------------------------------------------------------
# Special case: σ = c·I (scalar drift), diagonal β  (Case C — fully analytic)
# ---------------------------------------------------------------------------

def _scalar_drift_phi_diag(
    beta_diag: Tensor,   # (d,) diagonal entries of β
    c:         float,    # scalar from σ = cI   (Python float or 0-d tensor)
    tau:       float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-mode transition blocks for σ=cI, diagonal β.

    Effective frequency:  wᵢ = √(bᵢ + c²)

    Returns (P11, P12, P22, w) each of shape (d,):
      P11ᵢ = cosh(wᵢτ) − (c/wᵢ) sinh(wᵢτ)
      P12ᵢ = sinh(wᵢτ) / wᵢ
      P22ᵢ = cosh(wᵢτ) + (c/wᵢ) sinh(wᵢτ)
      wᵢ   = √(bᵢ + c²)

    P21ᵢ = bᵢ·P12ᵢ  (not returned separately; compute on demand).
    """
    if not isinstance(c, Tensor):
        c_val = float(c)
    else:
        c_val = c.item()
    w      = torch.sqrt((beta_diag + c_val ** 2).clamp(min=1e-30))  # (d,)
    wt     = w * tau
    ch     = torch.cosh(wt)
    sh     = torch.sinh(wt)
    sh_w   = sh / w                                    # sinh(wτ)/w
    c_sh_w = c_val * sh_w                              # c·sinh(wτ)/w
    P11    = ch - c_sh_w
    P12    = sh_w
    P22    = ch + c_sh_w
    return P11, P12, P22, w


def _scalar_drift_diag_analytic(
    beta_diag: Tensor,   # (d,)
    c:         float,    # scalar from σ = cI
    g:         Tensor,   # (d,)  g_k = β_k ν_k
    tau:       float,
    A_ref:     Tensor,   # (d, d)
    B_ref:     Tensor,   # (d, d)
    C_ref:     Tensor,   # (d, d)
    tx_ref:    Tensor,   # (d,)
    ty_ref:    Tensor,   # (d,)
) -> CoeffState:
    """Analytic propagation for σ=cI, diagonal β.

    Shifts the effective frequency wᵢ = √(bᵢ+c²) and reuses the same
    scalar-per-mode structure as _zero_drift_diag_analytic.

    θ_x augmented drive integrals (eq Gtilde_scalar_drift):
        Γ̃₁ᵢ = gᵢ (sinh(wᵢτ)/wᵢ − c(cosh(wᵢτ)−1)/wᵢ²)
        Γ̃₂ᵢ = gᵢ (cosh(wᵢτ)−1)/wᵢ²                          [same as zero-drift]

    X = diag(P11) + diag(P12) @ A_ref   (d×d, full matrix)
    θ_x = X⁻¹ @ r_x,   r_x = Γ̃₁ + A_refᵀ @ (g ⊙ Γ̃₂_vec) + tx_ref

    θ_y exact (tx_ref = 0):
        θ_yᵢ = ty_ref,i + (gᵢ/wᵢ) tanh(wᵢτ/2)    (eq thetay_scalar_drift)
    θ_y general: scalar GL per mode (no matrix_exp).
    """
    d   = beta_diag.shape[0]
    dev = beta_diag.device
    dty = beta_diag.dtype
    I_d = torch.eye(d, device=dev, dtype=dty)

    P11, P12, P22, w = _scalar_drift_phi_diag(beta_diag, c, tau)
    P21 = beta_diag * P12                               # bᵢ · sinh(wᵢτ)/wᵢ

    # X, Y, A_t, B_t  (full d×d because A_ref may not be diagonal)
    X     = P11[:, None] * I_d + P12[:, None] * A_ref  # (d, d)
    Y     = P21[:, None] * I_d + P22[:, None] * A_ref  # (d, d)
    X_inv = torch.linalg.inv(X)
    A_t   = Y @ X_inv
    B_t   = X_inv.T @ B_ref

    # C_t: use the (3d×3d) matrix_exp path — same reason as zero-drift case.
    # GL has near-singular integrand at s=0 when A_ref is large (delta BC).
    from .hamiltonian import backward_H_C
    sigma_cI  = (float(c) * torch.eye(d, device=dev, dtype=dty)
                 if not isinstance(c, Tensor)
                 else c * torch.eye(d, device=dev, dtype=dty))
    H_C_sd    = backward_H_C(sigma_cI, torch.diag(beta_diag), B_ref)
    Phi_C_sd  = torch.linalg.matrix_exp(H_C_sd * tau)
    Gamma2_sd = Phi_C_sd[2*d:, d:2*d]
    C_t = C_ref - Gamma2_sd @ X_inv @ B_ref

    # θ_x analytic:
    #   Γ̃₁ᵢ = gᵢ (P12ᵢ − c·Φ̃₁₂ᵢ)   where Φ̃₁₂ᵢ = (cosh(wᵢτ)−1)/wᵢ²
    ch     = torch.cosh(w * tau)
    Phi12  = (ch - 1.0) / (w * w)                      # ∫₀^τ P12(s) ds per mode
    Gtilde1 = g * (P12 - c * Phi12)                     # (d,)  Γ̃₁
    Gtilde2 = g * Phi12                                  # (d,)  diagonal of Γ̃₂

    # r_x_vec_j = tx_ref_j + Γ̃₁_j + Σᵢ gᵢ Φ̃₁₂ᵢ [A_ref]_{ij}
    #           = tx_ref_j + Γ̃₁_j + [A_refᵀ @ (g ⊙ Φ̃₁₂)]_j
    r_x = tx_ref + Gtilde1 + A_ref.T @ Gtilde2         # (d,)
    tx_t = X_inv @ r_x                                  # (d,)

    # θ_y: exact closed form (same ODE identity as zero-drift case)
    ty_t = _propagate_theta_y_zero_drift_diag(
        beta_diag, g, tau, A_ref, B_ref, A_t, B_t, C_ref, C_t, tx_ref, ty_ref
    )

    return CoeffState(A=A_t, B=B_t, C=C_t, theta_x=tx_t, theta_y=ty_t)


def _propagate_theta_y_scalar_drift_diag(
    *args, **kwargs
) -> Tensor:
    """Deprecated — use _propagate_theta_y_zero_drift_diag (closed-form)."""
    raise RuntimeError("_propagate_theta_y_scalar_drift_diag should not be called")


# ---------------------------------------------------------------------------
# Special case: σ = c·I, general SPD β_k  (Case C-SPD — via eigendecomp)
# ---------------------------------------------------------------------------

def _scalar_drift_spd_analytic(
    beta:   Tensor,   # (d, d) SPD
    c:      float,    # scalar from σ = cI
    g:      Tensor,   # (d,)
    tau:    float,
    A_ref:  Tensor,
    B_ref:  Tensor,
    C_ref:  Tensor,
    tx_ref: Tensor,
    ty_ref: Tensor,
    *,
    branch: str = 'backward',   # 'backward' or 'forward'
) -> CoeffState:
    """Propagation for σ=cI, general SPD β using the full matrix-exp path.

    Although σ = cI commutes with Q (eigenvectors of β), the rotated
    reference coefficients A_ref, B_ref may not be diagonal when adjacent
    intervals have different β eigenbases.  We use the general matrix-exp
    primitives for robustness (cost is negligible for small d).
    """
    d   = beta.shape[0]
    dev = beta.device
    dty = beta.dtype
    c_val = float(c) if not isinstance(c, Tensor) else c.item()
    sigma = c_val * torch.eye(d, device=dev, dtype=dty)

    if branch == 'backward':
        H2d   = backward_H2d(sigma, beta)
        H_lin = backward_H_lin(sigma, beta, g)
    else:
        H2d   = forward_H2d(sigma, beta)
        H_lin = forward_H_lin(sigma, beta, g)

    A_t, B_t, X, _ = _propagate_AB(H2d, tau, A_ref, B_ref)
    C_t             = _propagate_C(sigma, beta, tau, C_ref, B_ref, X,
                                   branch=branch)
    tx_t            = _propagate_theta_x(H_lin, tau, tx_ref, A_ref, X)
    ty_t            = _propagate_theta_y(H2d, H_lin, tau,
                                         A_ref, B_ref, tx_ref, ty_ref)
    return CoeffState(A=A_t, B=B_t, C=C_t, theta_x=tx_t, theta_y=ty_t)


# ---------------------------------------------------------------------------
# Core propagation primitives
# ---------------------------------------------------------------------------

def _propagate_AB(
    H2d: Tensor, tau: float,
    A_ref: Tensor, B_ref: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Propagate A and B by tau using H2d.

    Returns (A_t, B_t, X, Phi) where X = Phi11 + Phi12 @ A_ref.
    X is passed downstream to _propagate_C and _propagate_theta_x.
    """
    Phi, P11, P12, P21, P22 = phi_blocks(H2d, tau)
    X = P11 + P12 @ A_ref                    # (d, d)
    Y = P21 + P22 @ A_ref                    # (d, d)
    X_inv  = torch.linalg.inv(X)
    A_t    = Y @ X_inv                       # eq 39 / 53
    B_t    = X_inv.T @ B_ref                 # eq 40 / 54  (X^{-T} B_ref)
    return A_t, B_t, X, Phi


def _propagate_C(
    sigma: Tensor,
    beta:  Tensor,
    tau:   float,
    C_ref: Tensor,
    B_ref: Tensor,
    X:     Tensor,
    *,
    branch: str = 'backward',
) -> Tensor:
    """Propagate C by tau using a factored 3d×3d matrix exponential.

    The ODE is Ċ = BᵀB.  The standard augmented 3d×3d Hamiltonian H_C has
    B_ref^T as a matrix entry, making exp(H_C τ) ill-conditioned when
    B_ref = O(1/ε) near δ-BC boundaries.

    Key identity
    ------------
    The block Γ₂ of exp(H_C τ) is LINEAR in B_ref^T:

        Γ₂ = B_ref^T  Γ̃₂

    where Γ̃₂ comes from exp(H̃_C τ) with I replacing B_ref^T.

    Proof: B_ref^T appears only in row (3,2) of H_C.  In the power series
    exp(H_C τ) = I + H_C τ + ..., every term contributing to the (3,*) blocks
    picks up exactly one factor of B_ref^T from the (3,2) entry.

    The factored Hamiltonian H̃_C has entries of O(max(‖σ‖, ‖β‖, 1)),
    independent of ‖B_ref‖.  This makes the computation numerically stable
    for any BC_EPS, any σ, any d.

    Formula: C_t = C_ref − B_ref^T  Γ̃₂  X⁻¹  B_ref
    """
    d   = C_ref.shape[0]
    dev = C_ref.device
    dty = C_ref.dtype
    I_d = torch.eye(d, device=dev, dtype=dty)

    # Build factored H̃_C: same as H_C but with I replacing B_ref^T
    H_tilde = torch.zeros(3*d, 3*d, device=dev, dtype=dty)
    if branch == 'backward':
        H_tilde[:d,    :d]    = -sigma
        H_tilde[:d,    d:2*d] =  I_d
        H_tilde[d:2*d, :d]    =  beta
        H_tilde[d:2*d, d:2*d] =  sigma.T
    else:  # forward
        H_tilde[:d,    :d]    =  sigma
        H_tilde[:d,    d:2*d] =  I_d
        H_tilde[d:2*d, :d]    =  beta
        H_tilde[d:2*d, d:2*d] = -sigma.T
    H_tilde[2*d:, d:2*d] = I_d   # I instead of B_ref^T

    Phi_tilde = torch.linalg.matrix_exp(H_tilde * tau)
    Gamma2_tilde = Phi_tilde[2*d:, d:2*d]   # (d, d)

    X_inv = torch.linalg.inv(X)
    C_t   = C_ref - B_ref.T @ Gamma2_tilde @ X_inv @ B_ref
    return C_t


def _propagate_theta_x(
    H_lin: Tensor, tau: float,
    tx_ref: Tensor, A_ref: Tensor, X: Tensor,
) -> Tensor:
    """Propagate θ_x by tau using the augmented H_lin matrix.

    State: [p_v (d); q_v (d); r_x (1)] initial = [col of I; col of A_ref; tx_ref component]
    Then θ_x_t = last_component / first_component.

    For vector θ_x (d,), we run d independent (2d+1) systems in one batched call
    by building the (2d+1) × (d+1) initial condition matrix.

    Initial condition columns [I; A_ref; tx_ref_j] for j=0,...,d-1
    stacked as a (2d+1, d) matrix:

        init[:d, :] = I_d        (p_v starts as identity)
        init[d:2d,:] = A_ref     (q_v starts at A_ref)
        init[2d,  :] = tx_ref    (r_x starts at θ_x_ref, one entry per column)

    Φ_lin @ init → (2d+1, d); last row = r_x(τ), first d rows = p_v(τ).
    θ_x_t = X⁻¹ @ r_x(τ)   [since p_v(τ) = X after propagation].
    """
    d = A_ref.shape[0]
    Phi_lin = torch.linalg.matrix_exp(H_lin * tau)  # (2d+1, 2d+1)

    # Build initial state: (2d+1, d)
    dev, dty = A_ref.device, A_ref.dtype
    init = torch.zeros(2*d+1, d, device=dev, dtype=dty)
    init[:d, :]   = torch.eye(d, device=dev, dtype=dty)   # p_v(0) = I
    init[d:2*d,:] = A_ref                                  # q_v(0) = A_ref
    init[2*d, :]  = tx_ref                                 # r_x(0) = tx_ref

    state = Phi_lin @ init                  # (2d+1, d)
    r_x   = state[2*d, :]                  # (d,)  — last row
    # p_v(τ) columns = X (verified: same as from phi_blocks)
    X_inv = torch.linalg.inv(X)
    return X_inv @ r_x                     # (d,)  θ_x(τ)


# Threshold above which A_ref is treated as a delta-BC boundary condition,
# triggering the adaptive split in _propagate_theta_y.
# 1/eps_default ≈ 1000; anything > 10 safely catches delta-BC usage.
_DELTA_BC_THRESH = 10.0
# Split point: integrate separately over [0, factor/||A_ref||] to resolve
# the boundary layer.  factor=10 gives ~10 characteristic lengths of headroom.
_SPLIT_FACTOR    = 10.0


def _gl_ty_piece(
    H2d:    Tensor,
    H_lin:  Tensor,
    s_lo:   float,
    s_hi:   float,
    A_ref:  Tensor,
    B_ref:  Tensor,
    tx_ref: Tensor,
) -> Tensor:
    """GL-16 integral of B(s)^T theta_x(s) on [s_lo, s_hi].

    Evaluates both B(s) = X(s)^{-T} B_ref and theta_x(s) at each GL node
    from scratch (propagation from s=0) so no state needs to be threaded.
    """
    length   = s_hi - s_lo
    integral = torch.zeros(A_ref.shape[0], dtype=A_ref.dtype, device=A_ref.device)
    for node_f, weight_f in zip(_GL4_NODES_F, _GL4_WEIGHTS_F):
        s   = s_lo + node_f * length
        w_j = weight_f * length
        _, P11_j, P12_j, _, _ = phi_blocks(H2d, s)
        X_j     = P11_j + P12_j @ A_ref
        X_inv_j = torch.linalg.inv(X_j)
        B_j     = X_inv_j.T @ B_ref
        tx_j    = _propagate_theta_x(H_lin, s, tx_ref, A_ref, X_j)
        integral = integral + w_j * (B_j.T @ tx_j)
    return integral


def _propagate_theta_y(
    H2d:   Tensor,
    H_lin: Tensor,
    tau:   float,
    A_ref: Tensor,
    B_ref: Tensor,
    tx_ref: Tensor,
    ty_ref: Tensor,
) -> Tensor:
    """Propagate theta_y by tau via adaptive GL-16 quadrature.

    theta_y(tau) = theta_y_ref + integral_0^tau B(s)^T theta_x(s) ds

    Adaptive split
    --------------
    When A_ref is large (delta-BC regime, ||A_ref||_inf > _DELTA_BC_THRESH),
    the integrand has a boundary layer of width ~1/||A_ref|| near s=0.
    A single GL-16 on [0, tau] misses this layer.  We split:

        integral_0^tau = integral_0^tau_split + integral_tau_split^tau

    with tau_split = _SPLIT_FACTOR / ||A_ref||_inf, applying GL-16 to each piece
    independently.  Both sub-intervals have smooth integrands.
    Total cost: 2 x 16 = 32 evaluations (up from 16).
    """
    a_max     = float(A_ref.abs().max().item())
    tau_split = None
    if a_max > _DELTA_BC_THRESH:
        split_candidate = _SPLIT_FACTOR / a_max
        if split_candidate < tau * 0.9:
            tau_split = split_candidate

    if tau_split is not None:
        integral = (
            _gl_ty_piece(H2d, H_lin, 0.0,      tau_split, A_ref, B_ref, tx_ref) +
            _gl_ty_piece(H2d, H_lin, tau_split, tau,       A_ref, B_ref, tx_ref)
        )
    else:
        integral = _gl_ty_piece(H2d, H_lin, 0.0, tau, A_ref, B_ref, tx_ref)

    return ty_ref + integral


# ---------------------------------------------------------------------------
# Public: single-interval propagation
# ---------------------------------------------------------------------------

def backward_interval(
    sigma:  Tensor,   # (d, d)
    beta:   Tensor,   # (d, d)
    nu:     Tensor,   # (d,)
    tau:    float,    # elapsed backward time (= Δ_k for full interval)
    cs_right: CoeffState,
) -> CoeffState:
    """Propagate Green-function coefficients backward by tau.

    Parameters
    ----------
    sigma, beta, nu : interval parameters
    tau             : t_{k+1} − t  (positive; use Δ_k for full interval)
    cs_right        : CoeffState at t_{k+1}

    Returns
    -------
    CoeffState at t = t_{k+1} − tau

    Dispatch
    --------
    σ=0, diagonal β   → _zero_drift_diag_analytic   (fully analytic, no matrix_exp)
    σ=0, general SPD β → _zero_drift_spd_analytic   (eigenbasis, no (2d+1) matrix_exp)
    general            → general matrix_exp path
    """
    g = beta @ nu

    if _is_zero(sigma):
        if _is_diagonal(beta):
            return _zero_drift_diag_analytic(
                torch.diag(beta), g, tau,
                cs_right.A, cs_right.B, cs_right.C,
                cs_right.theta_x, cs_right.theta_y,
                branch='backward',
            )
        else:
            return _zero_drift_spd_analytic(
                beta, g, tau,
                cs_right.A, cs_right.B, cs_right.C,
                cs_right.theta_x, cs_right.theta_y,
                branch='backward',
            )

    is_scalar, c_val = _is_scalar_drift(sigma)
    if is_scalar:
        if _is_diagonal(beta):
            return _scalar_drift_diag_analytic(
                torch.diag(beta), c_val, g, tau,
                cs_right.A, cs_right.B, cs_right.C,
                cs_right.theta_x, cs_right.theta_y,
            )
        else:
            return _scalar_drift_spd_analytic(
                beta, c_val, g, tau,
                cs_right.A, cs_right.B, cs_right.C,
                cs_right.theta_x, cs_right.theta_y,
                branch='backward',
            )

    H2d   = backward_H2d(sigma, beta)
    H_lin = backward_H_lin(sigma, beta, g)

    A_t, B_t, X, _ = _propagate_AB(H2d, tau, cs_right.A, cs_right.B)
    C_t             = _propagate_C(sigma, beta, tau,
                                   cs_right.C, cs_right.B, X,
                                   branch='backward')
    tx_t            = _propagate_theta_x(H_lin, tau, cs_right.theta_x,
                                         cs_right.A, X)
    ty_t            = _propagate_theta_y(H2d, H_lin, tau,
                                         cs_right.A, cs_right.B,
                                         cs_right.theta_x, cs_right.theta_y)
    return CoeffState(A=A_t, B=B_t, C=C_t, theta_x=tx_t, theta_y=ty_t)


def forward_interval(
    sigma:  Tensor,
    beta:   Tensor,
    nu:     Tensor,
    tau:    float,
    cs_left: CoeffState,
) -> CoeffState:
    """Propagate Green-function coefficients forward by tau.

    Parameters
    ----------
    tau : t − t_k  (positive; use Δ_k for full interval)
    cs_left : CoeffState at t_k

    Dispatch
    --------
    σ=0, diagonal β   → _zero_drift_diag_analytic   (fully analytic, no matrix_exp)
    σ=0, general SPD β → _zero_drift_spd_analytic   (eigenbasis, no (2d+1) matrix_exp)
    general            → general matrix_exp path
    """
    g = beta @ nu

    if _is_zero(sigma):
        if _is_diagonal(beta):
            return _zero_drift_diag_analytic(
                torch.diag(beta), g, tau,
                cs_left.A, cs_left.B, cs_left.C,
                cs_left.theta_x, cs_left.theta_y,
                branch='forward',
            )
        else:
            return _zero_drift_spd_analytic(
                beta, g, tau,
                cs_left.A, cs_left.B, cs_left.C,
                cs_left.theta_x, cs_left.theta_y,
                branch='forward',
            )

    is_scalar, c_val = _is_scalar_drift(sigma)
    if is_scalar:
        if _is_diagonal(beta):
            return _scalar_drift_diag_analytic(
                torch.diag(beta), c_val, g, tau,
                cs_left.A, cs_left.B, cs_left.C,
                cs_left.theta_x, cs_left.theta_y,
            )
        else:
            return _scalar_drift_spd_analytic(
                beta, c_val, g, tau,
                cs_left.A, cs_left.B, cs_left.C,
                cs_left.theta_x, cs_left.theta_y,
                branch='forward',
            )

    H2d   = forward_H2d(sigma, beta)
    H_lin = forward_H_lin(sigma, beta, g)

    A_t, B_t, X, _ = _propagate_AB(H2d, tau, cs_left.A, cs_left.B)
    C_t             = _propagate_C(sigma, beta, tau,
                                   cs_left.C, cs_left.B, X,
                                   branch='forward')
    tx_t            = _propagate_theta_x(H_lin, tau, cs_left.theta_x,
                                         cs_left.A, X)
    ty_t            = _propagate_theta_y(H2d, H_lin, tau,
                                         cs_left.A, cs_left.B,
                                         cs_left.theta_x, cs_left.theta_y)
    return CoeffState(A=A_t, B=B_t, C=C_t, theta_x=tx_t, theta_y=ty_t)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def delta_bc(
    d: int,
    device: torch.device,
    dtype:  torch.dtype,
    *,
    eps: float = 1e-6,
) -> CoeffState:
    """δ-function boundary condition: G(x|y) → δ(x−y) as t → endpoint.

    Approximated as a tight Gaussian with precision 1/eps:
        A = C = (1/ε) I,  B = (1/ε) I,  θ_x = θ_y = 0,  ζ = 0.

    eps should be smaller than the smallest interval Δ_k to avoid
    contaminating the propagation, but not so small as to cause overflow.
    Typical values: 1e-5 … 1e-7.
    """
    scale = torch.tensor(1.0 / eps, device=device, dtype=dtype)
    I     = torch.eye(d, device=device, dtype=dtype)
    z     = torch.zeros(d, device=device, dtype=dtype)
    return CoeffState(
        A       = scale * I,
        B       = scale * I,
        C       = scale * I,
        theta_x = z,
        theta_y = z,
        zeta    = None,
    )
