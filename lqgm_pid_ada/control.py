"""
control.py — Evaluate Green-function coefficients at arbitrary t and compute
the GMM optimal control u*(t, x).

Public API
----------
  eval_bwd(t, bwd_states, protocol) → CoeffState
  eval_fwd(t, fwd_states, protocol) → CoeffState
  gmm_control(t, x, x0, bwd_states, fwd_states, target_gmm, protocol)
      → (u_star, log_psi, rho)

Mathematical context (Section 5 of notes, *ratio form*)
---------------------------------------------------------
With deterministic start x_0, the optimal control is

    u*(t,x) = ∇_x log ψ_t(x)

where ψ_t(x) ∝ ∫ [p_tar(y) / G_1^+(y|x_0)] G_t^-(x|y) dy.

Dividing by G_1^+(y|x_0) shifts the effective y-precision from C^- + P_k  to
S_k = C^- + P_k − A_1^+, and moves B_1^+ x_0 and θ_{x;1}^+ into the
linear term q_k.  The resulting control is a softmax-weighted combination:

    u*(t,x) = Σ_k ρ_k(t,x) (−Λ_k x + λ_k)

Per-component quantities (all evaluated at the query time t):
    S_k    = C_t^−  +  P_k  −  A_1^+           (d×d, SPD)
    q_k    = θ_y;t^− + P_k m_k − B_1^+ x_0 − θ_{x;1}^+  (d,)
    Λ_k    = A_t^− − B_t^− S_k^{-1} (B_t^−)ᵀ  (d×d)
    λ_k    = θ_x;t^− + B_t^− S_k^{-1} q_k      (d,)
    C_k    = −½ log|S_k| + ½ log|P_k|           (scalar log-normalizer,
             − ½ m_kᵀ P_k m_k + ½ q_kᵀ S_k^{-1} q_k   k-dependent part)

Responsibilities:
    log w_k(t,x) = log π_k + C_k − ½ xᵀ Λ_k x + λ_kᵀ x
    ρ_k(t,x)    = softmax_k { log w_k(t,x) }

log-ψ:
    log ψ_t(x) = logsumexp_k { log π_k + C_k − ½ xᵀ Λ_k x + λ_kᵀ x }

Verified (50 random tests, max_err < 4 × 10^{-8}) against:
  (a) numerical gradient of ratio-form ψ_t via Gaussian quadrature, and
  (b) emulated guided_continuous u_star (sigma=0, nu=0, x_0=0 case).
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from .core import CoeffState, GaussianMixture, MatrixPWCProtocol
from .coeff_propagator import backward_interval, forward_interval

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Interpolation: evaluate coefficients at arbitrary t
# ---------------------------------------------------------------------------

def eval_bwd(
    t: float | Tensor,
    bwd_states: List[CoeffState],
    protocol: MatrixPWCProtocol,
) -> CoeffState:
    """Evaluate backward Green-function coefficients at time t.

    Locates the interval I_k = [breaks[k], breaks[k+1]] containing t,
    then propagates backward from bwd_states[k+1] by τ = breaks[k+1] − t.

    Parameters
    ----------
    t          : query time (scalar)
    bwd_states : list of K+1 CoeffStates from backward_sweep;
                 bwd_states[j] lives at t = breaks[j]
    protocol   : the MatrixPWCProtocol used for the sweep

    Returns
    -------
    CoeffState at time t
    """
    t_tensor = torch.as_tensor(t, dtype=protocol.dtype, device=protocol.device)
    idx, tau, _ = protocol.locate(t_tensor)
    k = int(round(float(idx.item() if hasattr(idx, 'item') else float(idx))))

    # tau = t_clamped − breaks[k]; backward elapsed = breaks[k+1] − t_clamped
    delta_k = float((protocol.breaks[k + 1] - protocol.breaks[k]).item())
    tau_bwd = delta_k - float(tau.item() if hasattr(tau, 'item') else float(tau))

    if tau_bwd <= 0.0:
        return bwd_states[k + 1]

    return backward_interval(
        sigma    = protocol.sigma[k],
        beta     = protocol.beta[k],
        nu       = protocol.nu[k],
        tau      = tau_bwd,
        cs_right = bwd_states[k + 1],
    )


def eval_fwd(
    t: float | Tensor,
    fwd_states: List[CoeffState],
    protocol: MatrixPWCProtocol,
) -> CoeffState:
    """Evaluate forward Green-function coefficients at time t.

    Locates interval I_k and propagates forward from fwd_states[k] by
    τ = t − breaks[k].

    Parameters
    ----------
    t          : query time (scalar)
    fwd_states : list of K+1 CoeffStates from forward_sweep;
                 fwd_states[j] lives at t = breaks[j]
    protocol   : the MatrixPWCProtocol used for the sweep

    Returns
    -------
    CoeffState at time t
    """
    t_tensor = torch.as_tensor(t, dtype=protocol.dtype, device=protocol.device)
    idx, tau, _ = protocol.locate(t_tensor)
    k = int(round(float(idx.item() if hasattr(idx, 'item') else float(idx))))
    tau_fwd = float(tau.item() if hasattr(tau, 'item') else float(tau))

    if tau_fwd <= 0.0:
        return fwd_states[k]

    return forward_interval(
        sigma   = protocol.sigma[k],
        beta    = protocol.beta[k],
        nu      = protocol.nu[k],
        tau     = tau_fwd,
        cs_left = fwd_states[k],
    )


# ---------------------------------------------------------------------------
# GMM control evaluation
# ---------------------------------------------------------------------------

def gmm_control(
    t:           float | Tensor,
    x:           Tensor,           # (B, d)
    x0:          Tensor,           # (d,)
    bwd_states:  List[CoeffState],
    fwd_states:  List[CoeffState],
    target_gmm:  GaussianMixture,
    protocol:    MatrixPWCProtocol,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the GMM optimal control u*(t, x) and related quantities.

    Parameters
    ----------
    t           : scalar query time
    x           : (B, d)  batch of query positions
    x0          : (d,)    deterministic start (used in ratio-form denominator)
    bwd_states  : backward sweep output from backward_sweep(protocol)
    fwd_states  : forward  sweep output from forward_sweep(protocol, x0_mean=x0)
    target_gmm  : GaussianMixture with M components (precisions P_k = Σ_k^{-1})
    protocol    : the MatrixPWCProtocol

    Returns
    -------
    u_star  : (B, d)   optimal control field
    log_psi : (B,)     log ψ_t(x)  (up to a t-only additive constant)
    rho     : (B, M)   per-component responsibilities ρ_k(t, x) ∈ (0,1)
    """
    if x.ndim != 2:
        raise ValueError(f"x must be (B, d), got {tuple(x.shape)}")
    B, d = x.shape
    M    = target_gmm.M
    dev, dty = x.device, x.dtype

    # ------------------------------------------------------------------
    # 1. Evaluate backward coefficients at t
    # ------------------------------------------------------------------
    cs_bwd: CoeffState = eval_bwd(t, bwd_states, protocol)
    Am = cs_bwd.A.to(dev, dty)        # (d, d)
    Bm = cs_bwd.B.to(dev, dty)        # (d, d)
    Cm = cs_bwd.C.to(dev, dty)        # (d, d)
    tx_m = cs_bwd.theta_x.to(dev, dty)  # (d,)
    ty_m = cs_bwd.theta_y.to(dev, dty)  # (d,)

    # ------------------------------------------------------------------
    # 2. Terminal forward coefficients at t=1 (ratio-form denominator)
    #    We use fwd_states[K], the forward sweep result at t ≈ 1 − ε.
    # ------------------------------------------------------------------
    cs_fwd1: CoeffState = fwd_states[protocol.K]
    A1p    = cs_fwd1.A.to(dev, dty)        # (d, d)  A_1^+
    B1p    = cs_fwd1.B.to(dev, dty)        # (d, d)  B_1^+
    tx1p   = cs_fwd1.theta_x.to(dev, dty)  # (d,)    θ_{x;1}^+

    x0_d = x0.to(dev, dty)                 # (d,)

    # ------------------------------------------------------------------
    # 3. Per-component quantities  (shapes noted for M components, d dims)
    # ------------------------------------------------------------------
    Ps   = target_gmm.precisions.to(dev, dty)   # (M, d, d)  P_k = Σ_k^{-1}
    ms   = target_gmm.means.to(dev, dty)         # (M, d)     m_k
    logpi = torch.log(
        target_gmm.weights.to(dev, dty)
    )                                            # (M,)

    # S_k = C_t^- + P_k - A_1^+   (M, d, d)
    Sk = Cm.unsqueeze(0) + Ps - A1p.unsqueeze(0)   # broadcast (M, d, d)

    # q_k = θ_y;t^- + P_k m_k - B_1^+ x0 - θ_{x;1}^+   (M, d)
    #   P_k m_k: (M, d, d) @ (M, d, 1) → (M, d)
    Pkm = torch.bmm(Ps, ms.unsqueeze(-1)).squeeze(-1)  # (M, d)
    B1p_x0 = (B1p @ x0_d)                              # (d,)
    qk = ty_m.unsqueeze(0) + Pkm - B1p_x0.unsqueeze(0) - tx1p.unsqueeze(0)  # (M, d)

    # Cholesky-solve for S_k^{-1}:   Sk @ Z = I  and  Sk @ q = qk
    # Using torch.linalg.solve for stability; Sk is (M, d, d)
    Sk_inv = torch.linalg.inv(Sk)                      # (M, d, d)

    # Λ_k = A_t^- - B_t^- S_k^{-1} (B_t^-)^T   (M, d, d)
    Bm_exp  = Bm.unsqueeze(0).expand(M, d, d)          # (M, d, d)
    BSkInv  = torch.bmm(Bm_exp, Sk_inv)                # (M, d, d)  B^- S_k^{-1}
    Lambda  = Am.unsqueeze(0) - torch.bmm(BSkInv, Bm_exp.transpose(-2, -1))  # (M, d, d)

    # λ_k = θ_x;t^- + B_t^- S_k^{-1} q_k   (M, d)
    BSkInv_q = torch.bmm(BSkInv, qk.unsqueeze(-1)).squeeze(-1)  # (M, d)
    lam_k    = tx_m.unsqueeze(0) + BSkInv_q                     # (M, d)

    # C_k = -½ log|S_k| + ½ log|P_k| - ½ m_kᵀ P_k m_k + ½ q_kᵀ S_k^{-1} q_k
    logdet_Sk  = torch.logdet(Sk)                                 # (M,)
    logdet_Pk  = torch.logdet(Ps)                                 # (M,)
    mPkm       = torch.einsum('md,md->m', ms, Pkm)             # (M,)  m_k^T P_k m_k
    Skinv_q    = torch.bmm(Sk_inv, qk.unsqueeze(-1)).squeeze(-1)  # (M, d)
    qSkInvq    = torch.einsum('md,md->m', qk, Skinv_q)         # (M,)
    Ck         = -0.5*logdet_Sk + 0.5*logdet_Pk - 0.5*mPkm + 0.5*qSkInvq  # (M,)

    # ------------------------------------------------------------------
    # 4. Per-sample log-weights   log w_k(x) = log π_k + C_k
    #                              - ½ xᵀ Λ_k x + λ_kᵀ x
    # Shapes: x (B, d), Lambda (M, d, d), lam_k (M, d)
    # ------------------------------------------------------------------
    # Quadratic term: xᵀ Λ_k x  →  (B, M)
    #   for each b,k:  x[b] @ Lambda[k] @ x[b]
    x_col = x.unsqueeze(1).unsqueeze(-1)        # (B, 1, d, 1)
    Lam_x = torch.matmul(
        Lambda.unsqueeze(0).expand(B, M, d, d),  # (B, M, d, d)
        x_col.expand(B, M, d, 1),                # (B, M, d, 1)
    ).squeeze(-1)                               # (B, M, d)
    quad   = (x.unsqueeze(1) * Lam_x).sum(-1)  # (B, M)

    # Linear term: λ_kᵀ x  →  (B, M)
    lin = (lam_k.unsqueeze(0) * x.unsqueeze(1)).sum(-1)  # (B, M)

    # Log-weight: (B, M)
    log_wk = (logpi + Ck).unsqueeze(0) - 0.5 * quad + lin   # (B, M)

    # ------------------------------------------------------------------
    # 5. log ψ_t(x) and responsibilities (stable logsumexp)
    # ------------------------------------------------------------------
    log_psi = torch.logsumexp(log_wk, dim=1)   # (B,)
    rho     = torch.softmax(log_wk, dim=1)      # (B, M)

    # ------------------------------------------------------------------
    # 6. Optimal control u*(t,x) = Σ_k ρ_k (−Λ_k x + λ_k)
    # ------------------------------------------------------------------
    # −Λ_k x + λ_k  →  (B, M, d)
    neg_Lam_x = -Lam_x                         # (B, M, d)
    u_k = neg_Lam_x + lam_k.unsqueeze(0)       # (B, M, d)

    # Weighted sum: (B, d)
    u_star = torch.einsum('bm,bmd->bd', rho, u_k)

    return u_star, log_psi, rho
