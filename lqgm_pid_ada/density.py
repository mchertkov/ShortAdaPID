"""
density.py — Exact marginal density in Gaussian-mixture form for LQ-GM-PID.

This module provides `exact_marginal_gmm`, which computes the instantaneous
marginal p_t(x) as an explicit Gaussian mixture using the analytic formulas
from Section 5.4 of the LQ-GM-PID notes (eqs 77–82).

Key implementation note
-----------------------
The LQGMPID object's forward sweep is run with x0_mean=self.x0, which
encodes x0 into the forward BC's θ^+_x via  θ^+_{x;0} = (1/ε)×x0.
After propagation, θ^+_{x;t} contains BOTH the ν-driven contribution AND
a propagated-x0 term that equals B^+_t × x0 exactly (because the θ_x and
B ODEs share the same homogeneous operator).

This "baked-in x0" convention is correct for the control computation (where
the terminal forward coefficients enter a ratio with minus signs) but causes
double-counting in the density formula (where the forward kernel at time t
enters as a numerator).

Fix: we run a SEPARATE forward sweep with x0_mean=None (standard two-point
Green function) and use it consistently for both the ratio-form quantities
(S, q, Λ, λ) and the density product (Π, μ, w̄).  The starting-point x0
enters only through the explicit  B^+ × x0  products, exactly once each.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .pid import LQGMPID
from .core import CoeffState, GaussianMixture, MatrixPWCProtocol
from .sweep import forward_sweep
from .control import eval_bwd, eval_fwd

Tensor = torch.Tensor


def exact_marginal_gmm(
    pid: LQGMPID,
    t: float | Tensor,
    *,
    fwd_std: Optional[List[CoeffState]] = None,
) -> Dict[str, Tensor]:
    """Return the instantaneous marginal p_t(x) as a Gaussian mixture.

    Parameters
    ----------
    pid : LQGMPID
        A precomputed controller (pid.precompute() must have been called).
    t : float
        Query time in (0, 1).
    fwd_std : list of CoeffState, optional
        Forward sweep with standard BC (x0_mean=None).  If None, uses
        pid._fwd which is already a standard sweep after the pid.py fix.

    Returns
    -------
    dict with keys:
        weights    : (M,)    mixture weights  (sum to 1)
        means      : (M, d)  component means
        covs       : (M, d, d)  component covariances
        precisions : (M, d, d)  component precisions (= inv(covs))
    """
    pid.precompute()

    # ----------------------------------------------------------------
    # Standard forward sweep (pid._fwd is standard after pid.py fix)
    # ----------------------------------------------------------------
    if fwd_std is None:
        fwd_std = pid._fwd

    dev = pid.protocol.device
    dty = pid.protocol.dtype
    K_intervals = pid.protocol.K

    target = pid.target.to(device=dev, dtype=dty)
    x0 = pid.x0.to(dev, dty)

    # ----------------------------------------------------------------
    # Backward coefficients at time t  (same as control code)
    # ----------------------------------------------------------------
    cs_bwd = eval_bwd(t, pid._bwd, pid.protocol)
    A_m  = cs_bwd.A.to(dev, dty)
    B_m  = cs_bwd.B.to(dev, dty)
    C_m  = cs_bwd.C.to(dev, dty)
    tx_m = cs_bwd.theta_x.to(dev, dty)
    ty_m = cs_bwd.theta_y.to(dev, dty)

    # ----------------------------------------------------------------
    # Standard forward coefficients at time t  (for the density numerator)
    # ----------------------------------------------------------------
    cs_fwd_t = eval_fwd(t, fwd_std, pid.protocol)
    A_p  = cs_fwd_t.A.to(dev, dty)
    B_p  = cs_fwd_t.B.to(dev, dty)
    tx_p = cs_fwd_t.theta_x.to(dev, dty)

    # ----------------------------------------------------------------
    # Standard forward coefficients at terminal t = 1  (for the ratio form)
    # ----------------------------------------------------------------
    cs_fwd1 = fwd_std[K_intervals]
    A1_p  = cs_fwd1.A.to(dev, dty)
    B1_p  = cs_fwd1.B.to(dev, dty)
    tx1_p = cs_fwd1.theta_x.to(dev, dty)

    # ----------------------------------------------------------------
    # Per-component ratio-form quantities  (eqs 67–72 of notes)
    # ----------------------------------------------------------------
    Ps    = target.precisions.to(dev, dty)     # (M, d, d)
    ms    = target.means.to(dev, dty)          # (M, d)
    logpi = torch.log(target.weights.to(dev, dty))  # (M,)
    M = target.M
    d = target.d

    # S_k = C^-_t + P_k − A^+_1                          (M, d, d)
    Sk = C_m.unsqueeze(0) + Ps - A1_p.unsqueeze(0)

    # q_k = θ^-_{y;t} + P_k m_k − B^+_1 x0 − θ^+_{x;1}  (M, d)
    Pkm    = torch.bmm(Ps, ms.unsqueeze(-1)).squeeze(-1)
    B1p_x0 = B1_p @ x0
    qk     = (ty_m.unsqueeze(0) + Pkm
              - B1p_x0.unsqueeze(0) - tx1_p.unsqueeze(0))

    Sk_inv = torch.linalg.inv(Sk)                          # (M, d, d)

    # Λ_k = A^-_t − B^-_t S_k^{-1} (B^-_t)^T              (M, d, d)
    Bm_exp = B_m.unsqueeze(0).expand(M, d, d)
    BSkInv = torch.bmm(Bm_exp, Sk_inv)
    Lambda = A_m.unsqueeze(0) - torch.bmm(BSkInv, Bm_exp.transpose(-2, -1))

    # λ_k = θ^-_{x;t} + B^-_t S_k^{-1} q_k                (M, d)
    BSkInv_q = torch.bmm(BSkInv, qk.unsqueeze(-1)).squeeze(-1)
    lam_k    = tx_m.unsqueeze(0) + BSkInv_q

    # ----------------------------------------------------------------
    # Density:  p*_t(x) ∝ G^+_t(x|x0) × Σ_k ψ_{k,t}(x)
    #
    # Π_k = A^+_t + Λ_k                                     (M, d, d)
    # μ_k = Π_k^{-1} (B^+_t x0 + θ^+_{x;t} + λ_k)         (M, d)
    #
    # With the STANDARD forward sweep, B^+_t x0 and θ^+_{x;t} are
    # independent — no double-counting.
    # ----------------------------------------------------------------
    Pi    = A_p.unsqueeze(0) + Lambda                       # (M, d, d)
    Sigma = torch.linalg.inv(Pi)                            # (M, d, d)

    Bp_x0 = B_p @ x0                                       # (d,)
    rhs   = (Bp_x0 + tx_p).unsqueeze(0) + lam_k            # (M, d)
    mu    = torch.bmm(Sigma, rhs.unsqueeze(-1)).squeeze(-1) # (M, d)

    # ----------------------------------------------------------------
    # Mixture weights  (eq 82, dropping k-independent constants)
    #
    # log w̃_k = log π_k + C_k − ½ log|Π_k| + ½ μ_k^T Π_k μ_k
    # ----------------------------------------------------------------
    logdet_Sk = torch.logdet(Sk)
    logdet_Pk = torch.logdet(Ps)
    mPkm      = torch.einsum('md,md->m', ms, Pkm)
    Skinv_qk  = torch.bmm(Sk_inv, qk.unsqueeze(-1)).squeeze(-1)
    qSinvq    = torch.einsum('md,md->m', qk, Skinv_qk)

    Ck = -0.5 * logdet_Sk + 0.5 * logdet_Pk - 0.5 * mPkm + 0.5 * qSinvq

    logdet_Pi = torch.logdet(Pi)
    muPimu    = torch.einsum('md,md->m', mu,
                             torch.bmm(Pi, mu.unsqueeze(-1)).squeeze(-1))

    logw = logpi + Ck - 0.5 * logdet_Pi + 0.5 * muPimu
    logw = logw - torch.logsumexp(logw, dim=0)
    wbar = torch.exp(logw)

    return {
        "weights":    wbar,
        "means":      mu,
        "covs":       Sigma,
        "precisions": Pi,
    }


def make_standard_forward_sweep(pid: LQGMPID) -> List[CoeffState]:
    """Return the standard forward sweep from the PID controller.

    After the pid.py fix, pid._fwd is already a standard sweep
    (x0_mean=None), so this simply returns pid._fwd.
    """
    pid.precompute()
    return pid._fwd
