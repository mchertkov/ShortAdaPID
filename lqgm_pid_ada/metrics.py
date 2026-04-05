from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .control import eval_bwd
from .core import GaussianMixture
from .pid import LQGMPID, _SimResult

Tensor = torch.Tensor


@dataclass
class CostToGoResult:
    times: Tensor
    kinetic_rate: Tensor
    potential_rate: Tensor
    total_rate: Tensor
    kinetic_cum: Tensor
    potential_cum: Tensor
    total_cum: Tensor
    kinetic_share: Tensor
    potential_share: Tensor


@dataclass
class VelocityGradientSensitivityResult:
    times: Tensor
    mean_frob_sq: Tensor
    mean_trace: Tensor
    mean_eig_max: Tensor
    mean_eig_min: Tensor
    n_particles_used: int


@dataclass
class AutoCorrelationResult:
    times: Tensor
    A: Tensor
    Ahat: Tensor
    t_cross_A_half: float
    t_cross_Ahat_half: float


@dataclass
class SpeciationTimingResult:
    times: Tensor
    confidence: Tensor
    margin: Tensor
    entropy: Tensor
    predicted_label: Tensor
    oracle_label: Tensor
    accuracy: Tensor
    risk: Tensor
    reliable_time: Tensor
    reliable_cdf: Tensor
    yhat: Tensor


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def cost_to_go_decomposition(
    pid: LQGMPID,
    sim: Optional[_SimResult] = None,
    *,
    B: int = 256,
    n_steps: int = 2000,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> CostToGoResult:
    """Estimate dynamic cost-to-go decomposition along simulated trajectories.

    Implements the AdaPID decomposition
        C = C_pot + C_kin,
    with instantaneous rates
        0.5 E ||u_t^*(X_t)||^2,
        0.5 E [X_t^T beta_t X_t].

    The current implementation assumes the unit-diffusion, zero-base-drift AdaPID
    restriction used in the revision experiments.  For general protocols, the same
    formulas remain valid for the kinetic/potential pieces when kappa = 1.
    """
    sim = _ensure_sim(pid, sim, B=B, n_steps=n_steps, seed=seed, dtype=dtype, device=device)
    times = sim.times
    traj = sim.traj
    dt = sim.dt

    Tm1 = times.numel() - 1
    kinetic_rate = torch.empty(Tm1, dtype=traj.dtype, device=traj.device)
    potential_rate = torch.empty(Tm1, dtype=traj.dtype, device=traj.device)

    for i in range(Tm1):
        t = times[i]
        x = traj[i]
        u = pid.control(t, x)
        kinetic_rate[i] = 0.5 * (u.square().sum(dim=1)).mean()

        beta_t = _beta_at(pid, t).to(device=x.device, dtype=x.dtype)
        beta_x = x @ beta_t.T
        potential_rate[i] = 0.5 * (x * beta_x).sum(dim=1).mean()

    kinetic_cum = _left_cumulative_integral(kinetic_rate, dt)
    potential_cum = _left_cumulative_integral(potential_rate, dt)
    total_rate = kinetic_rate + potential_rate
    total_cum = kinetic_cum + potential_cum
    denom = total_cum.clamp_min(torch.finfo(total_cum.dtype).eps)
    kinetic_share = kinetic_cum / denom
    potential_share = potential_cum / denom

    return CostToGoResult(
        times=times[:-1],
        kinetic_rate=kinetic_rate,
        potential_rate=potential_rate,
        total_rate=total_rate,
        kinetic_cum=kinetic_cum,
        potential_cum=potential_cum,
        total_cum=total_cum,
        kinetic_share=kinetic_share,
        potential_share=potential_share,
    )


def velocity_gradient_sensitivity(
    pid: LQGMPID,
    sim: Optional[_SimResult] = None,
    *,
    B: int = 256,
    n_steps: int = 2000,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    max_particles: int = 16,
    time_stride: int = 10,
) -> VelocityGradientSensitivityResult:
    """Estimate Lagrangian velocity-gradient statistics along sample paths.

    Returns the time series of mean Frobenius-squared norm, trace, and extremal
    eigenvalues of the Jacobian Omega_t(x) = du*(t,x)/dx, averaged over a subset
    of particles at sampled times.
    """
    sim = _ensure_sim(pid, sim, B=B, n_steps=n_steps, seed=seed, dtype=dtype, device=device)
    times_full = sim.times[:-1]
    traj_full = sim.traj[:-1]

    time_idx = torch.arange(0, times_full.numel(), time_stride, device=times_full.device)
    times = times_full[time_idx]
    traj = traj_full[time_idx]

    B_total = traj.shape[1]
    n_use = min(B_total, max_particles)
    particle_idx = torch.arange(n_use, device=traj.device)

    mean_frob_sq = torch.empty(times.numel(), dtype=traj.dtype, device=traj.device)
    mean_trace = torch.empty_like(mean_frob_sq)
    mean_eig_max = torch.empty_like(mean_frob_sq)
    mean_eig_min = torch.empty_like(mean_frob_sq)

    for ti in range(times.numel()):
        t = times[ti]
        xs = traj[ti, particle_idx]
        frob_vals = []
        trace_vals = []
        eigmax_vals = []
        eigmin_vals = []
        for x in xs:
            J = _control_jacobian_single(pid, t, x)
            frob_vals.append((J.square().sum()).unsqueeze(0))
            trace_vals.append(torch.trace(J).unsqueeze(0))
            symJ = 0.5 * (J + J.T)
            evals = torch.linalg.eigvalsh(symJ)
            eigmin_vals.append(evals[0].unsqueeze(0))
            eigmax_vals.append(evals[-1].unsqueeze(0))
        mean_frob_sq[ti] = torch.cat(frob_vals).mean()
        mean_trace[ti] = torch.cat(trace_vals).mean()
        mean_eig_max[ti] = torch.cat(eigmax_vals).mean()
        mean_eig_min[ti] = torch.cat(eigmin_vals).mean()

    return VelocityGradientSensitivityResult(
        times=times,
        mean_frob_sq=mean_frob_sq,
        mean_trace=mean_trace,
        mean_eig_max=mean_eig_max,
        mean_eig_min=mean_eig_min,
        n_particles_used=n_use,
    )



def auto_correlation(
    pid: LQGMPID,
    sim: Optional[_SimResult] = None,
    *,
    B: int = 256,
    n_steps: int = 2000,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    threshold: float = 0.5,
) -> AutoCorrelationResult:
    """Compute AdaPID auto-correlations A(t) and Ahat(t) from simulated paths.

    A(t)    = E[x_t^T x_1] / E[||x_1||^2]
    Ahat(t) = E[yhat(t; x_t)^T x_1] / E[||x_1||^2]
    """
    sim = _ensure_sim(pid, sim, B=B, n_steps=n_steps, seed=seed, dtype=dtype, device=device)
    times = sim.times[:-1]
    traj = sim.traj[:-1]
    x1 = sim.traj[-1]
    denom = x1.square().sum(dim=1).mean().clamp_min(torch.finfo(x1.dtype).eps)

    Tm1 = times.numel()
    A = torch.empty(Tm1, dtype=traj.dtype, device=traj.device)
    Ahat = torch.empty_like(A)
    for i in range(Tm1):
        t = times[i]
        xt = traj[i]
        A[i] = (xt * x1).sum(dim=1).mean() / denom
        yhat, _ = predicted_state_and_responsibilities(pid, t, xt)
        Ahat[i] = (yhat * x1).sum(dim=1).mean() / denom

    return AutoCorrelationResult(
        times=times,
        A=A,
        Ahat=Ahat,
        t_cross_A_half=_first_crossing_time(times, A, threshold),
        t_cross_Ahat_half=_first_crossing_time(times, Ahat, threshold),
    )


def speciation_timing(
    pid: LQGMPID,
    sim: Optional[_SimResult] = None,
    *,
    B: int = 256,
    n_steps: int = 2000,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    c_star: float = 0.92,
    margin_star: float = 0.5,
    entropy_star: Optional[float] = None,
    tau_window: float = 0.05,
    tau_min: float = 0.5,
) -> SpeciationTimingResult:
    """Evaluate prediction-based speciation timing diagnostics.

    Implements the AdaPID reliable-decision-time protocol using prediction-based
    responsibilities computed from the predicted state yhat(t; x_t).
    """
    sim = _ensure_sim(pid, sim, B=B, n_steps=n_steps, seed=seed, dtype=dtype, device=device)
    times = sim.times[:-1]
    traj = sim.traj[:-1]
    Tm1, Bn, d = traj.shape
    K = pid.target.M

    if entropy_star is None:
        p = torch.as_tensor(c_star, dtype=traj.dtype, device=traj.device)
        q = 1.0 - p
        entropy_star = float(-(p * torch.log(p) + q * torch.log(q)).item())

    confidence = torch.empty(Tm1, Bn, dtype=traj.dtype, device=traj.device)
    margin = torch.empty_like(confidence)
    entropy = torch.empty_like(confidence)
    predicted_label = torch.empty(Tm1, Bn, dtype=torch.long, device=traj.device)
    yhat = torch.empty(Tm1, Bn, d, dtype=traj.dtype, device=traj.device)

    for i in range(Tm1):
        t = times[i]
        x = traj[i]
        yhat_i, r_pred = predicted_state_and_responsibilities(pid, t, x)
        yhat[i] = yhat_i
        predicted_label[i] = torch.argmax(r_pred, dim=1)
        confidence[i] = torch.max(r_pred, dim=1).values
        top2 = torch.topk(r_pred, k=min(2, K), dim=1).values
        if K >= 2:
            margin[i] = top2[:, 0] - top2[:, 1]
        else:
            margin[i] = torch.ones(Bn, dtype=traj.dtype, device=traj.device)
        entropy[i] = -(r_pred.clamp_min(1e-12) * r_pred.clamp_min(1e-12).log()).sum(dim=1)

    # terminal oracle labels from target posterior at final simulated state
    x1 = sim.traj[-1]
    r_tar = target_posterior(pid.target, x1)
    oracle_label = torch.argmax(r_tar, dim=1)

    accuracy = (predicted_label == oracle_label.unsqueeze(0)).to(traj.dtype).mean(dim=1)
    risk = 1.0 - confidence.mean(dim=1)

    reliable_time = torch.ones(Bn, dtype=traj.dtype, device=traj.device)
    label_stable_steps = _window_steps(times, tau_window)
    min_idx = int(torch.searchsorted(times, torch.as_tensor(tau_min, dtype=times.dtype, device=times.device), right=False).item())

    for b in range(Bn):
        found = False
        for i in range(min_idx, Tm1):
            if not (
                confidence[i, b] >= c_star and
                margin[i, b] >= margin_star and
                entropy[i, b] <= entropy_star
            ):
                continue
            j0 = max(0, i - label_stable_steps + 1)
            if torch.all(predicted_label[j0:i + 1, b] == predicted_label[i, b]):
                reliable_time[b] = times[i]
                found = True
                break
        if not found:
            reliable_time[b] = torch.as_tensor(1.0, dtype=traj.dtype, device=traj.device)

    reliable_cdf = torch.stack([(reliable_time <= t).to(traj.dtype).mean() for t in times])

    return SpeciationTimingResult(
        times=times,
        confidence=confidence,
        margin=margin,
        entropy=entropy,
        predicted_label=predicted_label,
        oracle_label=oracle_label,
        accuracy=accuracy,
        risk=risk,
        reliable_time=reliable_time,
        reliable_cdf=reliable_cdf,
        yhat=yhat,
    )


# ---------------------------------------------------------------------------
# Additional public helpers
# ---------------------------------------------------------------------------

def predicted_state_and_responsibilities(
    pid: LQGMPID,
    t: float | Tensor,
    x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return yhat(t;x) and prediction-based responsibilities r_pred.

    For the GMM ratio-form posterior over terminal states, the per-component
    conditional mean is
        mhat_k(t,x) = S_k^{-1}[(B_t^-)^T x + q_k],
    and the global predicted state is yhat = sum_k rho_k mhat_k.

    The prediction-based responsibilities are then defined by evaluating the
    target-mixture posterior at yhat, matching the AdaPID speciation section.
    """
    pid.precompute()
    x = _as_batch(x, pid.x0.device, pid.x0.dtype)
    dev, dty = x.device, x.dtype
    Bn, d = x.shape
    target = pid.target.to(device=dev, dtype=dty)

    cs_bwd = eval_bwd(t, pid._bwd, pid.protocol)
    Am = cs_bwd.A.to(dev, dty)
    Bm = cs_bwd.B.to(dev, dty)
    Cm = cs_bwd.C.to(dev, dty)
    tx_m = cs_bwd.theta_x.to(dev, dty)
    ty_m = cs_bwd.theta_y.to(dev, dty)

    cs_fwd1 = pid._fwd[pid.protocol.K]
    A1p = cs_fwd1.A.to(dev, dty)
    B1p = cs_fwd1.B.to(dev, dty)
    tx1p = cs_fwd1.theta_x.to(dev, dty)
    x0 = pid.x0.to(dev, dty)

    Ps = target.precisions.to(dev, dty)
    ms = target.means.to(dev, dty)
    logpi = torch.log(target.weights.to(dev, dty))
    M = target.M

    Sk = Cm.unsqueeze(0) + Ps - A1p.unsqueeze(0)
    Sk_inv = torch.linalg.inv(Sk)

    Pkm = torch.bmm(Ps, ms.unsqueeze(-1)).squeeze(-1)
    qk = ty_m.unsqueeze(0) + Pkm - (B1p @ x0).unsqueeze(0) - tx1p.unsqueeze(0)

    Bm_exp = Bm.unsqueeze(0).expand(M, d, d)
    BSkInv = torch.bmm(Bm_exp, Sk_inv)
    Lambda = Am.unsqueeze(0) - torch.bmm(BSkInv, Bm_exp.transpose(-2, -1))
    lam_k = tx_m.unsqueeze(0) + torch.bmm(BSkInv, qk.unsqueeze(-1)).squeeze(-1)

    logdet_Sk = torch.logdet(Sk)
    logdet_Pk = torch.logdet(Ps)
    mPkm = torch.einsum('md,md->m', ms, Pkm)
    Skinv_q = torch.bmm(Sk_inv, qk.unsqueeze(-1)).squeeze(-1)
    qSkInvq = torch.einsum('md,md->m', qk, Skinv_q)
    Ck = -0.5 * logdet_Sk + 0.5 * logdet_Pk - 0.5 * mPkm + 0.5 * qSkInvq

    x_col = x.unsqueeze(1).unsqueeze(-1)
    Lam_x = torch.matmul(Lambda.unsqueeze(0).expand(Bn, M, d, d), x_col.expand(Bn, M, d, 1)).squeeze(-1)
    quad = (x.unsqueeze(1) * Lam_x).sum(dim=-1)
    lin = (lam_k.unsqueeze(0) * x.unsqueeze(1)).sum(dim=-1)
    log_wk = (logpi + Ck).unsqueeze(0) - 0.5 * quad + lin
    rho = torch.softmax(log_wk, dim=1)

    Bt_x = torch.matmul(Bm.T, x.T).T  # (B, d)
    rhs = Bt_x.unsqueeze(1) + qk.unsqueeze(0)  # (B, M, d)
    mhat = torch.linalg.solve(
        Sk.unsqueeze(0).expand(Bn, M, d, d),
        rhs.unsqueeze(-1),
    ).squeeze(-1)
    yhat = torch.einsum('bm,bmd->bd', rho, mhat)
    r_pred = target_posterior(target, yhat)
    return yhat, r_pred


def target_posterior(target: GaussianMixture, x: Tensor) -> Tensor:
    """Posterior responsibilities of the target GMM at query points x."""
    x = _as_batch(x, target.means.device, target.means.dtype)
    dev, dty = x.device, x.dtype
    target = target.to(device=dev, dtype=dty)
    M, d = target.M, target.d

    diff = x.unsqueeze(1) - target.means.unsqueeze(0)  # (B, M, d)
    P = target.precisions
    maha = torch.einsum('bmd,mdk,bmk->bm', diff, P, diff)
    logdetP = torch.logdet(P).unsqueeze(0)
    logpi = torch.log(target.weights).unsqueeze(0)
    logw = logpi + 0.5 * logdetP - 0.5 * maha
    return torch.softmax(logw, dim=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_crossing_time(times: Tensor, values: Tensor, threshold: float) -> float:
    mask = values >= threshold
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return float("inf")
    return float(times[int(idx[0].item())].item())


def _ensure_sim(pid: LQGMPID, sim: Optional[_SimResult], **sim_kwargs: Any) -> _SimResult:
    if sim is not None:
        return sim
    return pid.simulate(**sim_kwargs)


def _left_cumulative_integral(rate: Tensor, dt: Tensor) -> Tensor:
    return torch.cumsum(rate * dt, dim=0)


def _beta_at(pid: LQGMPID, t: Tensor) -> Tensor:
    idx, _, _ = pid.protocol.locate(t)
    k = int(idx.item()) if hasattr(idx, 'item') else int(idx)
    return pid.protocol.beta[k]


def _as_batch(x: Tensor, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    x = torch.as_tensor(x, device=device, dtype=dtype)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f'x must be (B, d) or (d,), got {tuple(x.shape)}')
    return x


def _control_jacobian_single(pid: LQGMPID, t: Tensor, x: Tensor) -> Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    u = pid.control(t, x_req.unsqueeze(0)).squeeze(0)
    d = x_req.numel()
    rows = []
    for i in range(d):
        grad_i = torch.autograd.grad(u[i], x_req, retain_graph=(i < d - 1), create_graph=False)[0]
        rows.append(grad_i.unsqueeze(0))
    return torch.cat(rows, dim=0)


def _window_steps(times: Tensor, tau_window: float) -> int:
    if tau_window <= 0:
        return 1
    dt0 = (times[1] - times[0]).item() if times.numel() > 1 else 1.0
    return max(1, int(round(tau_window / max(dt0, 1e-12))))
