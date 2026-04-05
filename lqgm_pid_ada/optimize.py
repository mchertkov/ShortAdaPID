
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Literal, Optional

import numpy as np
import torch

from .density import exact_marginal_gmm
from .metrics import predicted_state_and_responsibilities
from .pid import LQGMPID
from .protocol_spec import make_pwc_beta_spec
from .gmm_spec import GMMSpec, build_gmm

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

Tensor = torch.Tensor
QuadratureRule = Literal["means", "sigma_points"]
ObjectiveName = Literal["vgs", "ahat", "cost"]


@dataclass
class DeterministicCurveResult:
    times: Tensor
    A: Optional[Tensor] = None
    Ahat: Optional[Tensor] = None
    vgs: Optional[Tensor] = None
    kinetic: Optional[Tensor] = None
    potential: Optional[Tensor] = None
    total_cost: Optional[Tensor] = None
    t_cross_Ahat_half: Optional[float] = None


@dataclass
class OptimizeBetaResult:
    objective: str
    success: bool
    message: str
    beta_values: Tensor
    objective_value: float
    nfev: int
    nit: Optional[int]
    curves: Optional[DeterministicCurveResult] = None


def terminal_second_moment(target) -> Tensor:
    w = target.weights
    m = target.means
    cov = target.covs
    vals = torch.einsum('m,mii->', w, cov) + torch.einsum('m,md,md->', w, m, m)
    return vals


def first_crossing_time(times: Tensor, values: Tensor, threshold: float = 0.5) -> float:
    mask = values >= threshold
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return 1.0
    return float(times[int(idx[0].item())].item())


def _component_points(mean: Tensor, cov: Tensor, *, rule: QuadratureRule = "means", rank: int = 4):
    d = mean.numel()
    dev, dty = mean.device, mean.dtype
    if rule == "means":
        return mean.unsqueeze(0), torch.ones(1, device=dev, dtype=dty)

    # very small reduced sigma-point rule using top-r eigendirections
    rank = max(1, min(rank, d))
    evals, evecs = torch.linalg.eigh(cov)
    idx = torch.argsort(evals, descending=True)[:rank]
    evals = evals[idx].clamp_min(1e-12)
    evecs = evecs[:, idx]
    pts = [mean]
    wts = [torch.as_tensor(1.0 / (2 * rank + 1), device=dev, dtype=dty)]
    scale = torch.sqrt(torch.as_tensor(float(rank), device=dev, dtype=dty))
    for j in range(rank):
        delta = scale * torch.sqrt(evals[j]) * evecs[:, j]
        pts.append(mean + delta)
        pts.append(mean - delta)
        w = torch.as_tensor(1.0 / (2 * rank + 1), device=dev, dtype=dty)
        wts.extend([w, w])
    return torch.stack(pts, dim=0), torch.stack(wts, dim=0)


def _beta_at(pid: LQGMPID, t: Tensor) -> Tensor:
    breaks = pid.protocol.breaks
    idx = torch.searchsorted(breaks, t.to(breaks.dtype), right=True).item() - 1
    idx = max(0, min(int(idx), pid.protocol.K - 1))
    return pid.protocol.beta[idx]


def _control_jacobian_single(pid: LQGMPID, t: Tensor, x: Tensor) -> Tensor:
    x = x.detach().clone().requires_grad_(True)
    u = pid.control(t, x.unsqueeze(0)).squeeze(0)
    d = x.numel()
    rows = []
    for j in range(d):
        (grad_j,) = torch.autograd.grad(
            u[j], x, retain_graph=(j < d - 1), create_graph=False, allow_unused=False
        )
        rows.append(grad_j.unsqueeze(0))
    return torch.cat(rows, dim=0)


def deterministic_expectation(
    pid: LQGMPID,
    t: float | Tensor,
    fn: Callable[[Tensor], Tensor],
    *,
    rule: QuadratureRule = "means",
    rank: int = 4,
) -> Tensor:
    gm = exact_marginal_gmm(pid, t)
    w = gm["weights"]
    means = gm["means"]
    covs = gm["covs"]
    out = None
    for k in range(w.numel()):
        pts, pwt = _component_points(means[k], covs[k], rule=rule, rank=rank)
        vals = fn(pts)
        comp = (pwt.reshape(-1, *([1] * (vals.ndim - 1))) * vals).sum(dim=0)
        out = w[k] * comp if out is None else out + w[k] * comp
    return out


def deterministic_curves(
    pid: LQGMPID,
    times: Tensor,
    *,
    compute_Ahat: bool = True,
    compute_vgs: bool = True,
    compute_cost: bool = True,
    rule: QuadratureRule = "means",
    rank: int = 4,
) -> DeterministicCurveResult:
    dev, dty = pid.x0.device, pid.x0.dtype
    times = times.to(device=dev, dtype=dty)
    denom = terminal_second_moment(pid.target.to(device=dev, dtype=dty)).clamp_min(torch.finfo(dty).eps)

    A = [] if compute_Ahat else None
    Ahat = [] if compute_Ahat else None
    vgs = [] if compute_vgs else None
    kin = [] if compute_cost else None
    pot = [] if compute_cost else None

    for t in times:
        if compute_Ahat:
            def fn_A(pts):
                yhat, _ = predicted_state_and_responsibilities(pid, t, pts)
                a = (pts * yhat).sum(dim=1) / denom
                ah = yhat.square().sum(dim=1) / denom
                return torch.stack([a, ah], dim=1)

            cur = deterministic_expectation(pid, t, fn_A, rule=rule, rank=rank)
            A.append(cur[0])
            Ahat.append(cur[1])

        if compute_vgs:
            def fn_vgs(pts):
                vals = []
                for x in pts:
                    J = _control_jacobian_single(pid, t, x)
                    vals.append(J.square().sum().unsqueeze(0))
                return torch.cat(vals, dim=0)
            cur = deterministic_expectation(pid, t, fn_vgs, rule=rule, rank=rank)
            vgs.append(cur)

        if compute_cost:
            beta_t = _beta_at(pid, t).to(device=dev, dtype=dty)
            gm = exact_marginal_gmm(pid, t)
            w = gm["weights"]
            means = gm["means"]
            covs = gm["covs"]

            # exact potential under GMM
            pot_t = torch.as_tensor(0.0, device=dev, dtype=dty)
            for k in range(w.numel()):
                tr_term = torch.trace(beta_t @ covs[k])
                mu_term = means[k] @ beta_t @ means[k]
                pot_t = pot_t + 0.5 * w[k] * (tr_term + mu_term)
            pot.append(pot_t)

            # deterministic kinetic approximation
            def fn_kin(pts):
                u = pid.control(t, pts)
                return 0.5 * u.square().sum(dim=1)
            kin_t = deterministic_expectation(pid, t, fn_kin, rule=rule, rank=rank)
            kin.append(kin_t)

    A_t = torch.stack(A) if A is not None else None
    Ahat_t = torch.stack(Ahat) if Ahat is not None else None
    vgs_t = torch.stack(vgs) if vgs is not None else None
    kin_t = torch.stack(kin) if kin is not None else None
    pot_t = torch.stack(pot) if pot is not None else None
    total_t = (kin_t + pot_t) if compute_cost else None
    tcross = first_crossing_time(times, Ahat_t, 0.5) if Ahat_t is not None else None

    return DeterministicCurveResult(
        times=times,
        A=A_t,
        Ahat=Ahat_t,
        vgs=vgs_t,
        kinetic=kin_t,
        potential=pot_t,
        total_cost=total_t,
        t_cross_Ahat_half=tcross,
    )


def integrate_time_series(times: Tensor, values: Tensor) -> Tensor:
    if values.numel() <= 1:
        return torch.as_tensor(0.0, device=values.device, dtype=values.dtype)
    return torch.trapz(values, times)


def objective_from_curves(curves: DeterministicCurveResult, objective: ObjectiveName) -> Tensor:
    if objective == "vgs":
        return integrate_time_series(curves.times, curves.vgs)
    if objective == "ahat":
        return -integrate_time_series(curves.times, curves.Ahat)
    if objective == "cost":
        return integrate_time_series(curves.times, curves.total_cost)
    raise ValueError(f"Unknown objective {objective!r}")


def make_pid_from_beta(
    gmm_spec: GMMSpec,
    d: int,
    breaks: Iterable[float],
    beta_values: Iterable[float],
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> LQGMPID:
    target = build_gmm(gmm_spec).to(device=device, dtype=dtype)
    pspec = make_pwc_beta_spec(d=d, breaks=list(breaks), beta_values=list(beta_values), family="optimized_pwc", device=device, dtype=dtype)
    proto = pspec.build()
    x0 = torch.zeros(d, device=device, dtype=dtype)
    pid = LQGMPID(proto, target, x0)
    pid.precompute()
    return pid


def optimize_pwc_beta(
    *,
    gmm_spec: GMMSpec,
    d: int,
    breaks: Iterable[float],
    objective: ObjectiveName,
    init_beta_values: Iterable[float],
    times: Iterable[float],
    bounds: tuple[float, float] = (0.1, 5.0),
    rule: QuadratureRule = "means",
    rank: int = 4,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    method: str = "Powell",
    maxiter: int = 40,
    return_curves: bool = True,
) -> OptimizeBetaResult:
    init_beta = np.asarray(list(init_beta_values), dtype=float)
    times_t = torch.as_tensor(list(times), device=device, dtype=dtype)

    lower, upper = bounds

    def wrapped(theta_np):
        beta = np.clip(theta_np, lower, upper)
        pid = make_pid_from_beta(
            gmm_spec=gmm_spec, d=d, breaks=breaks, beta_values=beta,
            device=device, dtype=dtype,
        )
        curves = deterministic_curves(
            pid, times_t,
            compute_Ahat=(objective == "ahat"),
            compute_vgs=(objective == "vgs"),
            compute_cost=(objective == "cost"),
            rule=rule, rank=rank,
        )
        val = objective_from_curves(curves, objective)
        return float(val.detach().cpu().item())

    if minimize is None:
        # simple fallback: evaluate initial only
        beta_best = np.clip(init_beta, lower, upper)
        pid = make_pid_from_beta(gmm_spec=gmm_spec, d=d, breaks=breaks, beta_values=beta_best, device=device, dtype=dtype)
        curves = deterministic_curves(
            pid, times_t, compute_Ahat=True, compute_vgs=True, compute_cost=True, rule=rule, rank=rank
        ) if return_curves else None
        val = wrapped(beta_best)
        return OptimizeBetaResult(
            objective=objective, success=False, message="scipy.optimize unavailable",
            beta_values=torch.as_tensor(beta_best, device=device, dtype=dtype),
            objective_value=float(val), nfev=1, nit=None, curves=curves
        )

    res = minimize(
        wrapped, x0=init_beta, method=method,
        bounds=[bounds] * len(init_beta),
        options={"maxiter": maxiter, "disp": False},
    )
    beta_best = np.clip(res.x, lower, upper)

    curves = None
    if return_curves:
        pid = make_pid_from_beta(
            gmm_spec=gmm_spec, d=d, breaks=breaks, beta_values=beta_best,
            device=device, dtype=dtype,
        )
        curves = deterministic_curves(
            pid, times_t, compute_Ahat=True, compute_vgs=True, compute_cost=True, rule=rule, rank=rank
        )

    return OptimizeBetaResult(
        objective=objective,
        success=bool(res.success),
        message=str(res.message),
        beta_values=torch.as_tensor(beta_best, device=device, dtype=dtype),
        objective_value=float(res.fun),
        nfev=int(getattr(res, "nfev", -1)),
        nit=getattr(res, "nit", None),
        curves=curves,
    )


def scan_constant_beta(
    *,
    gmm_spec: GMMSpec,
    d: int,
    betas: Iterable[float],
    times: Iterable[float],
    rule: QuadratureRule = "means",
    rank: int = 4,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    rows = []
    times_t = torch.as_tensor(list(times), device=device, dtype=dtype)
    for beta in betas:
        pid = make_pid_from_beta(gmm_spec=gmm_spec, d=d, breaks=[0.0, 0.25, 0.5, 0.75, 1.0], beta_values=[beta]*4, device=device, dtype=dtype)
        curves = deterministic_curves(pid, times_t, compute_Ahat=True, compute_vgs=True, compute_cost=True, rule=rule, rank=rank)
        rows.append({
            "beta": float(beta),
            "J_vgs": float(objective_from_curves(curves, "vgs").item()),
            "J_ahat": float(objective_from_curves(curves, "ahat").item()),
            "J_cost": float(objective_from_curves(curves, "cost").item()),
            "t_cross_Ahat_half": curves.t_cross_Ahat_half,
        })
    return rows
