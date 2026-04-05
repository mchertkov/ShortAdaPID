# %% [markdown]
# # Model B deterministic scaling notebook
#
# This notebook is a deterministic VGS-focused study.
#
# Two scan families are included:
#
# 1. Dimension scan with fixed K=9:
#    - one random 9-mode instance from {-a,0,a}^d for each d
#
# 2. Fixed d=8, varying number of modes:
#    - K in {9, 18, 27, 36, 45}
#    - modes sampled uniformly without replacement from {-a,0,a}^8
#
# For both families the workflow is:
#   constant-beta scan -> warm start from best constant -> 1 -> 2 -> 4 -> 8 coordinate descent
#
# Objective:
#   deterministic VGS
#
# This notebook includes guards so that plotting cells can be run safely even
# if earlier heavy cells were interrupted.

# %% [markdown]
# ## Imports and repository bootstrap

# %%
import json
import itertools
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if not (ROOT / "lqgm_pid_ada").exists():
    for parent in [ROOT, *ROOT.parents]:
        if (parent / "lqgm_pid_ada").exists():
            ROOT = parent
            break

if not (ROOT / "lqgm_pid_ada").exists():
    raise RuntimeError("Could not locate `lqgm_pid_ada`.")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display

from lqgm_pid_ada import (
    GaussianMixture,
    LQGMPID,
    auto_correlation,
    exact_marginal_gmm,
    make_pwc_beta_spec,
)

# %% [markdown]
# ## Global settings

# %%
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)
DEVICE = "cpu"

# Model B geometry
GRID_A = 1.5
SIGMA_BASE = 0.3

# Family 1: dimension scan with fixed K=9
DIMS = [2, 4, 8, 16]
K_FIXED = 9

# Family 2: fixed d=8, varying K
D_FIXED = 8
K_LIST = [9, 18, 27, 36, 45]

# Protocol hierarchy
LEVELS = [1, 2, 4, 8]
BETA_INIT = 1.0
BETA_MIN = 0.05
BETA_MAX = 20.0

# Constant-beta scan
BETA_SWEEP = [0.05, 0.1, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]

# Deterministic evaluation grid
TIME_GRID_DET = np.linspace(0.05, 0.95, 24)
MAX_COMPONENTS_VGS = 8

# Optional memory curve budget for selected comparisons
B_AC = 768
N_STEPS_AC = 320
SEED_SIM = 20241012

# Coordinate descent
STEP0 = 0.6
STEP_MIN = 1e-4
IMPROVE_TOL = 5e-4
MAX_SWEEPS = 20

# Output
FIG_DIR = ROOT / "figs" / "modelB_vgs_scaling"
TABLE_DIR = ROOT / "tables" / "modelB_vgs_scaling"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["savefig.dpi"] = 220

# %% [markdown]
# ## Output helpers

# %%
def savefig_all(fig, stem: str, folder=FIG_DIR, close=False):
    pdf_path = folder / f"{stem}.pdf"
    png_path = folder / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")
    if close:
        plt.close(fig)

def save_table(df: pd.DataFrame, stem: str, folder=TABLE_DIR):
    csv_path = folder / f"{stem}.csv"
    tex_path = folder / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    try:
        df.to_latex(tex_path, index=False, float_format=lambda x: f"{x:.6g}")
    except Exception:
        pass
    print(f"saved: {csv_path}")

def save_json(obj, stem: str, folder=TABLE_DIR):
    path = folder / f"{stem}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"saved: {path}")

# %% [markdown]
# ## Gaussian-mixture construction helpers

# %%
def build_gaussian_mixture(weights, means, covs):
    tries = [
        lambda: GaussianMixture(weights=weights, means=means, covs=covs),
        lambda: GaussianMixture(means=means, covs=covs, weights=weights),
        lambda: GaussianMixture(weights, means, covs),
        lambda: GaussianMixture(means, covs, weights),
    ]
    last_err = None
    for ctor in tries:
        try:
            gmm = ctor()
            _ = gmm.weights
            _ = gmm.means
            _ = gmm.covs
            return gmm
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not construct GaussianMixture. Last error: {last_err}")

def all_hypercube_centers(d, a):
    vals = np.array([-a, 0.0, a], dtype=float)
    return np.array(list(itertools.product(vals, repeat=d)), dtype=float)

def gmm_from_centers(centers_np, sigma=SIGMA_BASE, device=DEVICE, dtype=torch.float64):
    Kloc, d = centers_np.shape
    means = torch.tensor(centers_np, device=device, dtype=dtype)
    covs = torch.stack(
        [torch.eye(d, device=device, dtype=dtype) * float(sigma) ** 2 for _ in range(Kloc)],
        dim=0,
    )
    weights = torch.ones(Kloc, device=device, dtype=dtype) / Kloc
    return build_gaussian_mixture(weights, means, covs)

# %% [markdown]
# ## Build cases
#
# Each case has:
# - family
# - case_name
# - d
# - K
# - centers

# %%
TARGETS = {}
TARGET_META = {}
center_rows = []

# Family 1: one K=9 random subset per dimension
for d in DIMS:
    all_centers = all_hypercube_centers(d, GRID_A)
    rng = np.random.default_rng(SEED + d)
    idx = rng.choice(len(all_centers), size=K_FIXED, replace=False)
    centers = all_centers[idx]

    case_name = f"dimscan_d{d}_K{K_FIXED}"
    TARGETS[case_name] = gmm_from_centers(centers, sigma=SIGMA_BASE)
    TARGET_META[case_name] = {
        "family": "dimension_scan",
        "case_name": case_name,
        "d": d,
        "K": K_FIXED,
        "centers": centers,
        "seed": SEED + d,
    }

    for i, c in enumerate(centers):
        row = {"family": "dimension_scan", "case_name": case_name, "d": d, "K": K_FIXED, "mode_index": i}
        for j, val in enumerate(c, start=1):
            row[f"coord_{j}"] = float(val)
        center_rows.append(row)

# Family 2: fixed d=8, varying K
all_centers_d8 = all_hypercube_centers(D_FIXED, GRID_A)
for Kcur in K_LIST:
    rng = np.random.default_rng(SEED + 1000 + Kcur)
    idx = rng.choice(len(all_centers_d8), size=Kcur, replace=False)
    centers = all_centers_d8[idx]

    case_name = f"kscan_d{D_FIXED}_K{Kcur}"
    TARGETS[case_name] = gmm_from_centers(centers, sigma=SIGMA_BASE)
    TARGET_META[case_name] = {
        "family": "mode_scan",
        "case_name": case_name,
        "d": D_FIXED,
        "K": Kcur,
        "centers": centers,
        "seed": SEED + 1000 + Kcur,
    }

    for i, c in enumerate(centers):
        row = {"family": "mode_scan", "case_name": case_name, "d": D_FIXED, "K": Kcur, "mode_index": i}
        for j, val in enumerate(c, start=1):
            row[f"coord_{j}"] = float(val)
        center_rows.append(row)

centers_df = pd.DataFrame(center_rows)
display(centers_df.head(20))
save_table(centers_df, "case_centers_modelB_vgs_scaling")

# %% [markdown]
# ## PID builder

# %%
def build_pid(case_name, beta_values):
    d = TARGET_META[case_name]["d"]
    pspec = make_pwc_beta_spec(
        d=d,
        breaks=np.linspace(0.0, 1.0, len(beta_values) + 1).tolist(),
        beta_values=list(np.asarray(beta_values, dtype=float)),
        family="optimized_pwc",
        device=DEVICE,
        dtype=torch.float64,
    )
    proto = pspec.build()
    x0 = torch.zeros(d, device=DEVICE, dtype=torch.float64)
    pid = LQGMPID(proto, TARGETS[case_name], x0)
    pid.precompute()
    return pid

# %% [markdown]
# ## Deterministic VGS and memory helpers

# %%
def control_jacobian_single(pid, t, x):
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

def deterministic_vgs_from_pid(pid, time_grid=TIME_GRID_DET, max_components_eval=MAX_COMPONENTS_VGS):
    vals_t, tvals = [], []

    for t in time_grid:
        gm = exact_marginal_gmm(pid, float(t))
        weights = gm["weights"]
        means = gm["means"]

        if weights.numel() > max_components_eval:
            idx = torch.argsort(weights, descending=True)[:max_components_eval]
            weights_use = weights[idx]
            weights_use = weights_use / weights_use.sum()
            means_use = means[idx]
        else:
            weights_use = weights
            means_use = means

        t_tensor = torch.as_tensor(float(t), device=means.device, dtype=means.dtype)
        vgs_t = 0.0
        for wk, mk in zip(weights_use, means_use):
            J = control_jacobian_single(pid, t_tensor, mk)
            vgs_t += float(wk.detach().cpu().item()) * float(J.square().sum().detach().cpu().item())

        vals_t.append(vgs_t)
        tvals.append(float(t))

    return float(np.trapz(vals_t, tvals))

def deterministic_vgs(case_name, beta_values):
    pid = build_pid(case_name, beta_values)
    return deterministic_vgs_from_pid(pid)

def interpolated_crossing_time(times, values, threshold=0.8, append_terminal_one=True):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    if append_terminal_one and times[-1] < 1.0:
        times = np.concatenate([times, [1.0]])
        values = np.concatenate([values, [1.0]])

    idx = np.where(values >= threshold)[0]
    if len(idx) == 0:
        return 1.0

    i = idx[0]
    if i == 0:
        return float(times[0])

    t0, t1 = times[i - 1], times[i]
    y0, y1 = values[i - 1], values[i]
    if abs(y1 - y0) < 1e-14:
        return float(t1)

    frac = (threshold - y0) / (y1 - y0)
    frac = np.clip(frac, 0.0, 1.0)
    return float(t0 + frac * (t1 - t0))

def evaluate_memory_curve(case_name, beta_values, B=B_AC, n_steps=N_STEPS_AC, seed=SEED_SIM):
    pid = build_pid(case_name, beta_values)
    sim = pid.simulate(B=B, n_steps=n_steps, seed=int(seed), dtype=torch.float64, device=DEVICE)
    ac = auto_correlation(pid, sim=sim, threshold=0.8)
    t = ac.times.detach().cpu().numpy()
    Ahat = ac.Ahat.detach().cpu().numpy()
    return {
        "times": t,
        "Ahat": Ahat,
        "t_cross_Ahat_08": interpolated_crossing_time(t, Ahat, threshold=0.8, append_terminal_one=True),
    }

def summarize_row(case_name, beta_values, kind, extra=None):
    J = deterministic_vgs(case_name, beta_values)
    mem = evaluate_memory_curve(case_name, beta_values)
    row = {
        "family": TARGET_META[case_name]["family"],
        "case_name": case_name,
        "d": TARGET_META[case_name]["d"],
        "K": TARGET_META[case_name]["K"],
        "kind": kind,
        "betas": np.array2string(np.asarray(beta_values, dtype=float), precision=6, separator=", "),
        "J_VGS": J,
        "J_VGS_per_d": J / TARGET_META[case_name]["d"],
        "t_cross_Ahat_08": mem["t_cross_Ahat_08"],
    }
    if extra is not None:
        row.update(extra)
    return row, mem

# %% [markdown]
# ## Coordinate descent

# %%
def objective_vgs(case_name, beta_values):
    return deterministic_vgs(case_name, beta_values)

def coord_descent_vgs(
    case_name,
    betas0,
    step0=STEP0,
    step_min=STEP_MIN,
    improve_tol=IMPROVE_TOL,
    max_sweeps=MAX_SWEEPS,
    beta_min=BETA_MIN,
    beta_max=BETA_MAX,
    record=None,
):
    b = np.asarray(betas0, dtype=float).copy()
    fbest = objective_vgs(case_name, b)
    if record is not None:
        record.append(float(fbest))

    step = float(step0)
    for _ in range(max_sweeps):
        improved = False
        for k in range(len(b)):
            for sgn in (+1, -1):
                cand = b.copy()
                cand[k] = np.clip(cand[k] + sgn * step, beta_min, beta_max)
                f = objective_vgs(case_name, cand)
                if f + improve_tol < fbest:
                    b, fbest = cand, f
                    improved = True
                    if record is not None:
                        record.append(float(fbest))
        if not improved:
            step *= 0.5
            if step < step_min:
                break
    return b, float(fbest)

def prolongate_parent_to_children(parent_betas, L):
    parent_betas = np.asarray(parent_betas, dtype=float)
    if len(parent_betas) == L:
        return parent_betas.copy()

    splitsL = np.linspace(0.0, 1.0, L + 1)
    parent_edges = np.linspace(0.0, 1.0, len(parent_betas) + 1)

    child = np.empty(L, dtype=float)
    for i in range(L):
        t_mid = 0.5 * (splitsL[i] + splitsL[i + 1])
        j = np.searchsorted(parent_edges, t_mid, side="right") - 1
        j = max(0, min(len(parent_betas) - 1, j))
        child[i] = parent_betas[j]
    return child

def step_plot_from_betas(betas):
    betas = np.asarray(betas, dtype=float)
    L = len(betas)
    edges = np.linspace(0.0, 1.0, L + 1)
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(betas, 2)
    return x, y

# %% [markdown]
# ## Run a full study on a list of cases

# %%
def run_case_family(case_names, family_label):
    const_rows = []
    best_const_rows = []
    hier_rows = []
    best_pwc8_rows = []
    cmp_rows = []
    best_records = {}
    traces = {}

    for case_name in case_names:
        print("=" * 80)
        print(f"{family_label}: {case_name}")

        # constant scan
        local_const = []
        const_mem_artifacts = {}
        for beta in BETA_SWEEP:
            row, mem = summarize_row(case_name, [beta], "constant_scan", extra={"beta_const": beta})
            const_rows.append(row)
            local_const.append(row)
            const_mem_artifacts[beta] = mem

        const_df_case = pd.DataFrame(local_const).sort_values("J_VGS").reset_index(drop=True)
        best_const = const_df_case.iloc[0].to_dict()
        best_const_rows.append(best_const)

        # hierarchy warm-started from best constant
        b_parent = np.array([float(best_const["beta_const"])], dtype=float)
        level_records = {}
        traj_f = []
        level_marks = []

        for L in LEVELS:
            if L == 1:
                b_init = b_parent.copy()
            else:
                b_init = prolongate_parent_to_children(b_parent, L)

            start_idx = len(traj_f)
            b_star, f_star = coord_descent_vgs(case_name, b_init, record=traj_f)
            level_marks.append(start_idx)

            row, mem = summarize_row(case_name, b_star, "hierarchical_vgs", extra={"L": L, "objective_value": f_star})
            hier_rows.append(row)
            level_records[L] = {
                "betas": np.asarray(b_star, dtype=float).copy(),
                "row": row,
                "mem": mem,
                "const_mem_artifacts": const_mem_artifacts,
            }
            b_parent = np.asarray(b_star, dtype=float).copy()

        best_records[case_name] = level_records
        traces[case_name] = {"traj_f": np.asarray(traj_f), "level_marks": np.asarray(level_marks)}

        best_pwc8 = level_records[8]["row"]
        best_pwc8_rows.append(best_pwc8)

        cmp_rows.append({
            "family": TARGET_META[case_name]["family"],
            "case_name": case_name,
            "d": TARGET_META[case_name]["d"],
            "K": TARGET_META[case_name]["K"],
            "const_beta": best_const["beta_const"],
            "const_J_VGS": best_const["J_VGS"],
            "const_J_VGS_per_d": best_const["J_VGS_per_d"],
            "const_t_cross": best_const["t_cross_Ahat_08"],
            "pwc8_J_VGS": best_pwc8["J_VGS"],
            "pwc8_J_VGS_per_d": best_pwc8["J_VGS_per_d"],
            "pwc8_t_cross": best_pwc8["t_cross_Ahat_08"],
            "pwc8_betas": best_pwc8["betas"],
            "improvement_J_VGS_per_d": best_const["J_VGS_per_d"] - best_pwc8["J_VGS_per_d"],
        })

    const_df = pd.DataFrame(const_rows).sort_values(["d", "K", "case_name", "beta_const"]).reset_index(drop=True)
    best_const_df = pd.DataFrame(best_const_rows).sort_values(["d", "K", "case_name"]).reset_index(drop=True)
    hier_df = pd.DataFrame(hier_rows).sort_values(["d", "K", "case_name", "L"]).reset_index(drop=True)
    best_pwc8_df = pd.DataFrame(best_pwc8_rows).sort_values(["d", "K", "case_name"]).reset_index(drop=True)
    cmp_df = pd.DataFrame(cmp_rows).sort_values(["d", "K", "case_name"]).reset_index(drop=True)

    return {
        "const_df": const_df,
        "best_const_df": best_const_df,
        "hier_df": hier_df,
        "best_pwc8_df": best_pwc8_df,
        "cmp_df": cmp_df,
        "best_records": best_records,
        "traces": traces,
    }

# %% [markdown]
# ## Family 1: dimension scan with K=9

# %%
DIM_CASES = [name for name, meta in TARGET_META.items() if meta["family"] == "dimension_scan"]
dim_results = run_case_family(DIM_CASES, "dimension_scan")

display(dim_results["cmp_df"])
save_table(dim_results["const_df"], "dimension_scan_constant_beta_modelB_vgs")
save_table(dim_results["best_const_df"], "dimension_scan_best_constant_modelB_vgs")
save_table(dim_results["hier_df"], "dimension_scan_hierarchical_modelB_vgs")
save_table(dim_results["best_pwc8_df"], "dimension_scan_best_pwc8_modelB_vgs")
save_table(dim_results["cmp_df"], "dimension_scan_comparison_modelB_vgs")

# %% [markdown]
# ### Dimension-scan plots

# %%
if "dim_results" not in globals():
    DIM_CASES = [name for name, meta in TARGET_META.items() if meta["family"] == "dimension_scan"]
    dim_results = run_case_family(DIM_CASES, "dimension_scan")

fig, ax = plt.subplots(figsize=(8, 5))
cmp_df = dim_results["cmp_df"].sort_values("d")
ax.plot(cmp_df["d"], cmp_df["const_J_VGS_per_d"], marker="o", linewidth=2, label="best constant")
ax.plot(cmp_df["d"], cmp_df["pwc8_J_VGS_per_d"], marker="s", linewidth=2, label="best PWC-8")
ax.set_xlabel("dimension d")
ax.set_ylabel(r"$J_{\mathrm{VGS}}/d$")
ax.set_title("Model B deterministic dimension scan (K=9)")
ax.legend()
plt.tight_layout()
savefig_all(fig, "fig_dimension_scan_const_vs_pwc8_modelB_vgs")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cmp_df["d"], cmp_df["const_beta"], marker="o", linewidth=2)
ax.set_xlabel("dimension d")
ax.set_ylabel(r"best constant $\beta$")
ax.set_title("Model B deterministic dimension scan: best constant beta")
plt.tight_layout()
savefig_all(fig, "fig_dimension_scan_best_const_beta_modelB_vgs")

for case_name in DIM_CASES:
    d = TARGET_META[case_name]["d"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = dim_results["const_df"][dim_results["const_df"]["case_name"] == case_name].sort_values("beta_const")
    ax.plot(sub["beta_const"], sub["J_VGS_per_d"], marker="o", linewidth=2)
    ax.set_xlabel(r"constant $\beta$")
    ax.set_ylabel(r"$J_{\mathrm{VGS}}/d$")
    ax.set_title(fr"Model B deterministic constant scan: d={d}, K=9")
    plt.tight_layout()
    savefig_all(fig, f"fig_dimension_scan_const_curve_d{d}_modelB_vgs")

for case_name in DIM_CASES:
    d = TARGET_META[case_name]["d"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in LEVELS:
        rec = dim_results["best_records"][case_name][L]["betas"]
        x, y = step_plot_from_betas(rec)
        ax.plot(x, y, linewidth=2, label=f"L={L}")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\beta(t)$")
    ax.set_title(fr"Model B deterministic hierarchy: d={d}, K=9")
    ax.legend()
    plt.tight_layout()
    savefig_all(fig, f"fig_dimension_scan_schedule_d{d}_modelB_vgs")

# %% [markdown]
# ## Family 2: fixed d=8, varying K

# %%
K_CASES = [name for name, meta in TARGET_META.items() if meta["family"] == "mode_scan"]
k_results = run_case_family(K_CASES, "mode_scan")

display(k_results["cmp_df"])
save_table(k_results["const_df"], "mode_scan_constant_beta_modelB_vgs")
save_table(k_results["best_const_df"], "mode_scan_best_constant_modelB_vgs")
save_table(k_results["hier_df"], "mode_scan_hierarchical_modelB_vgs")
save_table(k_results["best_pwc8_df"], "mode_scan_best_pwc8_modelB_vgs")
save_table(k_results["cmp_df"], "mode_scan_comparison_modelB_vgs")

# %% [markdown]
# ### Mode-scan plots at d=8

# %%
if "k_results" not in globals():
    K_CASES = [name for name, meta in TARGET_META.items() if meta["family"] == "mode_scan"]
    k_results = run_case_family(K_CASES, "mode_scan")

fig, ax = plt.subplots(figsize=(8, 5))
cmp_df = k_results["cmp_df"].sort_values("K")
ax.plot(cmp_df["K"], cmp_df["const_J_VGS_per_d"], marker="o", linewidth=2, label="best constant")
ax.plot(cmp_df["K"], cmp_df["pwc8_J_VGS_per_d"], marker="s", linewidth=2, label="best PWC-8")
ax.set_xlabel("number of modes K")
ax.set_ylabel(r"$J_{\mathrm{VGS}}/d$")
ax.set_title("Model B deterministic mode scan at d=8")
ax.legend()
plt.tight_layout()
savefig_all(fig, "fig_mode_scan_const_vs_pwc8_modelB_vgs")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cmp_df["K"], cmp_df["const_beta"], marker="o", linewidth=2)
ax.set_xlabel("number of modes K")
ax.set_ylabel(r"best constant $\beta$")
ax.set_title("Model B deterministic mode scan: best constant beta at d=8")
plt.tight_layout()
savefig_all(fig, "fig_mode_scan_best_const_beta_modelB_vgs")

for case_name in K_CASES:
    Kcur = TARGET_META[case_name]["K"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = k_results["const_df"][k_results["const_df"]["case_name"] == case_name].sort_values("beta_const")
    ax.plot(sub["beta_const"], sub["J_VGS_per_d"], marker="o", linewidth=2)
    ax.set_xlabel(r"constant $\beta$")
    ax.set_ylabel(r"$J_{\mathrm{VGS}}/d$")
    ax.set_title(fr"Model B deterministic constant scan: d=8, K={Kcur}")
    plt.tight_layout()
    savefig_all(fig, f"fig_mode_scan_const_curve_K{Kcur}_modelB_vgs")

for case_name in K_CASES:
    Kcur = TARGET_META[case_name]["K"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in LEVELS:
        rec = k_results["best_records"][case_name][L]["betas"]
        x, y = step_plot_from_betas(rec)
        ax.plot(x, y, linewidth=2, label=f"L={L}")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\beta(t)$")
    ax.set_title(fr"Model B deterministic hierarchy: d=8, K={Kcur}")
    ax.legend()
    plt.tight_layout()
    savefig_all(fig, f"fig_mode_scan_schedule_K{Kcur}_modelB_vgs")

# %% [markdown]
# ## Selected memory-curve comparisons
#
# These are optional support figures, not part of the deterministic objective.

# %%
if "dim_results" not in globals():
    DIM_CASES = [name for name, meta in TARGET_META.items() if meta["family"] == "dimension_scan"]
    dim_results = run_case_family(DIM_CASES, "dimension_scan")

sel_case_dim = "dimscan_d8_K9"
if sel_case_dim in dim_results["best_records"]:
    best_const_row = dim_results["best_const_df"][dim_results["best_const_df"]["case_name"] == sel_case_dim].iloc[0]
    const_beta = float(best_const_row["beta_const"])
    const_mem = dim_results["best_records"][sel_case_dim][1]["const_mem_artifacts"][const_beta]
    pwc8_mem = dim_results["best_records"][sel_case_dim][8]["mem"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(const_mem["times"], const_mem["Ahat"], linewidth=2, linestyle="--", label=fr"best constant $\beta={const_beta:g}$")
    ax.plot(pwc8_mem["times"], pwc8_mem["Ahat"], linewidth=2, label="best PWC-8")
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\hat A(t)$")
    ax.set_title("Selected memory-curve comparison: d=8, K=9")
    ax.legend()
    plt.tight_layout()
    savefig_all(fig, "fig_selected_memory_curve_dimscan_d8_modelB_vgs")

if "k_results" not in globals():
    K_CASES = [name for name, meta in TARGET_META.items() if meta["family"] == "mode_scan"]
    k_results = run_case_family(K_CASES, "mode_scan")

sel_case_k = "kscan_d8_K27"
if sel_case_k in k_results["best_records"]:
    best_const_row = k_results["best_const_df"][k_results["best_const_df"]["case_name"] == sel_case_k].iloc[0]
    const_beta = float(best_const_row["beta_const"])
    const_mem = k_results["best_records"][sel_case_k][1]["const_mem_artifacts"][const_beta]
    pwc8_mem = k_results["best_records"][sel_case_k][8]["mem"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(const_mem["times"], const_mem["Ahat"], linewidth=2, linestyle="--", label=fr"best constant $\beta={const_beta:g}$")
    ax.plot(pwc8_mem["times"], pwc8_mem["Ahat"], linewidth=2, label="best PWC-8")
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\hat A(t)$")
    ax.set_title("Selected memory-curve comparison: d=8, K=27")
    ax.legend()
    plt.tight_layout()
    savefig_all(fig, "fig_selected_memory_curve_kscan_K27_modelB_vgs")

# %% [markdown]
# ## Manifest

# %%
manifest = {
    "repo_root": str(ROOT),
    "notebook": "modelB_vgs_scaling",
    "model": "B",
    "objective": "deterministic_vgs",
    "families": ["dimension_scan", "mode_scan"],
    "dims": DIMS,
    "K_fixed": K_FIXED,
    "K_list": K_LIST,
    "d_fixed": D_FIXED,
    "grid_a": GRID_A,
    "sigma_base": SIGMA_BASE,
    "levels": LEVELS,
    "beta_sweep": BETA_SWEEP,
    "beta_min": BETA_MIN,
    "beta_max": BETA_MAX,
}
save_json(manifest, "manifest_modelB_vgs_scaling")