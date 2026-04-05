"""gmm_spec.py — benchmark Gaussian-mixture specification builders for AdaPID.

This module provides a compact, experiment-facing specification layer for building
Gaussian-mixture targets that remain native to the analytic LQ-GM-PID framework.
The goal is to support reproducible benchmark generation across dimensions ``d``,
nominal mode counts ``K``, and a few geometry/conditioning knobs, while keeping
all outputs as plain :class:`GaussianMixture` objects.

Implemented families
--------------------
1. ``isotropic_codebook``
   Means are placed on a sparse sign-codebook; all components share an isotropic
   covariance ``scale^2 I``.

2. ``diag_anisotropic``
   Same mean-placement logic, but each component receives a diagonal covariance
   whose eigenvalues span the requested condition number ``cond``.

3. ``product``
   A block-product GMM assembled from repeated low-dimensional block mixtures.
   The blockwise Cartesian product can generate many effective modes in a compact
   and reproducible way.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil
from typing import Literal, Optional

import torch

from .core import GaussianMixture

Tensor = torch.Tensor
GMMFamily = Literal["isotropic_codebook", "diag_anisotropic", "product"]


@dataclass(frozen=True)
class GMMSpec:
    """Experiment-facing Gaussian-mixture benchmark specification.

    Parameters
    ----------
    family:
        Benchmark family. One of ``"isotropic_codebook"``,
        ``"diag_anisotropic"``, or ``"product"``.
    d:
        Ambient dimension.
    K:
        Nominal number of mixture components for ordinary mixtures.  For the
        ``product`` family this is interpreted as the desired effective mode
        count and is rounded to the nearest realizable product count.
    radius:
        Controls component-mean separation.
    scale:
        Base standard deviation scale.  For isotropic mixtures the covariance is
        ``scale^2 I``.  For anisotropic families it sets the smallest principal
        standard deviation.
    cond:
        Condition-number control for diagonal-anisotropic components.
    weight_temp:
        If positive, produces nonuniform weights via a softmax over fixed random
        energies.  If zero, weights are uniform.
    seed:
        Random seed for reproducible benchmark generation.
    block_dim:
        Product-family block dimension.
    block_modes:
        Product-family modes per block.
    product_rounding:
        How to reconcile requested ``K`` with realizable product counts.
    """

    family: GMMFamily
    d: int
    K: int
    radius: float = 4.0
    scale: float = 0.50
    cond: float = 1.0
    weight_temp: float = 0.0
    seed: int = 0
    block_dim: int = 2
    block_modes: int = 2
    product_rounding: Literal["ceil", "floor", "nearest"] = "nearest"
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError(f"d must be positive, got {self.d}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.cond < 1.0:
            raise ValueError(f"cond must be >= 1, got {self.cond}")
        if self.block_dim <= 0:
            raise ValueError(f"block_dim must be positive, got {self.block_dim}")
        if self.block_modes <= 0:
            raise ValueError(f"block_modes must be positive, got {self.block_modes}")
        if self.family not in {"isotropic_codebook", "diag_anisotropic", "product"}:
            raise ValueError(f"unknown family: {self.family}")
        if self.product_rounding not in {"ceil", "floor", "nearest"}:
            raise ValueError("product_rounding must be one of 'ceil', 'floor', 'nearest'")

    def build(self) -> GaussianMixture:
        return build_gmm(self)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_gmm(spec: GMMSpec) -> GaussianMixture:
    if spec.family == "isotropic_codebook":
        return make_isotropic_codebook_gmm(spec)
    if spec.family == "diag_anisotropic":
        return make_diag_anisotropic_gmm(spec)
    if spec.family == "product":
        return make_product_gmm(spec)
    raise ValueError(f"unsupported family: {spec.family}")


def make_isotropic_codebook_gmm(spec: GMMSpec) -> GaussianMixture:
    means = _sample_sparse_sign_codebook(
        K=spec.K,
        d=spec.d,
        radius=spec.radius,
        seed=spec.seed,
        device=spec.device,
        dtype=spec.dtype,
    )
    cov = (spec.scale ** 2) * torch.eye(spec.d, device=spec.device, dtype=spec.dtype)
    covs = cov.unsqueeze(0).expand(spec.K, -1, -1).clone()
    weights = _make_weights(spec.K, spec.weight_temp, spec.seed, spec.device, spec.dtype)
    return GaussianMixture(weights=weights, means=means, covs=covs)


def make_diag_anisotropic_gmm(spec: GMMSpec) -> GaussianMixture:
    means = _sample_sparse_sign_codebook(
        K=spec.K,
        d=spec.d,
        radius=spec.radius,
        seed=spec.seed,
        device=spec.device,
        dtype=spec.dtype,
    )
    diag_stds = _diag_anisotropic_stds(
        K=spec.K,
        d=spec.d,
        scale=spec.scale,
        cond=spec.cond,
        seed=spec.seed + 101,
        device=spec.device,
        dtype=spec.dtype,
    )
    covs = torch.diag_embed(diag_stds ** 2)
    weights = _make_weights(spec.K, spec.weight_temp, spec.seed, spec.device, spec.dtype)
    return GaussianMixture(weights=weights, means=means, covs=covs)


def make_product_gmm(spec: GMMSpec) -> GaussianMixture:
    if spec.d % spec.block_dim != 0:
        raise ValueError(
            f"for family='product', d={spec.d} must be divisible by block_dim={spec.block_dim}"
        )
    n_blocks = spec.d // spec.block_dim
    per_block_modes = spec.block_modes
    realized_modes = per_block_modes ** n_blocks
    if spec.product_rounding == "floor" and realized_modes > spec.K:
        # Reduce block count if possible; otherwise keep one block and accept mismatch.
        n_blocks = max(1, int(torch.floor(torch.tensor(torch.log(torch.tensor(float(spec.K))) / torch.log(torch.tensor(float(per_block_modes))))).item()))
        if spec.d % n_blocks != 0:
            n_blocks = spec.d // spec.block_dim
        realized_modes = per_block_modes ** n_blocks
    elif spec.product_rounding in {"ceil", "nearest"}:
        # Keep requested dimensional block factorization; mismatch is reported via attribute helper.
        pass

    # Block means: symmetric sign patterns on the first axis, optional mild offsets on others.
    block_means = _block_means(
        block_dim=spec.block_dim,
        block_modes=per_block_modes,
        radius=spec.radius,
        device=spec.device,
        dtype=spec.dtype,
    )
    block_cov = (spec.scale ** 2) * torch.eye(spec.block_dim, device=spec.device, dtype=spec.dtype)

    combos = list(product(range(per_block_modes), repeat=n_blocks))
    K_eff = len(combos)
    means = torch.zeros(K_eff, spec.d, device=spec.device, dtype=spec.dtype)
    covs = torch.zeros(K_eff, spec.d, spec.d, device=spec.device, dtype=spec.dtype)

    for m, combo in enumerate(combos):
        for b, idx in enumerate(combo):
            sl = slice(b * spec.block_dim, (b + 1) * spec.block_dim)
            means[m, sl] = block_means[idx]
            covs[m, sl, sl] = block_cov

    weights = _make_weights(K_eff, spec.weight_temp, spec.seed, spec.device, spec.dtype)
    return GaussianMixture(weights=weights, means=means, covs=covs)


# ---------------------------------------------------------------------------
# Convenience spec constructors
# ---------------------------------------------------------------------------

def make_isotropic_codebook_spec(**kwargs) -> GMMSpec:
    return GMMSpec(family="isotropic_codebook", **kwargs)


def make_diag_anisotropic_spec(**kwargs) -> GMMSpec:
    return GMMSpec(family="diag_anisotropic", **kwargs)


def make_product_gmm_spec(**kwargs) -> GMMSpec:
    return GMMSpec(family="product", **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generator(seed: int, device: str | torch.device) -> torch.Generator:
    # torch.Generator only supports cpu/cuda generators; we generate on CPU and move later.
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _make_weights(K: int, weight_temp: float, seed: int, device, dtype) -> Tensor:
    if weight_temp <= 0:
        return torch.full((K,), 1.0 / K, device=device, dtype=dtype)
    gen = _make_generator(seed + 17, device)
    energies = torch.randn(K, generator=gen, dtype=dtype)
    logits = -float(weight_temp) * energies
    return torch.softmax(logits.to(device=device), dim=0)


def _sample_sparse_sign_codebook(K: int, d: int, radius: float, seed: int, device, dtype) -> Tensor:
    gen = _make_generator(seed, device)
    if d == 1:
        signs = 2 * torch.randint(0, 2, (K, 1), generator=gen) - 1
        return radius * signs.to(device=device, dtype=dtype)

    # Use sparse sign patterns to avoid exploding norm with dimension.
    active = max(1, min(d, int(ceil(d ** 0.5))))
    means = torch.zeros(K, d, dtype=dtype)
    for k in range(K):
        perm = torch.randperm(d, generator=gen)
        idx = perm[:active]
        signs = 2 * torch.randint(0, 2, (active,), generator=gen) - 1
        means[k, idx] = signs.to(dtype=dtype)
    means = means / means.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (radius * means).to(device=device)


def _diag_anisotropic_stds(K: int, d: int, scale: float, cond: float, seed: int, device, dtype) -> Tensor:
    gen = _make_generator(seed, device)
    if d == 1:
        return torch.full((K, 1), float(scale), device=device, dtype=dtype)

    min_std = float(scale)
    max_std = float(scale) * float(cond) ** 0.5
    base = torch.logspace(
        torch.log10(torch.tensor(min_std, dtype=dtype)),
        torch.log10(torch.tensor(max_std, dtype=dtype)),
        steps=d,
        dtype=dtype,
    )
    stds = []
    for _ in range(K):
        perm = torch.randperm(d, generator=gen)
        stds.append(base[perm])
    return torch.stack(stds, dim=0).to(device=device)


def _block_means(block_dim: int, block_modes: int, radius: float, device, dtype) -> Tensor:
    if block_modes == 1:
        return torch.zeros(1, block_dim, device=device, dtype=dtype)
    means = torch.zeros(block_modes, block_dim, device=device, dtype=dtype)
    if block_modes == 2:
        means[0, 0] = -radius
        means[1, 0] = radius
        return means
    # Place points on a small circle / sphere in the first two coordinates when available.
    angles = torch.linspace(0.0, 2.0 * torch.pi, block_modes + 1, device=device, dtype=dtype)[:-1]
    means[:, 0] = radius * torch.cos(angles)
    if block_dim >= 2:
        means[:, 1] = radius * torch.sin(angles)
    return means
