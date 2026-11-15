"""Evaluate approved VR responses with a *saved* CR parameter ensemble.

This module neither loads a fit nor infers a transfer factor. The caller supplies
the exact card-process partition and all CR/eraScale/RZ/photon/lnN factors. No
observations enter this API. Samples stay separate until the plotting adapter.

Response contract: ``trotasr_vr_background_response_v1``; 148 VR bins, two Nb
groups and eight native recoil bins. A flat cell is ``(vr_bin*2+nb)*8+recoil``.
Sparse endpoint values are absolute, not ratios. A missing cell in an existing
endpoint is zero; an absent sample/nuisance has no shape response.

Interpolation follows Combine's CMSHistFunc QuadLinear + autoMCStats hist-mode
1: normalize endpoints inside each *card process/channel*, add the smoothed
shape changes, crop native histogram bins at 1e-9, and multiply by AsymPow of
the integral ratios. Sources: HiggsAnalysis-CombinedLimit src/CMSHistFunc.cc
(updateCache/smoothStepFunc), python/ShapeTools.py (getPdf/getExtraNorm), and
interface/CombineMathFuncs.h (logKappaForX). This is not shapeN interpolation.
Endpoint sumw2 is retained in the producer payload but is not a second nuisance:
the independent VR MC error uses nominal sumw2, as native CMSHistFunc binErrors.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


SCHEMA = "trotasr_vr_background_response_v1"
N_VR, N_NB, N_RECOIL = 148, 2, 8
N_CELL = N_VR * N_NB * N_RECOIL
Component = Tuple[str, str]
Parameters = Mapping[str, float]
FactorValue = Union[float, Sequence[float], np.ndarray]
FactorProvider = Union[Mapping[Component, FactorValue],
                       Callable[[str, str, Parameters], FactorValue]]


def smooth_step(value):
    """CMSHistFunc's C2-continuous smoothing, with unit smooth region."""
    x = np.asarray(value, dtype=float)
    z = np.clip(x, -1., 1.)
    return .125 * z * (z*z * (3.*z*z - 10.) + 15.)


def asym_pow(theta, low, high):
    """Combine AsymPow, including its |theta| < 0.5 transition."""
    x, lo, hi = np.broadcast_arrays(np.asarray(theta, float),
                                    np.asarray(low, float), np.asarray(high, float))
    if (not all(np.isfinite(a).all() for a in (x, lo, hi))
            or np.any(lo <= 0) or np.any(hi <= 0)):
        raise ValueError("AsymPow requires finite parameters and positive endpoints")
    loglo, loghi = np.log(lo), np.log(hi)
    logk = .5 * (loghi - loglo) + .5 * (loghi + loglo) * smooth_step(2*x)
    result = np.exp(x * logk)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite AsymPow result")
    return result


def photon_recoil_factors(prefit_weights, gnn_logshape):
    """Approved five-recoil projection of four relative GNN coefficients.

    Weights are the fixed prefit gamma+jets (Q*Sgamma) yields for ONE era,
    indexed [recoil, GNN], with shape (5, 5). This does not substitute fitted
    photon yields or normalize jointly across eras.
    """
    w, theta = np.asarray(prefit_weights, float), np.asarray(gnn_logshape, float)
    if (w.shape != (5, 5) or theta.shape != (4,)
            or not np.isfinite(w).all() or not np.isfinite(theta).all()
            or np.any(w < 0) or np.any(w.sum(axis=1) <= 0)):
        raise ValueError("Photon projection requires five supported prefit recoil rows")
    # A common exponential offset cancels between numerator and denominator.
    logs = np.r_[0., theta]
    multipliers = np.exp(logs - logs.max())
    average = (w.sum(axis=0) @ multipliers) / w.sum()
    return (w @ multipliers) / w.sum(axis=1) / average


def _dense_cells(sparse, *, yields_only=False):
    result = np.zeros((N_CELL, 3), dtype=float)
    if not isinstance(sparse, Mapping):
        raise ValueError("Expected a sparse flat-cell mapping")
    seen = set()
    for key, value in sparse.items():
        cell = int(key)
        if str(cell) != str(key) or cell in seen or not 0 <= cell < N_CELL:
            raise ValueError("Invalid or duplicate flat cell: {!r}".format(key))
        seen.add(cell)
        row = np.asarray(value, dtype=float)
        if yields_only and row.ndim == 0:
            row = np.array([row, 0., 0.])
        if row.shape != (3,) or not np.isfinite(row).all() or np.any(row[1:] < 0):
            raise ValueError("Invalid sumw/sumw2/entries at flat cell {}".format(cell))
        result[cell] = row
    return result


def _cell_array(value):
    a = np.asarray(value, dtype=float)
    if a.shape == ():
        a = np.full(N_CELL, float(a))
    elif a.shape == (N_VR, N_NB, N_RECOIL):
        a = a.reshape(N_CELL)
    if a.shape != (N_CELL,) or not np.isfinite(a).all():
        raise ValueError("Factors require a scalar or 2368 finite cell values")
    return a


def _view_weights(eras, selection):
    result = {}
    if set(selection) - set(eras):
        raise ValueError("View names an absent era")
    for era in eras:
        value = np.asarray(selection.get(era, 0.), dtype=float)
        if value.shape == ():
            value = np.full(N_VR, float(value))
        if value.shape != (N_VR,) or not np.isfinite(value).all():
            raise ValueError("View weights require a scalar or 148 finite values per era")
        result[era] = value
    return result


@dataclass
class Evaluation:
    total: np.ndarray
    mc_variance: np.ndarray
    by_era_sample: Dict[Component, np.ndarray]
    mc_variance_by_era_sample: Dict[Component, np.ndarray]
    raw_nominal: np.ndarray
    raw_nominal_by_era_sample: Dict[Component, np.ndarray]
    clipping: Tuple[Mapping[str, Any], ...]

    def view(self, selection):
        """Select eras and/or VR bins without destroying sample identities."""
        weights = _view_weights({e for e, _ in self.by_era_sample}, selection)
        means = {key: y * weights[key[0]] for key, y in self.by_era_sample.items()}
        variances = {key: v * weights[key[0]]**2
                     for key, v in self.mc_variance_by_era_sample.items()}
        return {"total": sum(means.values(), np.zeros(N_VR)),
                "mc_variance": sum(variances.values(), np.zeros(N_VR)),
                "raw_nominal": sum((y*weights[key[0]] for key, y in
                                    self.raw_nominal_by_era_sample.items()), np.zeros(N_VR)),
                "by_era_sample": means}


@dataclass
class _Block:
    name: str
    cells: np.ndarray
    bins: np.ndarray
    active: np.ndarray
    anchors: np.ndarray
    nominal: np.ndarray
    variance: np.ndarray
    # shape name, low-normalized, high-normalized, low-norm, high-norm
    shapes: Tuple[Tuple[str, np.ndarray, np.ndarray, float, float], ...]


class VRResponse:
    """Compiled, explicit card-process response partition.

    ``block_map[(era, sample)]`` maps flat-cell indices to exact process/channel
    names. Include zero cells that belong to a process's native histogram; they
    determine where native underflow cropping applies. Distinct processes MUST
    stay distinct even when they share an alpha parameter. Unassigned cells may
    be omitted only when nominal AND all endpoints have no support there.

    ``extra_shapes[(era,sample)][nuisance]`` contains absolute ``up``/``down``
    sparse cell maps (triplets or yield scalars), for approved RZ/closure etc.
    Extra shapes cannot silently replace a producer endpoint with the same name.
    Factors exclude shape-integral AsymPow factors: this class supplies those.
    They must be constant across cells of one process and one VR bin. Different
    process bins may have different supplied transfer factors.
    """

    def __init__(self, payload, block_map, *, parameter_map=None, extra_shapes=None,
                 allow_incomplete=False):
        axes = payload.get("axes", {})
        edges = np.asarray(axes.get("recoil_edges", []), float)
        if (payload.get("schema") != SCHEMA
                or payload.get("fields") != ["sumw", "sumw2", "entries"]
                or axes.get("vr_bin") != N_VR
                or axes.get("nb_groups") != ["Nb1", "Nb2plus"]
                or edges.shape != (N_RECOIL + 1,) or not np.isfinite(edges).all()
                or not np.all(np.diff(edges) > 0)
                or axes.get("recoil_last_open") is not True):
            raise ValueError("Unrecognized VR response schema/axes")
        if payload.get("status") != "complete" and not allow_incomplete:
            raise ValueError("VR response production is incomplete")
        self.status = payload.get("status")
        self.coverage = payload.get("coverage", {})
        self.source_nominal_reused = payload.get("source_nominal_reused")
        self.parameter_map = dict(parameter_map or {})
        self.blocks = {}
        self.raw_nominal_by_component = {}
        extras = extra_shapes or {}
        for era, samples in payload.get("era_samples", {}).items():
            done_total = self.coverage.get(era)
            if (not isinstance(done_total, (tuple, list)) or len(done_total) != 2
                    or any(type(v) is not int or v < 0 for v in done_total)
                    or done_total[0] > done_total[1]
                    or (self.status == "complete" and done_total[0] != done_total[1])):
                raise ValueError("Invalid production coverage for {}".format(era))
            for sample, record in samples.items():
                key = (era, sample)
                if key not in block_map:
                    raise ValueError("No explicit native process partition for {}".format(key))
                nominal = _dense_cells(record["nominal"])
                shapes = dict(record.get("variations", {}))
                added = extras.get(key, {})
                if set(shapes).intersection(added):
                    raise ValueError("Extra shape would replace a producer nuisance")
                shapes.update(added)
                endpoints = {}
                for name, pair in shapes.items():
                    if set(pair) != {"up", "down"}:
                        raise ValueError("Both absolute endpoints required for " + name)
                    endpoints[name] = {direction: _dense_cells(sparse, yields_only=name in added)
                                       for direction, sparse in pair.items()}
                groups = {}
                assigned = np.zeros(N_CELL, bool)
                for cell, block in block_map[key].items():
                    idx = int(cell)
                    if str(idx) != str(cell) or not 0 <= idx < N_CELL or assigned[idx]:
                        raise ValueError("Invalid or duplicate process-partition cell")
                    if not isinstance(block, str) or not block:
                        raise ValueError("An explicit native process name is required")
                    assigned[idx] = True
                    groups.setdefault(block, []).append(idx)
                supported = np.any(nominal != 0, axis=1)
                for pair in endpoints.values():
                    for a in pair.values():
                        supported |= np.any(a != 0, axis=1)
                if np.any(supported & ~assigned):
                    raise ValueError("Unassigned nominal/endpoint support in {}".format(key))
                compiled = []
                for name, indices in groups.items():
                    cells = np.array(sorted(indices), int)
                    bins = cells // (N_NB * N_RECOIL)
                    active, anchors = np.unique(bins, return_index=True)
                    y = np.bincount(bins, weights=nominal[cells, 0], minlength=N_VR)
                    v = np.bincount(bins, weights=nominal[cells, 1], minlength=N_VR)
                    integral = float(y.sum())
                    shape_terms = []
                    for nuisance, pair in endpoints.items():
                        up = np.bincount(bins, weights=pair["up"][cells, 0], minlength=N_VR)
                        down = np.bincount(bins, weights=pair["down"][cells, 0], minlength=N_VR)
                        if (np.allclose(up, y, rtol=1e-12, atol=1e-15)
                                and np.allclose(down, y, rtol=1e-12, atol=1e-15)):
                            continue
                        if integral <= 0 or up.sum() <= 0 or down.sum() <= 0:
                            raise ValueError("Native mode1 has nonpositive integral: {} {} {}"
                                             .format(key, name, nuisance))
                        lo, hi = float(down.sum()/integral), float(up.sum()/integral)
                        shape_terms.append((nuisance, down/lo, up/hi, lo, hi))
                    if integral <= 0:
                        if np.any(y) or np.any(v):
                            raise ValueError("Unsupported nonpositive native process: {} {}"
                                             .format(key, name))
                        continue  # No process is invented for a completely absent template.
                    compiled.append(_Block(name, cells, bins, active, anchors, y, v, tuple(shape_terms)))
                self.blocks[key] = tuple(compiled)
                self.raw_nominal_by_component[key] = nominal[:, 0].reshape(N_VR, -1).sum(axis=1)
        if not self.blocks or set(extras) - set(self.blocks):
            raise ValueError("Absent response components or extra-shape component mismatch")
        self.eras = tuple(sorted({e for e, _ in self.blocks}))
        self.shape_parameters = tuple(sorted({self.parameter_map.get(n, n)
            for blocks in self.blocks.values() for b in blocks for n, *_ in b.shapes}))

    @classmethod
    def from_payload(cls, payload, block_map, **kwargs):
        return cls(payload, block_map, **kwargs)

    def evaluate(self, parameters: Parameters, *, factors: Optional[FactorProvider] = None):
        """Evaluate every era/sample with one common nuisance-parameter state."""
        if any(n not in parameters or not np.isfinite(parameters[n]) for n in self.shape_parameters):
            raise ValueError("Missing or nonfinite shape parameter; no silent zero initialization")
        means, variances, clipping = {}, {}, []
        for key, blocks in self.blocks.items():
            if factors is None:
                f = np.ones(N_CELL)
            elif callable(factors):
                f = _cell_array(factors(*key, parameters))
            else:
                if key not in factors:
                    raise ValueError("Missing component factor: {}".format(key))
                f = _cell_array(factors[key])
            if np.any(f < 0):
                raise ValueError("Negative transfer/normalization factor is not approved")
            result, variance = np.zeros(N_VR), np.zeros(N_VR)
            for block in blocks:
                factor = np.zeros(N_VR)
                factor[block.active] = f[block.cells[block.anchors]]
                if not np.allclose(f[block.cells], factor[block.bins], rtol=1e-12, atol=1e-15):
                    raise ValueError("Factor varies inside native process/VR bin; use an "
                                     "absolute endpoint before native aggregation: {} {}"
                                     .format(key, block.name))
                raw = block.nominal.copy()
                normalization = 1.
                for nuisance, down, up, lo, hi in block.shapes:
                    x = float(parameters[self.parameter_map.get(nuisance, nuisance)])
                    raw += .5*x*((up-down) + smooth_step(x)*(up+down-2*block.nominal))
                    # ShapeTools skips integral factors when BOTH are within 1e-3.
                    if not (abs(lo-1.) < 1e-3 and abs(hi-1.) < 1e-3):
                        normalization *= float(asym_pow(x, lo, hi))
                cropped = raw.copy()
                cropped[block.active] = np.maximum(raw[block.active], 1e-9)
                delta = (cropped-raw)*normalization*factor
                if np.any(delta):
                    clipping.append({"era": key[0], "sample": key[1], "process": block.name,
                        "bins": np.flatnonzero(delta).tolist(),
                        "yield_delta": float(delta.sum()),
                        "max_abs_delta": float(np.max(np.abs(delta))),
                        "raw_negative_bins": np.flatnonzero(raw < 0).tolist()})
                result += cropped*normalization*factor
                variance += block.variance*(normalization*factor)**2
            if not np.isfinite(result).all() or not np.isfinite(variance).all():
                raise ValueError("Nonfinite response evaluation for {}".format(key))
            means[key], variances[key] = result, variance
        return Evaluation(sum(means.values(), np.zeros(N_VR)),
                          sum(variances.values(), np.zeros(N_VR)), means, variances,
                          sum(self.raw_nominal_by_component.values(), np.zeros(N_VR)),
                          self.raw_nominal_by_component,
                          tuple(clipping))


def correlated_draws(names, central, covariance, *, count=500, seed=12345):
    """Draw the full saved covariance; no diagonalization of physical errors.

    Caller may instead pass ROOT ``fit.randomizePars()`` mappings to propagate.
    No bounds clipping or regularization is performed here. A non-PSD covariance
    is an explicit error, not grounds for silently refitting the saved result.
    """
    names = tuple(names)
    mean, cov = np.asarray(central, float), np.asarray(covariance, float)
    if (not names or len(set(names)) != len(names) or mean.shape != (len(names),)
            or cov.shape != (len(names), len(names)) or count < 2
            or not np.isfinite(mean).all() or not np.isfinite(cov).all()
            or not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-14)):
        raise ValueError("Invalid full covariance or parameter ordering")
    values, vectors = np.linalg.eigh((cov + cov.T)*.5)
    tolerance = 64*np.finfo(float).eps*max(1., float(np.max(np.abs(values))))
    if np.any(values < -tolerance):
        raise ValueError("Saved covariance is not positive semidefinite")
    transform = vectors * np.sqrt(np.maximum(values, 0.))
    samples = mean + np.random.default_rng(seed).standard_normal((count, len(names))) @ transform.T
    return [dict(zip(names, map(float, row))) for row in samples]


def prior_draws(central, *, gaussian=None, poisson=None, count=500, seed=12345):
    """Independent explicit prior draws, with caller-supplied nuisance meanings.

    Gaussian values are (mean, sigma), Poisson values are count means. These are
    parameter coordinates, not rate multipliers. Transform if the saved model
    uses another coordinate; no prior is inferred from its parameter name.
    """
    gaussian, poisson = gaussian or {}, poisson or {}
    if set(gaussian).intersection(poisson) or count < 2:
        raise ValueError("Invalid or overlapping prior definitions")
    rng = np.random.default_rng(seed)
    columns = {}
    for name, pair in gaussian.items():
        mu, sigma = map(float, pair)
        if not np.isfinite([mu, sigma]).all() or sigma < 0:
            raise ValueError("Invalid Gaussian prior")
        columns[name] = rng.normal(mu, sigma, count)
    for name, mean in poisson.items():
        mean = float(mean)
        if not np.isfinite(mean) or mean < 0:
            raise ValueError("Invalid Poisson prior")
        columns[name] = rng.poisson(mean, count)
    return [dict(central, **{n: float(v[i]) for n, v in columns.items()}) for i in range(count)]


def propagate(response: VRResponse, central: Parameters, draws: Iterable[Parameters], *,
              factors: Optional[FactorProvider] = None, views=None):
    """Shared-draw covariance plus independent VR MC variance at central values.

    ``views`` maps view names to ``{era: scalar or mask148}``; default is all
    eras. Each draw is applied once to ALL components. The returned joint
    covariance retains cross-view correlations (including shared VR MC entries).
    Bands are population standard deviations about the draw mean, centered on
    the requested central prediction, matching the saved CR plotting convention.
    Missing parameters in a draw retain the explicit ``central`` value.
    """
    selections = views or {"all": {era: 1. for era in response.eras}}
    weights = {name: _view_weights(response.eras, sel) for name, sel in selections.items()}
    names = tuple(weights)
    central_result = response.evaluate(central, factors=factors)
    rows, clipped_draws = [], 0
    for draw in draws:
        state = dict(central)
        state.update(draw)
        evaluated = response.evaluate(state, factors=factors)
        clipped_draws += bool(evaluated.clipping)
        rows.append(np.concatenate([evaluated.view(weights[n])["total"] for n in names]))
    if len(rows) < 2:
        raise ValueError("At least two shared draws are required")
    values = np.asarray(rows)
    centered = values-values.mean(axis=0)
    systematic = centered.T @ centered / len(rows)
    mc = np.zeros_like(systematic)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            variance = sum((v*weights[left][era]*weights[right][era]
                            for (era, _), v in central_result.mc_variance_by_era_sample.items()),
                           np.zeros(N_VR))
            mc[i*N_VR:(i+1)*N_VR, j*N_VR:(j+1)*N_VR] = np.diag(variance)
    total = systematic + mc
    bands = {}
    for i, name in enumerate(names):
        sl = slice(i*N_VR, (i+1)*N_VR)
        bands[name] = dict(central_result.view(weights[name]),
            draw_mean=values[:, sl].mean(axis=0), covariance=total[sl, sl],
            systematic_covariance=systematic[sl, sl],
            uncertainty=np.sqrt(np.maximum(np.diag(total[sl, sl]), 0.)))
    return {"central": central_result, "views": bands, "view_order": names,
            "joint_covariance": total, "joint_systematic_covariance": systematic,
            "joint_mc_covariance": mc, "draw_count": len(rows),
            "draws_with_native_clipping": clipped_draws,
            "mc_statistics": "independent VR nominal sumw2 at central coefficients; no CR MC copy",
            "covariance_scope": "one shared draw across every era/sample/bin/view",
            "fit_performed": False, "vr_data_fitted": False, "sr_observations_read": False}
