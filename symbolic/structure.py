"""How well fitted profiles reproduce the plane-to-plane spread of showers at fixed energy and zenith (family choice)."""
import numpy as np

R2_GO = 0.75            # per-plane: model residual RMS at most half the conditional shower-to-shower spread
PLANE_FRAC_GO = 0.75    # ... on at least this fraction of the eligible planes
COV_GO = 0.80           # fraction of the off-diagonal conditional covariance reproduced by the fitted profiles
MIN_FINITE_FRAC, MIN_ROWS, MIN_PAIR = 0.5, 50, 30
N_BOOT = 100


def design(log10_e, cos_z):
    l, u = np.asarray(log10_e) - 5.5, np.asarray(cos_z)
    return np.stack([np.ones_like(l), l, l * l, u, u * u, l * u], 1)


def eligible(Y):
    f = np.isfinite(Y)
    return (f.mean(0) >= MIN_FINITE_FRAC) & (f.sum(0) >= MIN_ROWS)


def condition_residuals(Y, D, planes):
    """Per eligible plane, Y minus its least-squares trend in the design D (fitted on that plane's finite rows)."""
    R = np.full(Y.shape, np.nan)
    for k in np.nonzero(planes)[0]:
        m = np.isfinite(Y[:, k])
        beta, *_ = np.linalg.lstsq(D[m], Y[m, k], rcond=None)
        R[m, k] = Y[m, k] - D[m] @ beta
    return R


def pairwise_cov(R, planes):
    """Covariance over eligible planes from pairwise-complete rows (residuals are already centred per plane)."""
    R = R[:, planes]
    M = np.isfinite(R)
    Z = np.where(M, R, 0.0)
    n = M.T.astype(np.float64) @ M
    C = (Z.T @ Z) / np.maximum(n - 1, 1)
    C[n < MIN_PAIR] = np.nan
    return C


def basic_interval(est, boots):
    """[estimate, lo, hi]: the basic bootstrap 68% interval, which corrects the resampling bias of norm-based metrics."""
    q16, q84 = np.percentile(boots, [16, 84])
    return [float(est), float(2 * est - q84), float(2 * est - q16)]


def _offdiag(C):
    return C[~np.eye(len(C), dtype=bool)]


def cov_explained(C_data, C_model):
    """1 - |offdiag(C_data - C_model)| / |offdiag(C_data)|, Frobenius, over entries finite in both."""
    a, b = _offdiag(C_data), _offdiag(C_model)
    m = np.isfinite(a) & np.isfinite(b)
    return float(1.0 - np.linalg.norm(a[m] - b[m]) / np.linalg.norm(a[m]))


def rank_k(C, k):
    """Best rank-k approximation of a symmetric matrix (NaN entries set to zero first)."""
    w, V = np.linalg.eigh(np.nan_to_num(C))
    top = np.argsort(w)[::-1][:k]
    return (V[:, top] * w[top]) @ V[:, top].T


def floor_model(Yhat, Y):
    """Model log energy with -inf (shape vanished on a plane that has energy) raised to 5 below the lowest data value."""
    out = Yhat.copy()
    bad = np.isneginf(out) & np.isfinite(Y)
    out[np.isneginf(out)] = np.nanmin(Y) - 5.0
    return out, int(bad.sum())


def _core(Y, Yhat, D, planes):
    R = condition_residuals(Y, D, planes)
    Rhat = condition_residuals(np.where(np.isfinite(Y), Yhat, np.nan), D, planes)
    return pairwise_cov(R, planes), pairwise_cov(Rhat, planes), R


def summarize(Y, Yhat, log10_e, cos_z, n_par, rng, n_boot=N_BOOT):
    """Fit quality of one population (one dataset, species, class, family). Rows with no fit (NaN model) are dropped."""
    fitted = np.all(np.isfinite(Yhat) | np.isneginf(Yhat), 1) & np.any(np.isfinite(Yhat), 1)
    Y, Yhat, D = Y[fitted], Yhat[fitted], design(log10_e[fitted], cos_z[fitted])
    Yhat, n_floor = floor_model(Yhat, Y)
    planes = eligible(Y)
    C, Cm, R = _core(Y, Yhat, D, planes)
    err = np.where(np.isfinite(Y), Y - Yhat, np.nan)
    r2_cond, r2_total = np.full(Y.shape[1], np.nan), np.full(Y.shape[1], np.nan)
    resid_mean, resid_sd, spread_sd = (np.full(Y.shape[1], np.nan) for _ in range(3))
    for k in np.nonzero(planes)[0]:
        ms = np.nanmean(err[:, k] ** 2)
        r2_cond[k], r2_total[k] = 1.0 - ms / np.nanvar(R[:, k]), 1.0 - ms / np.nanvar(Y[:, k])
        resid_mean[k], resid_sd[k], spread_sd[k] = np.nanmean(err[:, k]), np.nanstd(err[:, k]), np.nanstd(R[:, k])
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(Y), len(Y))
        Cb, Cmb, _ = _core(Y[i], Yhat[i], D[i], planes)
        boots.append(cov_explained(Cb, Cmb))
    frac_r2 = float(np.mean(r2_cond[planes] >= R2_GO)) if planes.any() else float("nan")
    cov = cov_explained(C, Cm)
    rnd = lambda a: [None if not np.isfinite(v) else round(float(v), 4) for v in a]
    return {"n": int(len(Y)), "n_unfit": int((~fitted).sum()), "n_model_floor": n_floor,
            "planes": np.nonzero(planes)[0].tolist(),
            "r2_cond": rnd(r2_cond), "r2_total": rnd(r2_total),
            "resid_mean": rnd(resid_mean), "resid_sd": rnd(resid_sd), "conditional_spread_sd": rnd(spread_sd),
            "r2_cond_median": float(np.nanmedian(r2_cond)) if planes.any() else float("nan"),
            "frac_planes_r2_go": frac_r2,
            "cov_explained": basic_interval(cov, boots),
            "cov_explained_rank_k_reference": cov_explained(C, rank_k(C, n_par)),
            "go_r2": bool(frac_r2 >= PLANE_FRAC_GO), "go_cov": bool(cov >= COV_GO),
            "_C": C, "_Cm": Cm}


def compare_covariance(Y_a, Y_b, log10_e, cos_z, rng, n_boot=N_BOOT):
    """Matched populations a (reference) and b on the same primaries: |C_b - C_a| / |C_a| on conditional residuals of
    the data, per-plane spread ratio and empty-plane fractions. Rows are resampled jointly."""
    D = design(log10_e, cos_z)
    planes = eligible(Y_a) & eligible(Y_b)

    def dist(i):
        Ca = pairwise_cov(condition_residuals(Y_a[i], D[i], planes), planes)
        Cb = pairwise_cov(condition_residuals(Y_b[i], D[i], planes), planes)
        m = np.isfinite(Ca) & np.isfinite(Cb)
        return float(np.linalg.norm((Cb - Ca)[m]) / np.linalg.norm(Ca[m])), Ca, Cb

    d, Ca, Cb = dist(np.arange(len(Y_a)))
    boots = [dist(rng.integers(0, len(Y_a), len(Y_a)))[0] for _ in range(n_boot)]
    sd_ratio = np.full(Y_a.shape[1], np.nan)
    sd_ratio[planes] = np.sqrt(np.diag(Cb) / np.diag(Ca))
    return {"planes": np.nonzero(planes)[0].tolist(), "cov_distance": basic_interval(d, boots),
            "sd_ratio": [None if not np.isfinite(v) else round(float(v), 4) for v in sd_ratio],
            "empty_frac_a": np.round(np.mean(~np.isfinite(Y_a), 0), 4).tolist(),
            "empty_frac_b": np.round(np.mean(~np.isfinite(Y_b), 0), 4).tolist()}
