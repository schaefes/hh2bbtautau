# encoding: utf-8

from columnflow.util import maybe_import

np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")

# flat s binning that requires at least N from each individual BG per bin
def flat_s_edges_2(histograms, N_bg):
    proc = list(histograms.keys())[0].split('ml_')[-1]
    h = histograms[f"cat_incl_ml_{proc}"]
    backgrounds = [b for b in h.keys() if b not in proc]
    edges = h[proc]['nominal'].axes.edges[0]
    edges_idx = np.linspace(0, len(edges), num=len(edges))

    s_counts = h[proc]['nominal'].counts()
    s_variances = h[proc]['nominal'].variances()
    bg_counts = np.zeros((len(backgrounds), len(s_counts)))
    bg_variances = np.zeros((len(backgrounds), len(s_variances)))

    for i, bg in enumerate(backgrounds):
        bg_counts[i, :] += h[bg]['nominal'].counts()
        bg_variances[i, :] += h[bg]['nominal'].variances()

    # if the counts are below 0 set counts and variances to 0
    bg_counts = np.where(bg_counts < 0, 0, bg_counts)
    bg_variances = np.where(bg_counts < 0, 0, bg_variances)
    n_bg = bg_counts**2 / bg_variances
    n_bg = np.nan_to_num(n_bg)

    n_bins = 20
    n_bg_req = N_bg
    s = np.sum(s_counts) / n_bins
    idx = len(s_counts + 1)

    new_edges_idx = np.zeros(n_bins + 1)
    new_edges_idx[-1] = len(edges) - 1

    for i in range(n_bins - 1):
        remaining_s = s_counts[:idx]
        remaining_n_bg = n_bg[:, :idx]
        idx = (np.cumsum(remaining_s) <= (np.sum(remaining_s) - s)).argmin()
        bg_count = np.sum(remaining_n_bg[:, idx:], axis=1)
        print(bg_count)
        if np.sum(bg_count < n_bg_req):
            threshold_1 = np.repeat((np.sum(remaining_n_bg, axis=1) - n_bg_req).reshape((len(backgrounds), -1)), remaining_n_bg.shape[1], axis=1)
            threshold_2 = np.repeat((np.sum(remaining_n_bg, axis=1) - n_bg_req - 1.).reshape((len(backgrounds), -1)), remaining_n_bg.shape[1], axis=1)
            bg_idx_1 = (np.cumsum(remaining_n_bg, axis=1) <= threshold_1).argmin(axis=1)
            bg_idx_2 = (np.cumsum(remaining_n_bg, axis=1) <= threshold_2).argmin(axis=1)
            idx_new = np.min(np.mean([bg_idx_1, bg_idx_2], axis=0))
            idx = np.floor(idx_new).astype('int')
        new_edges_idx[-(i + 2)] = idx
    if np.sum(n_bg[:idx]) < n_bg_req:
        raise Exception("Number of Background Events in remaining bin too small.")

    new_edges_idx = new_edges_idx.astype('int')
    new_edges = edges[new_edges_idx.astype('int')]

    if new_edges[0] != 0 and new_edges[-1] != 1:
        raise Exception("Edge Boundaries are incorrect")

    new_s_count = np.zeros(n_bins)
    new_n_bg = np.zeros((len(backgrounds), len(new_s_count)))
    for i, (lower, upper) in enumerate(zip(new_edges_idx[:-1], new_edges_idx[1:])):
        new_s_count[i] = np.sum(s_counts[lower:upper])
        new_n_bg[:, i] = np.sum(n_bg[:, lower:upper], axis=1)
    print(new_n_bg)

    if np.min(new_n_bg) < n_bg_req:
        raise Exception(f"Insufficient Background events in Bin {np.argmin(new_n_bg) + 1}")

    return new_edges, proc
