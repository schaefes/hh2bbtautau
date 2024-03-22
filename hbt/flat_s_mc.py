# encoding: utf-8

from columnflow.util import maybe_import

np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")


"""return the relative mc uncertainty for each step in the array, basically like np cumsum,
but returning the relative mc uncertainty istead of the sum"""
def mc_rel(yields, variances):
    yields = yields[:, ::-1]
    variances = variances[:, ::-1]
    if len(yields) != len(variances):
        raise Exception("Yields not compatible with Shape of Variances.")
    mc_rel_steps = np.zeros_like(yields)
    for i in range(yields.shape[1]):
        y = np.sum(yields[:, :(i + 1)], axis=1)
        v = np.sqrt(np.sum(variances[:, :(i + 1)]**2, axis=1))
        mc_rel_steps[:, -(i + 1)] = np.sqrt(v) / y

    return np.nan_to_num(mc_rel_steps)


# flat s binning that requires at least N from each individual BG per bin
def flat_s_edges_mc_rel(histograms, n_bins, mc_rel_req):
    proc = list(histograms.keys())[0].split('ml_')[-1]
    h = histograms[f"cat_incl_ml_{proc}"]
    backgrounds = [b for b in h.keys() if b not in proc]
    edges = h[proc]['nominal'].axes.edges[0]

    s_counts = h[proc]['nominal'].counts()
    s_variances = h[proc]['nominal'].variances()
    bg_counts = np.zeros((len(backgrounds), len(s_counts)))
    bg_variances = np.zeros((len(backgrounds), len(s_variances)))

    for i, bg in enumerate(backgrounds):
        bg_counts[i, :] += h[bg]['nominal'].counts()
        bg_variances[i, :] += h[bg]['nominal'].variances()

    s = np.sum(s_counts) / (n_bins + .5)
    idx = len(s_counts + 1)

    new_edges_idx = np.zeros(n_bins + 1)
    new_edges_idx[-1] = len(edges) - 1

    # start the cumsum fromn the end of the array
    all_counts = np.vstack((s_counts, bg_counts))
    all_variances = np.vstack((s_variances, bg_variances))
    all_counts_steps = np.cumsum(all_counts[:, ::-1], axis=1)[:, ::-1]
    all_variances_steps = np.cumsum(all_variances[:, ::-1], axis=1)[:, ::-1]
    remaining_rel_uncertainty = np.sqrt(all_variances_steps) / all_counts_steps
    remaining_rel_uncertainty = np.nan_to_num(remaining_rel_uncertainty)

    for i in range(n_bins - 1):

        remaining_s = s_counts[:idx]
        remaining_all_counts_steps = np.cumsum(all_counts[:, :idx][:, ::-1], axis=1)[:, ::-1]
        remaining_all_variances_steps = np.cumsum(all_variances[:, :idx][:, ::-1], axis=1)[:, ::-1]
        remaining_rel_uncertainty = np.sqrt(remaining_all_variances_steps) / remaining_all_counts_steps
        remaining_rel_uncertainty = np.nan_to_num(remaining_rel_uncertainty)
        idx = (np.cumsum(remaining_s) <= (np.sum(remaining_s) - s)).argmin()
        max_rel_uncert = np.max(remaining_rel_uncertainty[:, idx])
        min_rel_uncert = np.min(remaining_rel_uncertainty[:, idx])
        if (max_rel_uncert > mc_rel_req) or (min_rel_uncert <= 0.):
            idxs = np.where((remaining_rel_uncertainty <= mc_rel_req) & (remaining_rel_uncertainty > 0.))
            unique_idxs, unique_counts = np.unique(idxs[1], return_counts=True)
            idx = np.max(unique_idxs[(unique_counts == 4) & (unique_idxs < idx)])
        new_edges_idx[-(i + 2)] = idx

    new_edges_idx = new_edges_idx.astype('int')
    new_edges = edges[new_edges_idx.astype('int')]

    n_mini_bins = (new_edges * (len(edges) - 1))[1:].astype(int)
    n_mini_bins = n_mini_bins - np.concatenate(([0], n_mini_bins[:-1]))

    if (new_edges[0] != 0) and (new_edges[-1] != 1):
        raise Exception("Edge Boundaries are incorrect")

    if np.sum(n_mini_bins) != (len(edges) - 1):
        raise Exception("Sum of n_mini_bins does not match number of mini bins.")

    if 0 in np.diff(new_edges):
        raise Exception("Two neighboring bin edges of flat s are equal.")

    return new_edges, n_mini_bins
