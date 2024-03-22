# encoding: utf-8

from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")


def reverse_cumsum(arr):
    if arr.ndim <= 1:
        arr = arr[::-1]
        cum_sum = np.cumsum(arr)[::-1]
    else:
        arr = arr[:, ::-1]
        cum_sum = np.cumsum(arr, axis=1)[:, ::-1]
    return cum_sum


def bg_bins(histograms, mc_rel_req):
    proc = list(histograms.keys())[0].split('ml_')[-1]
    h = histograms[f"cat_incl_ml_{proc}"]
    processes = [b for b in h.keys() if b not in proc]

    s_counts = h[proc]['nominal'].counts()
    counts = np.zeros((len(processes), len(s_counts)))
    variances = np.zeros((len(processes), len(s_counts)))

    edges = h[proc]['nominal'].axes.edges[0]

    for i, bg in enumerate(processes):
        counts[i, :] += h[bg]['nominal'].counts()
        variances[i, :] += h[bg]['nominal'].variances()

    # start the cumsum from the end of the array
    counts_steps = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1]
    variances_steps = np.cumsum(variances[:, ::-1], axis=1)[:, ::-1]
    remaining_rel_uncertainty = np.sqrt(variances_steps) / counts_steps
    remaining_rel_uncertainty = np.nan_to_num(remaining_rel_uncertainty)

    cond = 0
    new_edges_idx = [len(counts[0])]
    while cond < 1:
        idxs = np.where((remaining_rel_uncertainty < mc_rel_req) & (remaining_rel_uncertainty > 0))
        # if 0 not in idxs[0] or 1 not in idxs[0] or 2 not in idxs[0]:
        unique_idxs, unique_counts = np.unique(idxs[1], return_counts=True)
        if len(processes) not in unique_counts:
            new_edges_idx[-1] = 0
            break
        idx = np.max(unique_idxs[unique_counts == len(processes)])
        # idx_0 = np.max(idxs[1][idxs[0] == 0])
        # idx_1 = np.max(idxs[1][idxs[0] == 1])
        # idx_2 = np.max(idxs[1][idxs[0] == 2])
        # idx = np.min((idx_0, idx_1, idx_2))
        if np.max(remaining_rel_uncertainty[:, idx]) > mc_rel_req:
            print('Crap:', remaining_rel_uncertainty[:, idx], 'max:', np.max(remaining_rel_uncertainty[:, idx]))

        remaining_counts_steps = np.cumsum(counts[:, :idx][:, ::-1], axis=1)[:, ::-1]
        remaining_variances_steps = np.cumsum(variances[:, :idx][:, ::-1], axis=1)[:, ::-1]
        remaining_rel_uncertainty = np.sqrt(remaining_variances_steps) / remaining_counts_steps
        remaining_rel_uncertainty = np.nan_to_num(remaining_rel_uncertainty)
        new_edges_idx.append(idx)

    new_edges_idx = np.array(new_edges_idx[::-1]).astype('int')
    new_edges = edges[new_edges_idx.astype('int')]

    n_mini_bins = new_edges_idx[1:] - new_edges_idx[:-1]

    if (np.sum(n_mini_bins) != len(s_counts)) or (np.sum(n_mini_bins <= 0) != 0):
        raise Exception("n_mini_bins incorrect in bg uncertainty binning!")

    return new_edges, n_mini_bins


# flat s binning that requires at least N from each individual BG per bin
def flat_s_rebin(histograms, n_bins, mc_uncert):
    proc = list(histograms.keys())[0].split('ml_')[-1]
    h = histograms[f"cat_incl_ml_{proc}"]
    processes = [b for b in h.keys()]
    edges = h[proc]['nominal'].axes.edges[0]
    edges_idx = np.arange(0, len(edges), dtype='int')

    s_counts = h[proc]['nominal'].counts()
    counts = np.zeros((len(processes), len(s_counts)))
    variances = np.zeros((len(processes), len(s_counts)))

    for i, p in enumerate(processes):
        counts[i, :] += h[p]['nominal'].counts()
        variances[i, :] += h[p]['nominal'].variances()

    s = np.sum(s_counts) / n_bins

    """Allow a tolerance on s when setting the flat s bin edges, unless this causes empty
     bins at the end. If that happens, redo with a lower tolerance on s."""
    best_n_mini_bins = np.zeros(n_bins)
    best_new_edges = np.zeros(n_bins + 1)
    old_std = np.inf
    for s_tolerance in np.arange(1, 1.2, 0.01):
        new_edges_idx = np.zeros(n_bins + 1)
        new_edges_idx[-1] = len(edges) - 1

        idx = len(s_counts)
        remaining_s = s_counts[:idx]
        remaining_edges_idx = edges_idx[:idx]
        for i in range(2, n_bins + 1):
            remaining_s = remaining_s[:idx]
            remaining_edges_idx = remaining_edges_idx[:idx]
            if np.sum(reverse_cumsum(remaining_s) <= s * s_tolerance) == 0:
                idx = len(remaining_edges_idx) - 1
            else:
                idx = (reverse_cumsum(remaining_s) <= s * s_tolerance).argmax()
            new_edges_idx[-i] = remaining_edges_idx[idx]

        new_edges_idx = new_edges_idx.astype('int')
        new_edges = edges[new_edges_idx.astype('int')]

        n_mini_bins = new_edges_idx[1:] - new_edges_idx[:-1]

        new_s_counts = ak.sum(ak.unflatten(s_counts, n_mini_bins), axis=1)
        s_std = ak.std(new_s_counts)

        new_counts = np.zeros((4, n_bins), dtype='float')
        new_variances = np.zeros((4, n_bins), dtype='float')
        for p in range(4):
            new_counts[p, :] = ak.sum(ak.unflatten(counts[p, :], n_mini_bins), axis=1)
            new_variances[p, :] = ak.sum(ak.unflatten(variances[p, :], n_mini_bins), axis=1)

        rel_uncerts = np.sqrt(new_variances) / new_counts
        rel_uncerts = np.nan_to_num(rel_uncerts)

        if (s_std < old_std) & (np.max(rel_uncerts) <= mc_uncert) & (np.min(rel_uncerts) >= 0):
            best_n_mini_bins = n_mini_bins
            best_new_edges = new_edges
            old_std = s_std

    if (len(best_n_mini_bins) != n_bins) or (np.sum(best_n_mini_bins) != len(s_counts)) or (np.sum(best_n_mini_bins <= 0) != 0):
        raise Exception("best_n_mini_bins incorrect in flat s rebinning!")

    return best_new_edges, best_n_mini_bins


def dummy_binning(histograms):
    proc = list(histograms.keys())[0].split('ml_')[-1]
    h = histograms[f"cat_incl_ml_{proc}"]
    backgrounds = [b for b in h.keys() if b not in proc]

    s_counts = h[proc]['nominal'].counts()
    s_variances = h[proc]['nominal'].variances()
    bg_counts = np.zeros((len(backgrounds), len(s_counts)))
    bg_variances = np.zeros((len(backgrounds), len(s_variances)))

    for i, bg in enumerate(backgrounds):
        bg_counts[i, :] += h[bg]['nominal'].counts()
        bg_variances[i, :] += h[bg]['nominal'].variances()

    edges = h[proc]['nominal'].axes.edges[0]
    print('bg counts:', bg_counts)
    print('bg variances:', bg_variances)

    new_edges = np.delete(edges, np.arange(1, edges.size-1, 2))

    n_mini_bins = (new_edges * (len(edges) - 1))[1:].astype(int)
    n_mini_bins = n_mini_bins - np.concatenate(([0], n_mini_bins[:-1]))

    return new_edges, n_mini_bins
