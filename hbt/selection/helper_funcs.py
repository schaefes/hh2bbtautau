from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


# Helper Functions to test what array are left after the different VBF object level selections
def mask_to_indices(masking_array):
    # pad the mask with additional false entries
    mask_padded = ak.pad_none(masking_array, max(ak.count, axis=1))
    mask_padded = ak.to_regular(ak.fill_none(mask_padded, False))
    mask_shape = ak.to_numpy(mask_padded).shape

    # Generate indice grid based on the shape of the mask (indices must first be generated in numpy)
    indices = np.zeros(mask_shape)
    for i in range(mask_shape[1]):
        indices[:, i] = i
    ak_indices = ak.from_numpy(indices)

    # apply mask to indices array with ak.mask to preserve array shape and make sure dtype is integer
    indices = ak.mask(ak_indices, mask_padded)
    indices_int = ak.values_astype(indices, np.int64)

    return indices_int


def get_jet_indices_from_pair_mask(events_jet, vbf_mask, local_jetindices, vbf_pair_index):
    # get the two indices referring to jets passing vbf_mask
    # and change them so that they point to jets in the full set, sorted by pt
    vbf_indices_local = ak.concatenate(
        [
            ak.singletons(idx) for idx in
            ak.unzip(ak.firsts(ak.argcombinations(events_jet[vbf_mask], 2, axis=1)[vbf_pair_index]))
        ],
        axis=1,
    )
    vbfjet_indices = local_jetindices[vbf_mask][vbf_indices_local]
    vbfjet_indices = vbfjet_indices[ak.argsort(events_jet[vbfjet_indices].pt, axis=1, ascending=False)]

    return vbfjet_indices


def get_unique_rows(array):
    unique_list = []
    for row in array:
        unique_row_idx = np.unique(row, return_index=True)[1]
        unique_row = [row[index] for index in sorted(unique_row_idx)]
        unique_list.append(unique_row)
    unique_array = ak.Array(unique_list)

    return unique_array
