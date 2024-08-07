# encoding: utf-8

from columnflow.util import maybe_import
from mpl_toolkits.axes_grid1 import make_axes_locatable

np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mpl = maybe_import("matplotlib")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")
tf = maybe_import("tensorflow")
shap = maybe_import("shap")


def plot_confusion(inputs, labels, save_path, input_set, jet_threshold):
    """
    Simple function to create and store a confusion matrix plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import os

    # labels = [f"$HH{label.split('HH')[-1]}" for label in labels]
    labels = ["$HH_{" + label.split("HH_{")[1].split("}")[0] + "}$" if "HH" in label else label for label in labels]

    # Create confusion matrix and normalizes it over predicted (columns)
    confusion = confusion_matrix(
        y_true=np.argmax(inputs['target'], axis=1),
        y_pred=np.argmax(inputs['prediction'], axis=1),
        sample_weight=inputs['weights'],
        normalize="true",
    )

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    matrix_display = ConfusionMatrixDisplay(confusion, display_labels=labels)
    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)

    ax.set_title(f"{input_set}, rows normalized", fontsize=32, loc="left")
    mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=2)
    file_path = f"{save_path}/confusion_test_set_jets{jet_threshold}.pdf"
    os.remove(file_path) if os.path.exists(file_path) else None
    plt.savefig(file_path)


def calculate_confusion(inputs):
    """
    Simple function to create and store a confusion matrix plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Create confusion matrix and normalizes it over predicted (columns)
    confusion = confusion_matrix(
        y_true=np.argmax(inputs['target'], axis=1),
        y_pred=np.argmax(inputs['prediction'], axis=1),
        sample_weight=inputs['weights'],
        normalize="true",
    )

    return confusion


def calculate_auc(inputs):

    from sklearn.metrics import roc_curve, roc_auc_score

    auc_scores = []
    n_classes = len(inputs['target'][0])

    fig, ax = plt.subplots()
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(
            y_true=inputs['target'][:, i],
            y_score=inputs['prediction'][:, i],
            sample_weight=inputs['weights'],
        )

        auc_scores.append(roc_auc_score(
            inputs['target'][:, i], inputs['prediction'][:, i], sample_weight=inputs['weights'],
            average="macro", multi_class="ovr",
        ))

        # create the plot
        ax.plot(fpr, tpr)

    return auc_scores


def plot_roc_ovr(inputs, labels, save_path, input_set, std, jet_threshold):
    """
    Simple function to create and store some ROC plots;
    mode: OvR (one versus rest)
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    import os

    auc_scores = []
    n_classes = len(inputs['target'][0])

    # labels = [f"$HH{label.split('HH')[-1]}" for label in labels]
    # labels = [f"{l.split('}')[0]}{'}'}$" for l in labels]
    labels = ["$HH_{" + label.split("HH_{")[1].split("}")[0] + "}$" if "HH" in label else label for label in labels]

    fig, ax = plt.subplots()
    for i in range(n_classes):
        fpr, tpr, thresholds = roc_curve(
            y_true=inputs['target'][:, i],
            y_score=inputs['prediction'][:, i],
            sample_weight=inputs['weights'],
        )
        auc_scores.append(roc_auc_score(
            inputs['target'][:, i], inputs['prediction'][:, i], sample_weight=inputs['weights'],
            average="macro", multi_class="ovr",
        ))

        # create the plot
        ax.plot(fpr, tpr)
    ax.set_title(f"ROC OvR, {input_set}")
    ax.set_xlabel("Background selection efficiency (FPR)")
    ax.set_ylabel("Signal selection efficiency (TPR)")

    # legend
    std = np.round(std, decimals=4)
    ax.legend(
        [f"{labels[i]} (AUC: {auc_score:.4f} ± {std[i]})" for i, auc_score in enumerate(auc_scores)],
        loc="best",
    )
    print(f'{save_path.split("/")[-1]}: {np.sum(auc_scores)}, {np.sum(std)}')
    mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=2)
    file_path = f"{save_path}/ROC_test_set_jets{jet_threshold}.pdf"
    os.remove(file_path) if os.path.exists(file_path) else None
    plt.savefig(file_path)


# def plot_roc_ovr_binary(
#         sorting,
#         inputs: DotDict,
#         output: law.FileSystemDirectoryTarget,
#         input_type: str,
#         process_insts: tuple[od.Process],
# ) -> None:
#     """
#     Simple function to create and store some ROC plots;
#     mode: OvR (one versus rest)
#     """
#     from sklearn.metrics import roc_curve, roc_auc_score

#     auc_scores = []

#     fig, ax = plt.subplots()
#     fpr, tpr, thresholds = roc_curve(
#         y_true=inputs['target_binary'],
#         y_score=inputs['prediction_binary'],
#         sample_weight=inputs['weights'],
#     )

#     auc_scores.append(roc_auc_score(
#         inputs['target_binary'], inputs['prediction_binary'],
#         average="macro", multi_class="ovr",
#     ))

#     # create the plot
#     ax.plot(fpr, tpr)

#     ax.set_title(f"ROC OvR, {input_type} set")
#     ax.set_xlabel("Background selection efficiency (FPR)")
#     ax.set_ylabel("Signal selection efficiency (TPR)")

#     # legend
#     # labels = [proc_inst.label for proc_inst in process_insts] if process_insts else range(n_classes)
#     ax.legend(
#         [f"(AUC: {auc_scores[0]:.4f})"],
#         loc="best",
#     )
#     mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=2)

#     output.child(f"ROC_ovr_{input_type}_{sorting}.pdf", type="f").dump(fig, formatter="mpl")


def plot_output_nodes(inputs, processes, labels, save_path, inputs_set, jet_threshold):
    """
    Function that creates a plot for each ML output node,
    displaying all processes per plot.
    """
    # use CMS plotting style
    import os

    plt.style.use(mplhep.style.CMS)

    n_classes = len(labels)

    colors = ['red', 'blue', 'green', 'orange', 'cyan', 'purple', 'yellow', 'magenta']
    # labels = [f"$HH{label.split('HH')[-1]}" for label in labels]
    labels = ["$HH_{" + label.split("HH_{")[1].split("}")[0] + "}$" if "HH" in label else label for label in labels]

    for i, proc in enumerate(processes):
        fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(9.0, 8.5))

        var_title = f"Output Node {labels[i]}"

        h = (
            hist.Hist.new
            .StrCat(["test"], name="type")
            .IntCat([], name="process", growth=True)
            .Reg(20, 0, 1, name=var_title)
            .Weight()
        )

        for j in range(n_classes):
            mask = (np.argmax(inputs['target'], axis=1) == j)
            fill_kwargs = {
                "type": "test",
                "process": j,
                var_title: inputs['prediction'][:, i][mask],
                "weight": inputs['weights'][mask] / np.sum(inputs['weights'][mask]),
            }
            h.fill(**fill_kwargs)

        plot_kwargs = {
            "ax": ax,
            "label": labels,
            "color": colors[:n_classes],
        }

        # plot training scores
        h[{"type": "test"}].plot1d(**plot_kwargs)

        # legend
        ax.legend(loc="best", title=f"{inputs_set}")

        ax.set(**{
            "ylabel": "Normalized Counts",
            "ylim": (0, 1),
            "xlim": (0, 1),
            # "yscale": 'log',
        })

        mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=0)
        file_path = f"{save_path}/output_none_{proc}_jets{jet_threshold}.pdf"
        os.remove(file_path) if os.path.exists(file_path) else None
        plt.savefig(file_path)


def plot_significance(inputs, processes, labels, save_path, inputs_set, jet_threshold):

    import os

    plt.style.use(mplhep.style.CMS)

    # labels = [f"$HH{label.split('HH')[-1]}" for label in labels]
    labels = ["$HH_{" + label.split("HH_{")[1].split("}")[0] + "}$" if "HH" in label else label for label in labels]

    store_dict = {}

    for i, proc in enumerate(processes):
        fig, ax = plt.subplots()

        for j in range(2):
            mask = inputs['target'][:, i] == j
            store_dict[f'node_{proc}_{j}_test_pred'] = inputs['prediction'][:, i][mask]
            store_dict[f'node_{proc}_{j}_test_weight'] = inputs['weights'][mask]

        n_bins = 10
        step_size = 1.0 / n_bins
        stop_val = 1.0 + step_size
        bins = np.arange(0.0, stop_val, step_size)
        x_vals = bins[:-1] + step_size / 2
        train_counts_0, train_bins_0 = np.histogram(store_dict[f'node_{proc}_0_test_pred'],
            bins=bins, weights=store_dict[f'node_{proc}_0_test_weight'])
        train_counts_1, train_bins_1 = np.histogram(store_dict[f'node_{proc}_1_test_pred'],
            bins=bins, weights=store_dict[f'node_{proc}_1_test_weight'])

        ax.scatter(x_vals, train_counts_1 / np.sqrt(train_counts_0), label="test", color="r")
        ax.set_ylabel(r"$S/\sqrt{B}$")
        ax.set_xlabel(f"Significance Node {labels[i]}")
        ax.legend(frameon=True, title=f"{inputs_set}")

        mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=0)
        file_path = f"{save_path}/significance_{proc}_jets{jet_threshold}.pdf"
        os.remove(file_path) if os.path.exists(file_path) else None
        plt.savefig(file_path)


def check_distribution(save_path, input, feature, masking_val, fold_idx):

    import os

    plt.style.use(mplhep.style.CMS)
    mask = (input != masking_val)
    input_feature = input[mask]
    fig, ax = plt.subplots()
    binning = np.linspace(-10, 30, 60) if feature=="pT" else np.linspace(-15, 15, 50)
    ax.hist(input_feature, bins=binning)
    ax.set_xlabel(f"{feature}, Test Set Fold {fold_idx}")

    mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=0)
    file_path = f"{save_path}/input_distributon_{feature}_fold{fold_idx}.pdf"
    os.remove(file_path) if os.path.exists(file_path) else None
    plt.savefig(file_path)


# calculate the proper event weights from the normalization weights
# def event_weights(targets, weights):
#     N_events_processes = np.array([len(i) for i in weights])
#     ml_proc_weights = np.max(N_events_processes) / N_events_processes
#     weight_scalar = np.min(N_events_processes / ml_proc_weights)
#     sum_eventweights_proc = np.array([np.sum(i) for i in weights])
#     sample_weights = ak.Array(weights)
#     sample_weights = sample_weights * weight_scalar / sum_eventweights_proc
#     sample_weights = sample_weights * ml_proc_weights
#     sample_weights = ak.to_numpy(ak.flatten(sample_weights))


def event_weights(targets, weights_all):
    from IPython import embed; embed()
    sum_eventweights_proc = np.zeros(targets.shape[1])
    N_events_processes = np.sum(targets, axis=0)
    ml_proc_weights = np.max(N_events_processes) / N_events_processes
    weight_scalar = np.min(N_events_processes / ml_proc_weights)
    for i in range(targets.shape[1]):
        mask = np.where(targets[:, i] == 1, True, False)
        sum_eventweights_proc[i] = np.sum(weights_all[mask])
    scaling_factor = (weight_scalar / sum_eventweights_proc) * ml_proc_weights
    for i in range(targets.shape[1]):
        mask = np.where(targets[:, i] == 1, True, False)
        weights_all = np.where(mask, weights_all * scaling_factor[i], weights_all)

    return weights_all


def norm_weights(targets, weights):
    weights_scaler = np.mean(np.sum(targets, axis=0))
    weights = np.where(weights < 0, 0, weights)
    for i in range(targets.shape[1]):
        mask = np.where(targets[:, i] == 1, True, False)
        weights_sum = np.sum(weights[mask])
        weights = np.where(mask, weights / weights_sum, weights)
    normalized_weights = weights * weights_scaler

    return normalized_weights


def plot_confusion_std(confusion, std, labels, save_path, jet_threshold):
    import os
    import matplotlib.pyplot as plt
    import seaborn as s

    std = std[0]

    # labels = [f"$HH{label.split('HH')[-1]}" for label in labels]
    # labels = [f"{l.split('}')[0]}{'}'}$" for l in labels]
    labels = ["$HH_{" + label.split("HH_{")[1].split("}")[0] + "}$" if "HH" in label else label for label in labels]

    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()

    # creating text array using numpy
    confusion_str = np.round(confusion, decimals=2).astype(str)
    for i, _ in enumerate(confusion.tolist()):
        for j, _ in enumerate(confusion.tolist()[i]):
            confusion_str[i][j] = "{:.2f}".format(confusion[i][j])
    std_str = np.round(std, decimals=2).astype(str)
    for i, _ in enumerate(std.tolist()):
        for j, _ in enumerate(std.tolist()[i]):
            std_str[i][j] = "{:.2f}".format(std[i][j])
    pm = np.full_like(confusion_str, "\n± ")
    annot_1 = np.char.add(confusion_str, pm)
    cell_annotations = np.squeeze(np.char.add(annot_1, std_str))

    # defining heatmap on current axes using seaborn
    ax = s.heatmap(confusion, annot=cell_annotations, fmt="", annot_kws={'color': 'yellow'},
                   xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, square=True,
                   cmap="viridis", linecolor='black', linewidths=0.01, cbar=False)
    ax = s.heatmap(confusion, mask=confusion < 0.4, annot=cell_annotations, fmt="", annot_kws={'color': 'black'},
                   xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, square=True,
                   cmap="viridis", cbar=False, linecolor='black', linewidths=0.01)
    for _, spine in ax.spines.items():
        spine.set(visible=True, lw=2, edgecolor="black")

    plt.yticks(rotation=0)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    im_ratio = confusion.shape[1] / confusion.shape[0]
    ax.figure.colorbar(ax.collections[0], fraction=0.0454 * im_ratio)

    # save the confusion matrix
    mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=0)
    file_path = f"{save_path}/confusion_and_std_jets{jet_threshold}.pdf"
    os.remove(file_path) if os.path.exists(file_path) else None
    plt.tight_layout()
    plt.savefig(file_path)


def plot_shap(npz_path, datacards, mean_idx, latex_dict, model_parse):
    import os
    import shap
    import awkward as ak

    # read shap values and safe in dictionary
    shap_vals = {}
    for proc in datacards:
        shaps_per_proc = []
        path = os. path.join(npz_path, proc, "shap")
        for f in np.sort(os.listdir(path)):
            vals = np.load(os.path.join(path, f))
            shaps_per_proc.append(vals)
        shap_vals[proc] = shaps_per_proc

    fold_shaps = ak.concatenate(list(shap_vals.values()), axis=2)

    class_list = np.load(os.path.join(npz_path, 'tt_sl_powheg', 'class_list.npy'))
    # features_list = np.load(os.path.join(npz_path, 'tt_sl_powheg', 'features_list.npy'))
    features_list = np.load(os.path.join(npz_path, 'features.npy'))
    # if 'baseline_s' in model_parse:
    #     jet_list = [i.split(" ")[0] + " " + latex_dict[i.split(" ")[1]] for i in features_list[:40]]
    #     event_list = [latex_dict[i] for i in features_list[40:]]
    #     features_list = jet_list + event_list
    # if 'baseline_pairs' in model_parse:
    #     jet_list = [i.split(" ")[0] + " " + latex_dict[i.split(" ")[1]] for i in features_list[:40]]
    #     event_list = [latex_dict[i] for i in features_list[-12:]]
    #     pairs_list = features_list[40:-12]
    #     features_list = jet_list + pairs_list.tolist() + event_list
    os.makedirs(os.path.join(npz_path, 'plotted', 'shap_plotted'), exist_ok=True)

    # Plot Feature Ranking
    count_5_j = 0
    count_5_p = 0
    dsp_20 = []
    dsj_20 = []
    for i, s in enumerate(fold_shaps):
        if i != mean_idx:
            continue
        s = ak.to_numpy(s)
        fig1 = plt.figure(figsize=(21, 17))
        shap_values = [s[0], s[1], s[2], s[3]]
        # ranking = np.sum(np.sum(abs(np.array(shap_values)), axis=1), axis=0)
        # idx = np.argsort(ranking)[::-1]
        # idx_count_5_p = np.sum((idx[:5] >= 45) & (idx[:5] <= 89))
        # idx_count_p = np.sum((idx[:20] >= 45) & (idx[:20] <= 89))
        # dsp_20.append(idx_count_p)
        # if idx_count_5_p > 0:
        #     count_5_p += 1
        # idx_count_5_j = np.sum((idx[:5] <= 44))
        # idx_count_j = np.sum((idx[:20] <= 44))
        # dsj_20.append(idx_count_j)
        # if idx_count_5_j > 0:
        #     count_5_j += 1
        # continue
        shap.summary_plot(shap_values, plot_type="bar",
            feature_names=features_list, class_names=class_list, show=False, max_display=20)
        # plt.title('Feature Importance Ranking')
        plt.savefig(os.path.join(npz_path, 'plotted', 'shap_plotted', f'shap_plotted_individual.pdf'))
        break

    # print(count_5_j, np.mean(dsj_20))
    # print(count_5_p, np.sum(np.array(dsp_20) > 5))
    full_shaps = ak.to_numpy(np.concatenate(fold_shaps, axis=1))
    fig1 = plt.figure(figsize=(20, 17))
    shap_values = [full_shaps[0], full_shaps[1], full_shaps[2], full_shaps[3]]
    shap.summary_plot(shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False, max_display=20)
    # plt.title('Feature Importance Ranking')
    plt.savefig(os.path.join(npz_path, 'plotted', 'shap_plotted', 'shap_plotted_full.pdf'))


def correlation_matrix_events(npz_path, datacards, labels_directory_list, dataset_to_proc):
    import os
    import awkward as ak

    ds_labels, features_labels, directory_name, mean_idx = labels_directory_list

    # read shap values and safe in dictionary
    corr_vals = {}
    for i in dataset_to_proc.values():
        corr_vals[i] = []
    for proc in datacards:
        corr_per_proc = []
        path = os. path.join(npz_path, proc, directory_name)
        file_dir = np.sort(os.listdir(path))
        for f in file_dir:
            vals = np.load(os.path.join(path, f))
            corr_per_proc.append(vals)
        proc_name = dataset_to_proc[proc]
        corr_vals[proc_name].append(corr_per_proc)

    for proc_name, corr_proc in corr_vals.items():
        corr_vals[proc_name] = ak.mean(list(corr_proc), axis=0)

    fold_corr = ak.mean(list(corr_vals.values()), axis=0)
    fold_corr = ak.to_numpy(fold_corr)

    features_labels = np.load(os.path.join('/nfs/dust/cms/user/schaefes/models_shap_corrcoef/', f"{features_labels}"))
    ds_labels = np.load(os.path.join('/nfs/dust/cms/user/schaefes/models_shap_corrcoef/', f"{ds_labels}"))
    # features_labels = [r'$p_{T}$ fraction ' + i[-1] if 'None' in i else i for i in features_labels]
    # if 'pair' in ds_labels:
    #     ds_labels = np.load(os.path.join(npz_path, f"{ds_labels}"))
    # else:
    #     ds_labels = np.load(os.path.join(npz_path, 'tt_sl_powheg', f"{ds_labels}"))

    save_path = os.path.join(npz_path, 'plotted', f"{directory_name}_plotted")
    os.makedirs(save_path, exist_ok=True)
    mplhep.style.use("CMS")
    for i, corrcoef in enumerate(fold_corr):
        if i != mean_idx:
            continue
        del_row_idxs = np.arange(corrcoef.shape[0])[np.max(abs(corrcoef), axis=1) < 0.15]
        del_col_idxs = np.arange(corrcoef.shape[1])[np.max(abs(corrcoef), axis=0) < 0.15]

        if del_col_idxs.size:
            corrcoef = np.delete(corrcoef, del_col_idxs, axis=1)
            features = np.delete(features_labels, del_col_idxs)
        if del_row_idxs.size:
            corrcoef = np.delete(corrcoef, del_row_idxs, axis=0)
            ds = np.delete(ds_labels, del_row_idxs)
        if not del_col_idxs.size:
            features = features_labels
        if not del_row_idxs.size:
            ds = ds_labels

        for k, idx in enumerate([[0, 25], [25, -1]]):
            if k == 0:
                ds_i = ds[idx[0]: idx[1]]
                corrcoef_i = corrcoef[idx[0]: idx[1]]
            else:
                ds_i = ds[idx[0]:]
                corrcoef_i = corrcoef[idx[0]:]

            figsize = (len(features) * 1.6, len(ds_i) * 1.1)
            fig2 = plt.figure(figsize=figsize)
            plt.style.use(mplhep.style.CMS)
            mplhep.cms.label(llabel='Private Work', data=False, fontsize=18)
            corrcoef_i_str = np.round(corrcoef_i, decimals=2).astype(str)
            for i, _ in enumerate(corrcoef_i.tolist()):
                for j, _ in enumerate(corrcoef_i.tolist()[i]):
                    corrcoef_i_str[i][j] = "{:.2f}".format(corrcoef_i[i][j])
            # plot the coefficients in a heatmap
            im = plt.imshow(corrcoef_i, vmin=-1, vmax=1)
            im_ratio = corrcoef_i.shape[0] / corrcoef_i.shape[1]
            plt.colorbar(im, aspect=im_ratio * 20, fraction=0.046, pad=0.04)
            plt.xticks(np.arange(len(features)), labels=features, rotation=90)
            plt.yticks(np.arange(len(ds_i)), labels=ds_i)

            # set the grid
            plt.xticks(np.arange(corrcoef_i.shape[1] + 1) - .5, minor=True)
            plt.yticks(np.arange(corrcoef_i.shape[0] + 1) - .5, minor=True)
            plt.grid(which="minor", color="black", linestyle='-', linewidth=.25)

            # annotate the the heatmap with the values in each cell
            for n in range(corrcoef_i.shape[0]):
                for j in range(corrcoef_i.shape[1]):
                    plt.annotate(str(corrcoef_i_str[n][j]), xy=(j, n), ha='center', va='center', color='black',
                                fontsize=21)
            plt.savefig(os.path.join(save_path, f'{directory_name}_individual_{k}_04.pdf'))
        break

    #DeepSets, corrcoef_jets
    # plot mean over all folds
    # full_corr = np.mean(fold_corr, axis=0)

    # del_row_idxs = np.arange(full_corr.shape[0])[np.max(abs(full_corr), axis=1) < 0.15]
    # del_col_idxs = np.arange(full_corr.shape[1])[np.max(abs(full_corr), axis=0) < 0.15]

    # if del_col_idxs.size:
    #     full_corr = np.delete(full_corr, del_col_idxs, axis=1)
    #     features = np.delete(features_labels, del_col_idxs)
    # if del_row_idxs.size:
    #     full_corr = np.delete(full_corr, del_row_idxs, axis=0)
    #     ds = np.delete(ds_labels, del_row_idxs)
    # if not del_col_idxs.size:
    #     features = features_labels
    # if not del_row_idxs.size:
    #     ds = ds_labels

    # figsize = (len(features) + 8.5, len(ds) + 10)
    # fig2 = plt.figure(figsize=figsize)
    # # plot the coefficients in a heatmap
    # full_corr_str = np.round(full_corr, decimals=2).astype(str)
    # for i, _ in enumerate(full_corr.tolist()):
    #     for j, _ in enumerate(full_corr.tolist()[i]):
    #         full_corr_str[i][j] = "{:.2f}".format(full_corr[i][j])
    # plt.style.use(mplhep.style.CMS)
    # im = plt.imshow(full_corr, vmin=-1, vmax=1)
    # im_ratio = full_corr.shape[0] / full_corr.shape[1]
    # plt.colorbar(im, aspect=im_ratio * 20, fraction=0.046, pad=0.04)
    # plt.xticks(np.arange(len(features)), labels=features, rotation=90, fontsize=23)
    # plt.yticks(np.arange(len(ds)), labels=ds, fontsize=23)

    # # set the grid
    # plt.xticks(np.arange(full_corr.shape[1] + 1) - .5, minor=True)
    # plt.yticks(np.arange(full_corr.shape[0] + 1) - .5, minor=True)
    # plt.grid(which="minor", color="black", linestyle='-', linewidth=.25)
    # # annotate the the heatmap with the values in each cell
    # full_corr = np.round(full_corr, 2)
    # for i in range(full_corr.shape[0]):
    #     for j in range(full_corr.shape[1]):
    #         plt.annotate(str(full_corr_str[i][j]), xy=(j, i), ha='center', va='center', color='black', fontsize=24)
    # mplhep.cms.label(llabel="Private Work", data=False, loc=0, fontsize=16)
    # plt.savefig(os.path.join(save_path, f'{directory_name}_full.pdf'))
