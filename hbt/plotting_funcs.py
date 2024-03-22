# encoding: utf-8

from columnflow.util import maybe_import

np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")
tf = maybe_import("tensorflow")
shap = maybe_import("shap")


def plot_confusion(inputs, labels, save_path, input_set):
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
    fig, ax = plt.subplots(figsize=(15, 10))
    matrix_display = ConfusionMatrixDisplay(confusion, display_labels=labels)
    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)

    ax.set_title(f"{input_set}, rows normalized", fontsize=32, loc="left")
    mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=2)
    file_path = f"{save_path}/confusion_test_set.pdf"
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
            inputs['target'][:, i], inputs['prediction'][:, i],
            average="macro", multi_class="ovr",
        ))

        # create the plot
        ax.plot(fpr, tpr)

    return auc_scores


def plot_roc_ovr(inputs, labels, save_path, input_set, std):
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
            inputs['target'][:, i], inputs['prediction'][:, i],
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
        [f"Signal: {labels[i]} (AUC: {auc_score:.4f} ± {std[i]})" for i, auc_score in enumerate(auc_scores)],
        loc="best",
    )
    mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=2)
    file_path = f"{save_path}/ROC_test_set.pdf"
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


def plot_output_nodes(inputs, processes, labels, save_path, inputs_set):
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

        var_title = f"Output node {labels[i]}"

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
                "weight": inputs['weights'][mask],
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
            "ylabel": "Entries",
            "ylim": (0, 20000),
            "xlim": (0, 1),
            # "yscale": 'log',
        })

        mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=0)
        file_path = f"{save_path}/output_none_{proc}.pdf"
        os.remove(file_path) if os.path.exists(file_path) else None
        plt.savefig(file_path)


def plot_significance(inputs, processes, labels, save_path, inputs_set):

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
        file_path = f"{save_path}/significance_{proc}.pdf"
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


def plot_confusion_std(confusion, std, labels, save_path):
    import os
    import matplotlib.pyplot as plt
    import seaborn as s

    # labels = [f"$HH{label.split('HH')[-1]}" for label in labels]
    # labels = [f"{l.split('}')[0]}{'}'}$" for l in labels]
    labels = ["$HH_{" + label.split("HH_{")[1].split("}")[0] + "}$" if "HH" in label else label for label in labels]

    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots(figsize=(20, 10))

    # creating text array using numpy
    confusion_str = np.round(confusion, decimals=2).astype(str)
    std_str = np.round(std, decimals=2).astype(str)
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
    ax.figure.colorbar(ax.collections[0])

    # save the confusion matrix
    mplhep.cms.label(ax=ax, llabel="Private work", data=False, loc=0)
    file_path = f"{save_path}/confusion_and_std.pdf"
    os.remove(file_path) if os.path.exists(file_path) else None
    plt.savefig(file_path)
