# coding: utf-8

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import, DotDict


np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")
tf = maybe_import("tensorflow")
shap = maybe_import("shap")


# used later in shap functions
def correlation_matrix(topDS, inp_full, topDS_labels, event_labels, file_name, output):
    # calculate the correlation coefficients
    corrcoef = np.zeros((10, len(event_labels)))
    events_inp = inp_full[:, - len(event_labels):]
    for i, ds_col in enumerate(topDS):
        for inp2_col in range(events_inp.shape[1]):
            corr = np.corrcoef(inp_full[:, ds_col], events_inp[:, inp2_col])
            corr = np.round(corr, decimals=2)
            corrcoef[i, inp2_col] = corr[0, 1]
    figsize = (len(event_labels) + 15, len(topDS) + 10)
    fig2 = plt.figure(figsize=figsize)
    # plot the coefficients in a heatmap
    plt.style.use(mplhep.style.CMS)
    im = plt.imshow(corrcoef, vmin=-1, vmax=1)
    im_ratio = corrcoef.shape[0] / corrcoef.shape[1]
    plt.colorbar(im, fraction=im_ratio * 0.047)
    plt.xticks(np.arange(len(event_labels)), labels=event_labels, rotation=45)
    plt.yticks(np.arange(len(topDS_labels)), labels=topDS_labels)

    # set the grid
    plt.xticks(np.arange(corrcoef.shape[1] + 1) - .5, minor=True)
    plt.yticks(np.arange(corrcoef.shape[0] + 1) - .5, minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=.25)
    # annotate the the heatmap with the values in each cell
    for i in range(len(topDS)):
        for j in range(len(event_labels)):
            plt.annotate(str(corrcoef[i][j]), xy=(j, i), ha='center', va='center', color='black',
                        fontsize=17)
    mplhep.cms.label(llabel="Work in progress", data=False, loc=0)
    output.child(f"Correlation_Coefficients{file_name}.pdf", type="f").dump(fig2, formatter="mpl")

    # Scatter Plot for the leading correlation value
    leading = np.argwhere(corrcoef == corrcoef.max()).flatten()
    ds = inp_full[:, topDS[leading[0]]]
    ev = events_inp[:, leading[1]]
    fig3 = plt.figure()
    plt.style.use(mplhep.style.CMS)
    plt.scatter(ev, ds, color='blue', s=1)
    plt.xlabel(f"{event_labels[leading[1]]} (X)")
    plt.ylabel(f"{topDS_labels[leading[0]]} (Y)")
    title_str = "Scatter Correlation " + r"$\rho_{X,Y}=$" + f"{corrcoef.max()}"
    plt.title(title_str, loc='left')
    mplhep.cms.label(llabel="Work in progress", data=False, loc=2)
    output.child(f"Scatter_Correlation{file_name}.pdf", type="f").dump(fig3, formatter="mpl")


def plot_loss(history, output, classification="categorical") -> None:
    """
    Simple function to create and store a loss plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)
    fig, ax = plt.subplots()
    ax.plot(history["loss"])
    ax.plot(history["val_loss"])
    ax.set(**{
        "ylabel": "Loss",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False)

    output.child("Loss.pdf", type="f").dump(fig, formatter="mpl")


def plot_accuracy(history, output, classification="categorical") -> None:
    """
    Simple function to create and store an accuracy plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    ax.plot(history[f"{classification}_accuracy"])
    ax.plot(history[f"val_{classification}_accuracy"])
    ax.set(**{
        "ylabel": "Accuracy",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False)

    output.child("Accuracy.pdf", type="f").dump(fig, formatter="mpl")


def plot_confusion(
        sorting,
        model: tf.keras.models.Model,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
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

    labels_ext = [proc_inst.label for proc_inst in process_insts] if process_insts else None
    labels = ["$HH_{" + label.split("HH_{")[1].split("}")[0] + "}$" if "HH" in label else label for label in labels_ext]

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    matrix_display = ConfusionMatrixDisplay(confusion, display_labels=labels)
    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)

    ax.set_title(f"{input_type} set, rows normalized", fontsize=32, loc="left")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)
    output.child(f"Confusion_{input_type}_{sorting}.pdf", type="f").dump(fig, formatter="mpl")


def plot_roc_ovr(
        sorting,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvR (one versus rest)
    """
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

    ax.set_title(f"ROC OvR, {input_type} set")
    ax.set_xlabel("Background selection efficiency (FPR)")
    ax.set_ylabel("Signal selection efficiency (TPR)")

    # legend
    labels = [proc_inst.label for proc_inst in process_insts] if process_insts else range(n_classes)
    ax.legend(
        [f"Signal: {labels[i]} (AUC: {auc_score:.4f})" for i, auc_score in enumerate(auc_scores)],
        loc="best",
    )
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

    output.child(f"ROC_ovr_{input_type}_{sorting}.pdf", type="f").dump(fig, formatter="mpl")


def plot_roc_ovr_binary(
        sorting,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvR (one versus rest)
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    auc_scores = []

    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(
        y_true=inputs['target_binary'],
        y_score=inputs['prediction_binary'],
        sample_weight=inputs['weights'],
    )

    auc_scores.append(roc_auc_score(
        inputs['target_binary'], inputs['prediction_binary'],
        average="macro", multi_class="ovr",
    ))

    # create the plot
    ax.plot(fpr, tpr)

    ax.set_title(f"ROC OvR, {input_type} set")
    ax.set_xlabel("Background selection efficiency (FPR)")
    ax.set_ylabel("Signal selection efficiency (TPR)")

    # legend
    # labels = [proc_inst.label for proc_inst in process_insts] if process_insts else range(n_classes)
    ax.legend(
        [f"(AUC: {auc_scores[0]:.4f})"],
        loc="best",
    )
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

    output.child(f"ROC_ovr_{input_type}_{sorting}.pdf", type="f").dump(fig, formatter="mpl")


def plot_output_nodes(
        sorting,
        model: tf.keras.models.Model,
        train: DotDict,
        validation: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
) -> None:
    """
    Function that creates a plot for each ML output node,
    displaying all processes per plot.
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    n_classes = len(train['target'][0])

    colors = ['red', 'blue', 'green', 'orange', 'cyan', 'purple', 'yellow', 'magenta']

    for i in range(n_classes):
        fig, ax = plt.subplots()

        var_title = f"Output node {process_insts[i].label}"

        h = (
            hist.Hist.new
            .StrCat(["train", "validation"], name="type")
            .IntCat([], name="process", growth=True)
            .Reg(20, 0, 1, name=var_title)
            .Weight()
        )
        for input_type, inputs in (("train", train), ("validation", validation)):
            for j in range(n_classes):
                mask = (np.argmax(inputs['target'], axis=1) == j)
                fill_kwargs = {
                    "type": input_type,
                    "process": j,
                    var_title: inputs['prediction'][:, i][mask],
                    "weight": inputs['weights'][mask],
                }
                h.fill(**fill_kwargs)

        plot_kwargs = {
            "ax": ax,
            "label": [proc_inst.label for proc_inst in process_insts],
            "color": colors[:n_classes],
        }

        # dummy legend entries
        plt.hist([], histtype="step", label="Training", color="black")
        plt.hist([], histtype="step", label="Validation (scaled)", linestyle="dotted", color="black")

        # plot training scores
        h[{"type": "train"}].plot1d(**plot_kwargs)

        # legend
        ax.legend(loc="best")

        ax.set(**{
            "ylabel": "Entries",
            "ylim": (0.00001, ax.get_ylim()[1]),
            "xlim": (0, 1),
            # "yscale": 'log',
        })

        # plot validation scores, scaled to train dataset
        scale = h[{"type": "train"}].sum().value / h[{"type": "validation"}].sum().value
        (h[{"type": "validation"}] * scale).plot1d(**plot_kwargs, linestyle="dotted")

        mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=0)
        output.child(f"Node_{process_insts[i].name}_{sorting}.pdf", type="f").dump(fig, formatter="mpl")


def plot_significance(
        sorting,
        model: tf.keras.models.Model,
        train: DotDict,
        validation: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
) -> None:
    plt.style.use(mplhep.style.CMS)

    n_classes = len(train['target'][0])

    store_dict = {}

    for i in range(n_classes):
        fig, ax = plt.subplots()

        for input_type, inputs in (("train", train), ("validation", validation)):
            for j in range(2):
                mask = inputs['target'][:, i] == j
                store_dict[f'node_{process_insts[i].name}_{j}_{input_type}_pred'] = inputs['prediction'][:, i][mask]
                store_dict[f'node_{process_insts[i].name}_{j}_{input_type}_weight'] = inputs['weights'][mask]

        n_bins = 10
        step_size = 1.0 / n_bins
        stop_val = 1.0 + step_size
        bins = np.arange(0.0, stop_val, step_size)
        x_vals = bins[:-1] + step_size / 2
        train_counts_0, train_bins_0 = np.histogram(store_dict[f'node_{process_insts[i].name}_0_train_pred'],
            bins=bins, weights=store_dict[f'node_{process_insts[i].name}_0_train_weight'])
        train_counts_1, train_bins_1 = np.histogram(store_dict[f'node_{process_insts[i].name}_1_train_pred'],
            bins=bins, weights=store_dict[f'node_{process_insts[i].name}_1_train_weight'])
        validation_counts_0, validation_bins_0 = np.histogram(store_dict[f'node_{process_insts[i].name}_0_validation_pred'],
            bins=bins, weights=store_dict[f'node_{process_insts[i].name}_0_validation_weight'])
        validation_counts_1, validation_bins_1 = np.histogram(store_dict[f'node_{process_insts[i].name}_1_validation_pred'],
            bins=bins, weights=store_dict[f'node_{process_insts[i].name}_1_validation_weight'])

        ax.scatter(x_vals, train_counts_1 / np.sqrt(train_counts_0), label="train", color="r")
        ax.scatter(x_vals, validation_counts_1 / np.sqrt(validation_counts_0), label="validation", color="b")
        # title = "$" + process_insts[i].label.split(" ")[2]
        # ax.set_title(f"Significance Node {title}")
        ax.set_ylabel(r"$S/\sqrt{B}$")
        ax.set_xlabel(f"Significance Node {process_insts[i].label}")
        ax.legend(frameon=True)

        mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=0)
        output.child(f"Significance_Node_{process_insts[i].name}_{sorting}.pdf", type="f").dump(fig, formatter="mpl")


def plot_shap_deep_sets(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        latex_dict,
) -> None:

    # names of the features
    event_labels = list(map(latex_dict.get, feature_names[1]))
    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # get the Deep Sets model Output that will be used as Input for the subsequent FF Network
    subset_idx = 150
    deepSets_op_full = model.deepset_network.predict(train['inputs'])
    deepSets_op = deepSets_op_full[:subset_idx]
    deepSets_op2 = deepSets_op_full[-subset_idx:]

    # Get the feature names
    deepSets_features = [f'DeepSets{i+1}' for i in range(deepSets_op.shape[1])]
    features_list = deepSets_features + event_labels

    # calculate shap values
    concat_inp = np.concatenate((deepSets_op, train['inputs2'][:subset_idx]), axis=1)
    concat_inp2 = np.concatenate((deepSets_op2, train['inputs2'][-subset_idx:]), axis=1)
    ff_model = model.feed_forward_network
    explainer = shap.KernelExplainer(ff_model, concat_inp)
    shap_values = explainer.shap_values(concat_inp2)

    # Plot Feature Ranking
    fig1 = plt.figure(figsize=(20, 15))
    shap.summary_plot(shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False, max_display=25)
    plt.title("Feature Importance Ranking")
    output.child("DeepSets_Ranking.pdf", type="f").dump(fig1, formatter="mpl")

    # Begin preparation for the correlation matrix
    shap_values = np.array(shap_values)
    features_list = np.array(features_list)

    # calculate the proper ranking from the shap values
    ranking = np.sum(np.sum(abs(shap_values), axis=1), axis=0)

    # get the indices and sort them in descending order
    idx = np.argsort(ranking)[::-1]

    # get only the indices of the leading DS Nodes and the labels
    idx = idx[idx < len(deepSets_features)]
    topDS_idx = idx[:10]
    topDS_labels = features_list[topDS_idx]

    # get the first two jets of each event and arrange them in an array (N events, 2*N Jet Features)
    jets_12 = train['inputs'][:, :2, :].numpy()
    jets_12 = jets_12.reshape((jets_12.shape[0], -1))
    jet_labels = np.array(list(map(latex_dict.get, feature_names[0]))).astype(str)
    jet_labels_12 = np.concatenate((np.char.add(jet_labels, " 1"), np.char.add(jet_labels, " 2")))

    # inputs for the calculations of the correlation matrix
    inp_events = np.concatenate((deepSets_op_full, train['inputs2']), axis=1)
    inp_jets = np.concatenate((deepSets_op_full, jets_12), axis=1)
    correlation_matrix(topDS_idx, inp_events, topDS_labels, event_labels, "", output)
    correlation_matrix(topDS_idx, inp_jets, topDS_labels, jet_labels_12, "_to_jets_top", output)

    # create correlation matrices for different jet multiplicities
    # get the index of the column that contains the number of jets
    idx_njets = np.argwhere(np.char.find(feature_names[1], 'njets') != -1)[0][0]
    n_jets = train['inputs2'][:, idx_njets]
    for jet_num in [2, 3, 4, 5]:
        # generate mask for the specified jet multiplicity
        mask = np.isin(n_jets, jet_num)
        jets = train['inputs'][mask][:, :jet_num, :].numpy()
        jets = jets.reshape((jets.shape[0], -1))
        inp_jets = np.concatenate((deepSets_op_full[mask], jets), axis=1)
        jet_features = np.tile(jet_labels, jet_num)
        nums = np.repeat(np.arange(1, jet_num + 1), len(feature_names[0])).astype(str)
        labels = np.char.add(jet_features, nums)
        correlation_matrix(topDS_idx, inp_jets, topDS_labels, labels, f"_jets_{jet_num}", output)


def plot_shap_deep_sets_pp(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        features_pairs,
        latex_dict,
) -> None:

    import math
    # shap values for the DeepSets architecture taking jets and pairs and input (working parallel)
    # names of the features
    event_features = feature_names[1]
    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # get the Deep Sets model Output that will be used as Input for the subsequent FF Network
    subset_idx = 150
    # Output of DeepSets for Jets
    deepSets_jets_op_full = model.deepset_jets_network.predict(train['inputs'])
    deepSets_jets_op = deepSets_jets_op_full[:subset_idx]
    deepSets_jets_op2 = deepSets_jets_op_full[-subset_idx:]
    # Output of DeepSets for Pairs
    deepSets_pairs_op_full = model.deepset_pairs_network.predict(train['pairs_inp'])
    deepSets_pairs_op = deepSets_pairs_op_full[:subset_idx]
    deepSets_pairs_op2 = deepSets_pairs_op_full[-subset_idx:]
    deepSets_op_full = np.concatenate((deepSets_jets_op_full, deepSets_pairs_op_full), axis=1)

    inp_full = np.concatenate((deepSets_op_full, train['inputs2']), axis=1)

    deepSets_features_jets = [f'DeepSets Jets{i+1}' for i in range(deepSets_jets_op.shape[1])]
    deepSets_features_pairs = [f'DeepSets Pairs{i+1}' for i in range(deepSets_pairs_op.shape[1])]
    deepSets_features = deepSets_features_jets + deepSets_features_pairs

    # calculate shap values
    concat_inp = np.concatenate((deepSets_jets_op, deepSets_pairs_op, train['inputs2'][:subset_idx]), axis=1)
    concat_inp2 = np.concatenate((deepSets_jets_op2, deepSets_pairs_op2, train['inputs2'][-subset_idx:]), axis=1)
    ff_model = model.feed_forward_network
    explainer = shap.KernelExplainer(ff_model, concat_inp)
    shap_values = explainer.shap_values(concat_inp2)

    # Plot Feature Ranking
    event_labels = list(map(latex_dict.get, event_features))
    features_list = deepSets_features + event_labels
    fig1 = plt.figure(figsize=(20, 15))
    shap.summary_plot(shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False, max_display=25)
    plt.title('Feature Importance Ranking')
    output.child("DeepSetsPP_Ranking.pdf", type="f").dump(fig1, formatter="mpl")

    # Begin preparation for the correlation matrix
    shap_values = np.array(shap_values)
    features_list = np.array(features_list)

    # calculate the proper ranking from the shap values
    ranking = np.sum(np.sum(abs(shap_values), axis=1), axis=0)

    # get the indices and sort them in descending order
    idx = np.argsort(ranking)[::-1]

    # get only the indices of the leading DS Nodes and the labels
    idx = idx[idx < len(deepSets_features)]
    topDS_idx = idx[:10]
    topDS_labels = features_list[topDS_idx]

    # get only the indices of the leading DS jets Nodes and the labels
    idx_jets = idx[idx < len(deepSets_features_jets)]
    topDS_idx_jets = idx_jets[:10]
    topDS_jets_labels = features_list[topDS_idx_jets]

    # get only the indices of the leading DS pairs Nodes and the labels
    idx_pairs = idx[((idx < len(deepSets_features)) & (idx >= len(deepSets_features_jets)))]
    topDS_idx_pairs = idx_pairs[:10]
    topDS_pairs_labels = features_list[topDS_idx_pairs]

    correlation_matrix(topDS_idx, inp_full, topDS_labels, event_labels, "_Event_Features", output)
    correlation_matrix(topDS_idx_jets, inp_full, topDS_jets_labels, event_labels, "_Event_Features_DSJ", output)
    correlation_matrix(topDS_idx_pairs, inp_full, topDS_pairs_labels, event_labels, "_Event_Features_DSP", output)

    # get the first two jets of each event and arrange them in an array (N events, 2*N Jet Features)
    jets_12 = train['inputs'][:, :2, :].numpy()
    jets_12 = jets_12.reshape((jets_12.shape[0], -1))
    jet_labels = np.array(list(map(latex_dict.get, feature_names[0]))).astype(str)
    jet_labels_12 = np.concatenate((np.char.add(jet_labels, " 1"), np.char.add(jet_labels, " 2")))
    inp_jets = np.concatenate((deepSets_op_full, jets_12), axis=1)
    correlation_matrix(topDS_idx_jets, inp_jets, topDS_jets_labels, jet_labels_12, "_to_jets_DSJ", output)

    # get the first pair each event
    pair_1 = train['pairs_inp'][:, 0, :].numpy()
    pair_1 = pair_1.reshape((pair_1.shape[0], -1))
    pair_1_labels = np.char.add(features_pairs, " 1")
    inp_pair = np.concatenate((deepSets_op_full, pair_1), axis=1)
    correlation_matrix(topDS_idx_pairs, inp_pair, topDS_pairs_labels, pair_1_labels, "_to_pair_DSP", output)

    # create correlation matrices for different jet multiplicities
    # get the index of the column that contains the number of jets
    idx_njets = np.argwhere(np.char.find(feature_names[1], 'njets') != -1)[0][0]
    n_jets = train['inputs2'][:, idx_njets]
    for jet_num in [2, 3, 4, 5]:
        # generate mask for the specified jet multiplicity
        pair_num = math.factorial(jet_num) / (2 * math.factorial(jet_num - 2))
        pair_num = int(pair_num)
        mask = np.isin(n_jets, jet_num)
        jets = train['inputs'][mask][:, :jet_num, :].numpy()
        jets = jets.reshape((jets.shape[0], -1))
        pairs = train['pairs_inp'][mask][:, :pair_num, :].numpy()
        pairs = pairs.reshape((pairs.shape[0], -1))
        inp_jets = np.concatenate((deepSets_op_full[mask], jets), axis=1)
        inp_pairs = np.concatenate((deepSets_op_full[mask], pairs), axis=1)
        jet_features = np.tile(jet_labels, jet_num)
        pair_features = np.tile(features_pairs, pair_num)
        nums_jets = np.repeat(np.arange(1, jet_num + 1), len(feature_names[0])).astype(str)
        nums_pairs = np.repeat(np.arange(1, pair_num + 1), len(features_pairs)).astype(str)
        labels_jets = np.char.add(jet_features, nums_jets)
        labels_pairs = np.char.add(pair_features, nums_pairs)
        correlation_matrix(topDS_idx_jets, inp_jets, topDS_jets_labels, labels_jets, f"_jets_{jet_num}", output)
        correlation_matrix(topDS_idx_pairs, inp_pairs, topDS_pairs_labels, labels_pairs, f"_pairs_{pair_num}", output)


def plot_shap_deep_sets_ps(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        features_pairs,
        latex_dict,
) -> None:
    # shap values for the DeepSets architecture taking jets and pairs and input (working parallel)
    # names of the features
    event_features = feature_names[1]
    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # get the Deep Sets model Output that will be used as Input for the subsequent FF Network
    subset_idx = 150
    # Output of DeepSetsPS
    deepSets_op_full = model.deepset_network.predict([train['inputs'], train['pairs_inp']])

    deepSets_features_jets = [f'DeepSets Jets{i+1}' for i in range(deepSets_op_full[0].shape[1])]
    deepSets_features_pairs = [f'DeepSets Pairs{i+1}' for i in range(deepSets_op_full[1].shape[1])]
    deepSets_features = deepSets_features_jets + deepSets_features_pairs

    # calculate shap values
    deepSets_op_full = np.concatenate(deepSets_op_full, axis=1)
    deepSets_op = deepSets_op_full[:subset_idx]
    deepSets_op2 = deepSets_op_full[-subset_idx:]
    concat_inp = np.concatenate((deepSets_op, train['inputs2'][:subset_idx]), axis=1)
    concat_inp2 = np.concatenate((deepSets_op2, train['inputs2'][-subset_idx:]), axis=1)
    ff_model = model.feed_forward_network
    explainer = shap.KernelExplainer(ff_model, concat_inp)
    shap_values = explainer.shap_values(concat_inp2)

    # Plot Feature Ranking
    event_labels = list(map(latex_dict.get, event_features))
    features_list = deepSets_features + event_labels
    fig1 = plt.figure()
    shap.summary_plot(shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False, max_display=25)
    plt.title('Feature Importance Ranking')
    output.child("DeepSetPS_Ranking.pdf", type="f").dump(fig1, formatter="mpl")

    # Begin preparation for the correlation matrix
    shap_values = np.array(shap_values)
    features_list = np.array(features_list)

    # calculate the proper ranking from the shap values
    ranking = np.sum(np.sum(abs(shap_values), axis=1), axis=0)

    # get the indices and sort them in descending order
    idx = np.argsort(ranking)[::-1]

    # get only the indices of the leading DS Nodes and the labels
    idx = idx[idx < len(deepSets_features)]
    topDS_idx = idx[:10]
    topDS_labels = features_list[topDS_idx]

    # get only the indices of the leading DS jets Nodes and the labels
    idx_jets = idx[idx < len(deepSets_features_jets)]
    topDS_idx_jets = idx_jets[:10]
    topDS_jets_labels = features_list[topDS_idx_jets]

    # get only the indices of the leading DS pairs Nodes and the labels
    idx_pairs = idx[((idx < len(deepSets_features)) & (idx >= len(deepSets_features_jets)))]
    topDS_idx_pairs = idx_pairs[:10]
    topDS_pairs_labels = features_list[topDS_idx_pairs]

    inp_events = np.concatenate((deepSets_op_full, train['inputs2']), axis=1)
    correlation_matrix(topDS_idx, inp_events, topDS_labels, event_labels, "_Event_Features", output)

    # get the first two jets of each event and arrange them in an array (N events, 2*N Jet Features)
    jets_12 = train['inputs'][:, :2, :].numpy()
    jets_12 = jets_12.reshape((jets_12.shape[0], -1))
    jet_labels = np.array(list(map(latex_dict.get, feature_names[0]))).astype(str)
    jet_labels_12 = np.concatenate((np.char.add(jet_labels, " 1"), np.char.add(jet_labels, " 2")))
    inp_jets = np.concatenate((deepSets_op_full, jets_12), axis=1)
    correlation_matrix(topDS_idx_jets, inp_jets, topDS_jets_labels, jet_labels_12, "_to_jets_DSJ", output)

    # get the first pair each event
    pair_1 = train['pairs_inp'][:, 0, :].numpy()
    pair_1 = pair_1.reshape((pair_1.shape[0], -1))
    pair_1_labels = np.char.add(features_pairs, " 1")
    inp_pair = np.concatenate((deepSets_op_full, pair_1), axis=1)
    correlation_matrix(topDS_idx_pairs, inp_pair, topDS_pairs_labels, pair_1_labels, "_to_pair_DSP", output)


def plot_shap_baseline(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        features_pairs,
        baseline_jets,
        baseline_pairs,
        model_type,
) -> None:
    feature_names_jets = np.repeat([f"Jet{i} " for i in range(1, baseline_jets + 1)], len(feature_names[0]))
    feature_names_jets = np.char.add(feature_names_jets, np.tile(feature_names[0], baseline_jets)).tolist()
    feature_names_pairs = np.repeat([f"Pair{i} " for i in range(1, baseline_pairs + 1)], len(features_pairs))
    feature_names_pairs = np.char.add(feature_names_pairs, np.tile(features_pairs, baseline_pairs)).tolist()
    features_inputs2 = feature_names[1]

    if model_type == "baseline_pairs":
        features = feature_names_jets + feature_names_pairs + features_inputs2
        inp = train["inputs_baseline_pairs"].numpy()
    else:
        features = feature_names_jets + features_inputs2
        inp = train["inputs_baseline"].numpy()

    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # calculate shap values
    subset_idx = 150
    explainer = shap.KernelExplainer(model, inp[:subset_idx])
    shap_values = explainer.shap_values(inp[-subset_idx:])

    # Feature Ranking
    fig1 = plt.figure()
    shap.summary_plot(shap_values, inp[:subset_idx], plot_type="bar",
        feature_names=features, class_names=class_list, show=False, max_display=25)
    plt.title('Feature Importance Ranking')
    output.child("Feature_Ranking.pdf", type="f").dump(fig1, formatter="mpl")


def plot_feature_ranking_deep_sets(
        masking_model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        sorting,
) -> None:
    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # Shape the Input for the Masking Model
    new_dim = train[f'inputs_{sorting}'].shape[1]
    expanded = tf.expand_dims(train['inputs2'], 1)
    tile_expander = tf.constant([1, new_dim, 1], tf.int32)
    inp_event = tf.tile(expanded, tile_expander)
    inp_deep = train[f'inputs_{sorting}']
    concat_inp = np.concatenate((inp_deep, inp_event), axis=2)

    # Create Model that can be applied for shap values
    inp = tf.keras.layers.Input(shape=[concat_inp.shape[1], concat_inp.shape[2]])
    output = masking_model(inp)
    m = tf.keras.Model(inp, output)
    # calculate shap values
    subset_idx = 50
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer(m, concat_inp[:subset_idx])
    shap_values = explainer.shap_values(concat_inp[-subset_idx:])

    # Plot Feature Ranking
    features_list = feature_names[0] + feature_names[1]
    fig1 = plt.figure()
    shap.summary_plot(shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False, max_display=25)
    plt.title('Feature Importance Ranking')
    output.child("Feature_Ranking_DeepSets.pdf", type="f").dump(fig1, formatter="mpl")


def pca(arr):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=arr.shape[1], svd_solver='full')
    pca.fit(arr)
    pca_values = 100 * pca.explained_variance_ratio_
    return np.array2string(pca_values), np.sum((pca_values > 1))


def PCA(
        model: tf.keras.models.Model,
        train: DotDict,
        model_type,
        output: law.FileSystemDirectoryTarget,
) -> None:

    txt_input = 'PCA Values Jets: \n'
    if model_type == "DeepSets":
        deepSets_op = model.deepset_network.predict(train['inputs'])
        pca_vals, num = pca(deepSets_op)
        txt_input += pca_vals
        txt_input += f'\nNumber of Values above 1%: {num}'
    if model_type == "DeepSetsPP" or model_type == "DeepSetsPS":
        if model_type == "DeepSetsPP":
            deepSets_op_jets = model.deepset_jets_network.predict(train['inputs'])
            deepSets_op_pairs = model.deepset_pairs_network.predict(train['pairs_inp'])
        else:
            deepSets_op_jets, deepSets_op_pairs = model.deepset_network.predict([train['inputs'], train['pairs_inp']])
        pca_vals_jets, num_jets = pca(deepSets_op_jets)
        pca_vals_pairs, num_pairs = pca(deepSets_op_pairs)
        txt_input += pca_vals_jets
        txt_input += f'\nNumber of Values above 1% (Jets): {num_jets}\n'
        txt_input += 'PCA Values Pairs: \n'
        txt_input += pca_vals_pairs
        txt_input += f'\nNumber of Values above 1% (Pairs): {num_pairs}'

    output.child(f'PCA_{model_type}.txt', type="d").dump(txt_input, formatter="text")


def write_info_file(
        output: law.FileSystemDirectoryTarget,
        agg_funcs,
        nodes_deepSets,
        nodes_ff,
        n_output_nodes,
        batch_norm_deepSets,
        batch_norm_ff,
        feature_names,
        process_insts,
        activation_func_deepSets,
        activation_func_ff,
        learningrate,
        ml_proc_weights,
        min_jet_num,
        loss_weights,
        model_type,
        jet_collection,
        phi_projection,
        sequential_mode,
        l2,
        event_to_jet,
) -> None:

    # write info on model for the txt file
    if model_type == "DeepSetsPS":
        txt_input = f'Model Type: {model_type} {sequential_mode}\n'
    else:
        txt_input = f'Model Type: {model_type}\n'
    txt_input = f'Processes: {[process_insts[i].name for i in range(len(process_insts))]}\n'
    txt_input += f'Jet Collection used: {jet_collection}\n'
    txt_input += f'Initial Learning Rate: {learningrate}, Input Handling: Standardization Z-Score \n'
    txt_input += f'Required number of Jets per Event: {min_jet_num + 1}\n'
    txt_input += f'L2: {l2}'
    txt_input += f'Weights used in Loss: {loss_weights.items()}\n'
    if model_type == 'baseline':
        txt_input += f'Input Features Baseline: {feature_names[0]} + {feature_names[1]}\n'
        txt_input += f'Phi Projection relative to {phi_projection}\n'
    if model_type != 'baseline':
        txt_input += 'Deep Sets Architecture:\n'
        txt_input += f'Input Features Deep Sets: {feature_names[0]}\n'
        if event_to_jet:
            txt_input += 'All event Level features added to each jet for DS'
        txt_input += f'Phi Projection relative to {phi_projection}\n'
        txt_input += f'Layers: {len(nodes_deepSets)}, Nodes: {nodes_deepSets}, Activation Function: {activation_func_deepSets}, Batch Norm: {batch_norm_deepSets}\n'
        txt_input += f'Input Features FF: {feature_names[1]}\n'
        txt_input += f'Aggregation Functions: {agg_funcs} \n'
    txt_input += f'{ml_proc_weights}'
    txt_input += 'FF Architecture:\n'
    txt_input += f'Layers: {len(nodes_ff)}, Nodes: {nodes_ff}, Activation Function: {activation_func_ff}, Batch Norm: {batch_norm_ff}\n'

    output.child('model_specifications.txt', type="d").dump(txt_input, formatter="text")


def check_distribution(
        output: law.FileSystemDirectoryTarget,
        input_feature,
        feature_name,
        masking_val,
        input_set,
) -> None:
    plt.style.use(mplhep.style.CMS)
    binning = np.linspace(min(input_feature), max(input_feature), 45)
    mask = (input_feature != masking_val)
    input_feature = input_feature[mask]
    fig, ax = plt.subplots()
    ax.hist(input_feature, bins=binning)
    ax.set_xlabel(f"{feature_name}")
    ax.set_title(f"{input_set} set")

    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=0)
    output.child(f"distribution_{feature_name}_{input_set}.pdf", type="f").dump(fig, formatter="mpl")
