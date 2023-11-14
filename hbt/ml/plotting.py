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
    labels = [label.split("HH_{")[1].split("}")[0] for label in labels_ext]
    labels = ["$HH_{" + label for label in labels]
    labels = [label + "}$" for label in labels]

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    matrix_display = ConfusionMatrixDisplay(confusion, display_labels=labels)
    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)

    ax.set_title(f"{input_type} set, rows normalized", fontsize=32, loc="left")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)
    output.child(f"Confusion_{input_type}_{sorting}.pdf", type="f").dump(fig, formatter="mpl")


def plot_confusion2(
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
        # sample_weight=inputs['weights'],
        normalize="true",
    )

    labels_ext = [proc_inst.label for proc_inst in process_insts] if process_insts else None
    labels = [label.split("HH_{")[1].split("}")[0] for label in labels_ext]
    labels = ["$HH_{" + label for label in labels]
    labels = [label + "}$" for label in labels]

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    matrix_display = ConfusionMatrixDisplay(confusion, display_labels=labels)
    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)

    ax.set_title(f"{input_type} set, rows normalized", fontsize=32, loc="left")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)
    output.child(f"Confusion_{input_type}_{sorting}2.pdf", type="f").dump(fig, formatter="mpl")


def plot_roc_ovr2(
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
            # sample_weight=inputs['weights'],
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

    output.child(f"ROC_ovr_{input_type}_{sorting}2.pdf", type="f").dump(fig, formatter="mpl")


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


def plot_shap_values(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        sorting,
) -> None:

    feature_dict = {
        "mjj": r"$m_{jj}$",
        "mbjetbjet": r"$m_{bb}$",
        "mHH": r"$m_{HH}$",
        "mtautau": r"$m_{\tau\tau}$",
        "jets_max_d_eta": r"max $\Delta \eta$",
        "jets_d_eta_inv_mass": r"$m_{jj, \Delta \eta}$",
        "ht": r"$h_{t}$",
        "n_jets": r"$n_{jets}$"
    }

    # names of features and classes
    feature_list = [feature_dict[feature] for feature in feature_names[1]]
    feature_list.insert(0, 'Deep Sets')

    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # calculate shap values
    inp_deepSets = train['prediction_deepSets'].numpy()
    inp_ff = train['inputs2'].numpy()
    inp = np.concatenate((inp_deepSets, inp_ff), axis=1)
    explainer = shap.KernelExplainer(model, inp[:50])
    shap_values = explainer.shap_values(inp[-50:])

    # Feature Ranking
    fig1 = plt.figure()
    shap.summary_plot(shap_values, inp[:500], plot_type="bar",
        feature_names=feature_list, class_names=class_list)
    output.child("Feature_Ranking.pdf", type="f").dump(fig1, formatter="mpl")

    # Violin Plots
    for i, node in enumerate(class_list):
        fig2 = plt.figure()
        shap.summary_plot(shap_values[i], inp[:100], plot_type="violin",
            feature_names=feature_list, class_names=node)
        output.child(f"Violin_{class_sorted[i]}.pdf", type="f").dump(fig2, formatter="mpl")


def plot_shap_values_deep_sets_mean(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        sorting,
) -> None:

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
    deepSets_op = model.deepset_network.predict(train[f'inputs_{sorting}'][:subset_idx])
    deepSets_op2 = model.deepset_network.predict(train[f'inputs_{sorting}'][-subset_idx:])
    deepSets_features = ['DeepSets']

    # calculate shap values
    concat_inp = np.concatenate((deepSets_op, train['inputs2'][:subset_idx]), axis=1)
    concat_inp2 = np.concatenate((deepSets_op2, train['inputs2'][-subset_idx:]), axis=1)
    ff_model = model.feed_forward_network
    explainer = shap.KernelExplainer(ff_model, concat_inp)
    shap_values = explainer.shap_values(concat_inp2)

    mean_shap_values = []
    for node in shap_values:
        node_values = []
        for values in node:
            mean_deepSets = [np.mean(abs(values[:deepSets_op.shape[1]]))]
            event_values = values[deepSets_op.shape[1]:]
            new_values = np.concatenate((mean_deepSets, event_values))
            node_values.append(new_values)
        mean_shap_values.append(np.array(node_values))

    # Plot Feature Ranking
    features_list = deepSets_features + event_features
    fig1 = plt.figure()
    shap.summary_plot(mean_shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False)
    plt.title('Feature Importance Ranking')
    output.child("DeepSet_Ranking_Mean.pdf", type="f").dump(fig1, formatter="mpl")


def plot_shap_values_deep_sets_sum(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        sorting,
) -> None:

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
    deepSets_op = model.deepset_network.predict(train[f'inputs_{sorting}'][:subset_idx])
    deepSets_op2 = model.deepset_network.predict(train[f'inputs_{sorting}'][-subset_idx:])
    deepSets_features = ['DeepSets']

    # calculate shap values
    concat_inp = np.concatenate((deepSets_op, train['inputs2'][:subset_idx]), axis=1)
    concat_inp2 = np.concatenate((deepSets_op2, train['inputs2'][-subset_idx:]), axis=1)
    ff_model = model.feed_forward_network
    explainer = shap.KernelExplainer(ff_model, concat_inp)
    shap_values = explainer.shap_values(concat_inp2)

    mean_shap_values = []
    for node in shap_values:
        node_values = []
        for values in node:
            mean_deepSets = [np.sum(abs(values[:deepSets_op.shape[1]]))]
            event_values = values[deepSets_op.shape[1]:]
            new_values = np.concatenate((mean_deepSets, event_values))
            node_values.append(new_values)
        mean_shap_values.append(np.array(node_values))

    # Plot Feature Ranking
    features_list = deepSets_features + event_features
    fig1 = plt.figure()
    shap.summary_plot(mean_shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False)
    plt.title('Feature Importance Ranking')
    output.child("DeepSet_Ranking_Sum.pdf", type="f").dump(fig1, formatter="mpl")


def plot_shap_values_deep_sets(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        sorting,
) -> None:

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
    deepSets_op = model.deepset_network.predict(train[f'inputs_{sorting}'][:subset_idx])
    deepSets_op2 = model.deepset_network.predict(train[f'inputs_{sorting}'][-subset_idx:])
    deepSets_features = [f'DeepSets{i+1}' for i in range(deepSets_op.shape[1])]

    # calculate shap values
    concat_inp = np.concatenate((deepSets_op, train['inputs2'][:subset_idx]), axis=1)
    concat_inp2 = np.concatenate((deepSets_op2, train['inputs2'][-subset_idx:]), axis=1)
    ff_model = model.feed_forward_network
    explainer = shap.KernelExplainer(ff_model, concat_inp)
    shap_values = explainer.shap_values(concat_inp2)

    # Plot Feature Ranking
    features_list = deepSets_features + event_features
    fig1 = plt.figure()
    shap.summary_plot(shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False, max_display=50)
    plt.title('Feature Importance Ranking')
    output.child("DeepSet_Ranking.pdf", type="f").dump(fig1, formatter="mpl")


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
    from IPython import embed; embed()
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer(m, concat_inp[:subset_idx])
    shap_values = explainer.shap_values(concat_inp[-subset_idx:])

    # Plot Feature Ranking
    features_list = feature_names[0] + feature_names[1]
    fig1 = plt.figure()
    shap.summary_plot(shap_values, plot_type="bar",
        feature_names=features_list, class_names=class_list, show=False)
    plt.title('Feature Importance Ranking')
    output.child("Feature_Ranking_DeepSets.pdf", type="f").dump(fig1, formatter="mpl")


def plot_shap_baseline(
        model: tf.keras.models.Model,
        train: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        target_dict,
        feature_names,
        baseline_jets,
        sorting,
) -> None:

    feature_dict = {
        "mjj": r"$m_{jj}$",
        "mbjetbjet": r"$m_{bb}$",
        "mHH": r"$m_{HH}$",
        "mtautau": r"$m_{\tau\tau}$",
        "jets_max_d_eta": r"max $\Delta \eta$",
        "jets_d_eta_inv_mass": r"$m_{jj, \Delta \eta}$",
        "ht": r"$h_{t}$",
        "n_jets": r"$n_{jets}$"
    }

    quantity_dict = {
        "pt": r"$p_{T}$",
        "eta": r"$\eta$",
        "phi": r"$\Phi$",
        "mass": "mass",
        "e": "E",
        "btag": "btag"
    }
    feature_names_jets = []
    for i in range(1, baseline_jets + 1):
        feature_list = [feature_name + f"_{i}" for feature_name in feature_names[0]]
        feature_names_jets.append(feature_list)
    feature_names_jets = np.array(feature_names_jets).flatten()
    event_features = np.array(feature_names[1])
    features = np.concatenate((feature_names_jets, event_features))
    features = features.tolist()

    # for name in feature_names_jets:
    #     str_slice_quantity, str_slice_num = name.split("_")[1:]
    #     feature_dict[f"{name}"] = f"Jet {str_slice_num} {quantity_dict[str_slice_quantity]}"

    # # names of features and classes
    # jet_features = [feature_dict[feature] for feature in feature_names_jets]
    # feature_list_2 = [feature_dict[feature] for feature in feature_names[1]]

    # all_features = np.concatenate((jet_features, feature_list_2))

    # make sure class names are sorted correctly in correspondence to their target index
    classes = sorted(target_dict.items(), key=lambda x: x[1])
    class_sorted = np.array(classes)[:, 0]
    class_list = ['empty' for i in range(len(process_insts))]
    for proc in process_insts:
        idx = np.where(class_sorted == proc.name)
        class_list[idx[0][0]] = proc.label

    # calculate shap values
    subset_idx = 150
    inp = train[f'input_jets_baseline_{sorting}']
    explainer = shap.KernelExplainer(model, inp[:subset_idx])
    shap_values = explainer.shap_values(inp[-subset_idx:])

    # Sanity Plots to check represesantation of subset used for shap values
    # jet1_pt = train[f'input_jets_baseline_{sorting}'][:, 0]
    # jet1_pt_partial = jet1_pt[:subset_idx]
    # inv_mass_idx = feature_names[1].index("jets_d_eta_inv_mass")
    # inp2_inv_mass_eta = train['inputs2'][:, inv_mass_idx]
    # inp2_inv_mass_eta_partial = inp2_inv_mass_eta[:subset_idx]

    # Binning and Hist Counts used to normalize
    # n_bins = 20
    # step_pt = (np.max(jet1_pt) - np.min(jet1_pt)) / n_bins
    # step_d_eta = (np.max(inp2_inv_mass_eta) - np.min(inp2_inv_mass_eta)) / n_bins
    # binning_edges_pt = np.arange(np.min(jet1_pt), np.max(jet1_pt), step_pt)
    # binning_edges_d_eta = np.arange(np.min(inp2_inv_mass_eta), np.max(inp2_inv_mass_eta), step_d_eta)

    # counts_pt_full, binning_edges_pt, _ = plt.hist(jet1_pt, bins=binning_edges_pt)
    # counts_pt_sub, binning_edges_pt, _ = plt.hist(jet1_pt_partial, bins=binning_edges_pt)
    # counts_d_eta_full, binning_edges_d_eta, _ = plt.hist(inp2_inv_mass_eta, bins=binning_edges_d_eta)
    # counts_d_eta_sub, binning_edges_d_eta, _ = plt.hist(inp2_inv_mass_eta_partial, bins=binning_edges_d_eta)

    # binning_middle_pt = (binning_edges_pt - step_pt / 2)[1:]
    # binning_middle_d_eta = (binning_edges_d_eta - step_d_eta / 2)[1:]

    # fig_check, axs = plt.subplots(2, 2, sharex=False, sharey=False)
    # fig_check.suptitle('Distributions for Subset and complete Set')

    # axs[0, 0].hist(binning_middle_pt, weights=counts_pt_sub / np.sum(counts_pt_sub), bins=binning_edges_pt)
    # axs[0, 1].hist(binning_middle_pt, weights=counts_pt_full / np.sum(counts_pt_full), bins=binning_edges_pt)

    # axs[1, 0].hist(binning_middle_d_eta, weights=counts_d_eta_sub / np.sum(counts_d_eta_sub), bins=binning_edges_d_eta)
    # axs[1, 1].hist(binning_middle_d_eta, weights=counts_d_eta_full / np.sum(counts_d_eta_full), bins=binning_edges_d_eta)

    # axs[0, 0].set_title(r'Subset')
    # axs[0, 0].set_title(r'Complete Set')
    # axs[0, 0].set(ylabel=r'Jet 1 $p_{T}$')
    # axs[0, 1].set_title(r'Subset')
    # axs[1, 0].set(ylabel=f"{feature_dict['jets_d_eta_inv_mass']}")

    # output.child("Subset_Check.pdf", type="f").dump(fig_check, formatter="mpl")

    # Violin Plots
    # for i, class_node in enumerate(class_list):
    #     fig_vio = plt.figure()
    #     shap.summary_plot(shap_values[i], inp[:subset_idx], plot_type="violin",
    #         feature_names=features, class_names=class_node, show=False)
    #     plt.title(f"Violin for {class_node}")
    #     output.child(f"Violin_{class_sorted[i]}.pdf", type="f").dump(fig_vio, formatter="mpl")

    # fig_vio0 = plt.figure()
    # shap.summary_plot(shap_values[0], inp[:subset_idx], plot_type="violin",
    #     feature_names=all_features, class_names=class_list[0], title=class_list[0])
    # output.child(f"Violin_{class_sorted[0]}.pdf", type="f").dump(fig_vio0, formatter="mpl")

    # fig_vio1 = plt.figure()
    # shap.summary_plot(shap_values[1], inp[:subset_idx], plot_type="violin",
    #     feature_names=all_features, class_names=class_list[1], title=class_list[1])
    # output.child(f"Violin_{class_sorted[1]}.pdf", type="f").dump(fig_vio1, formatter="mpl")

    # fig_vio2 = plt.figure()
    # shap.summary_plot(shap_values[2], inp[:subset_idx], plot_type="violin",
    #     feature_names=all_features, class_names=class_list[2], title=class_list[2])
    # output.child(f"Violin_{class_sorted[2]}.pdf", type="f").dump(fig_vio2, formatter="mpl")

    # Feature Ranking
    fig1 = plt.figure()
    shap.summary_plot(shap_values, inp[:subset_idx], plot_type="bar",
        feature_names=features, class_names=class_list, show=False)
    plt.title('Feature Importance Ranking')
    output.child("Feature_Ranking.pdf", type="f").dump(fig1, formatter="mpl")


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
        empty_overwrite,
        ml_proc_weights,
        min_jet_num,
        loss_weights,
        model_type,
        jet_collection,
        resorting_feature,
        phi_projection,
        train_sorting,
) -> None:

    # write info on model for the txt file
    if model_type == 'baseline':
        txt_input = 'Model Type: Baseline\n'
    else:
        txt_input = f'Model Type: {model_type}\n'
    txt_input = f'Processes: {[process_insts[i].name for i in range(len(process_insts))]}\n'
    txt_input += f'Jet Collection used: {jet_collection}\n'
    txt_input += f'Jet sorting according to {train_sorting}\n'
    txt_input += f'Initial Learning Rate: {learningrate}, Input Handling: Standardization Z-Score \n'
    txt_input += f'Required number of Jets per Event: {min_jet_num + 1}\n'
    txt_input += f'Weights used in Loss: {loss_weights.items()}\n'
    if model_type == 'baseline':
        txt_input += f'Input Features Baseline: {feature_names[0]} + {feature_names[1]}\n'
        txt_input += f'Phi Projection relative to {phi_projection}\n'
    if model_type != 'baseline':
        txt_input += 'Deep Sets Architecture:\n'
        txt_input += f'Input Features Deep Sets: {feature_names[0]}\n'
        txt_input += f'Phi Projection relative to {phi_projection}\n'
        txt_input += f'Layers: {len(nodes_deepSets)}, Nodes: {nodes_deepSets}, Activation Function: {activation_func_deepSets}, Batch Norm: {batch_norm_deepSets}\n'
        txt_input += f'Input Features FF: {feature_names[1]}\n'
        txt_input += f'Aggregation Functions: {agg_funcs} \n'
    txt_input += f'EMPTY_FLOAT overwrite: {empty_overwrite}\n'
    txt_input += f'{ml_proc_weights}'
    txt_input += 'FF Architecture:\n'
    txt_input += f'Layers: {len(nodes_ff)}, Nodes: {nodes_ff}, Activation Function: {activation_func_ff}, Batch Norm: {batch_norm_ff}\n'
    txt_input += f'Resorting Feature for additional Plots: {resorting_feature}'

    output.child('model_specifications.txt', type="d").dump(txt_input, formatter="text")


def check_distribution(
        output: law.FileSystemDirectoryTarget,
        input_feature,
        feature_name,
        masking_val,
) -> None:
    plt.style.use(mplhep.style.CMS)

    binning = np.linspace(-10, 30, 60) if "pt" in feature_name else np.linspace(-15, 15, 50)
    mask = (input_feature != masking_val)
    input_feature = input_feature[mask]
    fig, ax = plt.subplots()
    ax.hist(input_feature, bins=binning)
    ax.set_xlabel(f"{feature_name}")

    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=0)
    output.child(f"distribution_{feature_name}.pdf", type="f").dump(fig, formatter="mpl")
