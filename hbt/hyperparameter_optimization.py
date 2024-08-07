# encoding: utf-8

import numpy as np
import os
import awkward as ak


# t = MLEvaluationWrapper(
#     version="BtagsCustomVBFMaskJets2",
#     ml_model="test",
#     datasets=("graviton_hh_vbf_bbtautau_m400_madgraph",
#              "graviton_hh_ggf_bbtautau_m400_madgraph",
#              "hh_ggf_bbtautau_madgraph"),
#     calibrators=("skip_jecunc",),
#     print_status=(3,),
# )

model_types = ['DeepSets']#, 'DeepSetsPP', 'DeepSetsPS_sum', 'DeepSetsPS_concat', 'DeepSetsPS_two_inp', 'baseline', 'baseline_pairs']
# model_types = ["baseline_pairs"]
random_seeds = [1201, 1337, 1598, 1730, 1922]
file_name = "auc_values_validation.txt"

set_ups = ['setup1', 'setup2', 'setup3', 'setup4']
for t in model_types:
    print(f'-----{t}-----')
    for s in set_ups:
        model_name = f"4classes_{t}_{s}"

        collector = []

        path = f"/nfs/dust/cms/user/schaefes/hbt_store/analysis_hbt/cf.MLTraining/run2_2017_nano_uhh_v11_limited/calib__skip_jecunc/sel__default/prod__default/ml__{model_name}/limits_xsecs"
        models = [m for m in os.listdir(path) if model_name in m]

        for i in models:
            f = open(os.path.join(path, i, file_name), 'r')
            content = f.read()
            content_split = content[1:-3].split(',')
            content_float = [float(c) for c in content_split]
            collector.append(content_float)
            f.close()

        collector = np.array(collector)
        sums = np.sum(collector, axis=1)
        std = np.std(sums)
        mean = np.mean(sums)
        diffs = np.abs(sums - mean).reshape((-1, 5))
        best = np.unravel_index(diffs.argmin(), diffs.shape)
        best_flat = diffs.argmin()
        print(f"mean: {mean}")
        print(f"std: {std}")
        print(f'mlmodel_f{best[0]}of10_4classes_{t}_{s}_{random_seeds[best[1]]}')
        print('best flat: ', best_flat, '\n')
    print('----------')
