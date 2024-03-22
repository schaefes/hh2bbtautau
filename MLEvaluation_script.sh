#!/bin/bash

while getopts m:w:v: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        w) worker=${OPTARG};;
	v) ver=${OPTARG};;
    esac
done

law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers $worker --dataset graviton_hh_vbf_bbtautau_m400_madgraph --calibrators skip_jecunc
law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers 1 --dataset graviton_hh_ggf_bbtautau_m400_madgraph --calibrators skip_jecunc
law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers 1 --dataset tt_sl_powheg --calibrators skip_jecunc
law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers 1 --dataset tt_dl_powheg --calibrators skip_jecunc
law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers 1 --dataset dy_lep_pt50To100_amcatnlo --calibrators skip_jecunc
law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers 1 --dataset dy_lep_pt100To250_amcatnlo --calibrators skip_jecunc
law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers 1 --dataset dy_lep_pt250To400_amcatnlo --calibrators skip_jecunc
law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers 1 --dataset dy_lep_pt400To650_amcatnlo --calibrators skip_jecunc
law run cf.MLEvaluation --version $ver --ml-model $model --config run2_2017_nano_uhh_v11_limited --workers 1 --dataset dy_lep_pt650_amcatnlo --calibrators skip_jecunc
