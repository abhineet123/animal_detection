<!-- MarkdownTOC -->

- [detrac](#detra_c_)
    - [save_sim_dets       @ detrac](#save_sim_dets___detrac_)
        - [rec_40_prec_40       @ save_sim_dets/detrac](#rec_40_prec_40___save_sim_dets_detrac_)
    - [eval_sim       @ detrac](#eval_sim___detrac_)
- [gram](#gra_m_)
    - [idot       @ gram](#idot___gram_)
    - [no_idot       @ gram](#no_idot___gram_)
- [ctc](#ctc_)
    - [syn_3       @ ctc](#syn_3___ct_c_)
    - [save_sim_dets       @ ctc](#save_sim_dets___ct_c_)
        - [all       @ save_sim_dets/ctc](#all___save_sim_dets_ct_c_)
        - [rec_100_prec_100       @ save_sim_dets/ctc](#rec_100_prec_100___save_sim_dets_ct_c_)
        - [rec_80_prec_80       @ save_sim_dets/ctc](#rec_80_prec_80___save_sim_dets_ct_c_)
        - [rec_60_prec_60       @ save_sim_dets/ctc](#rec_60_prec_60___save_sim_dets_ct_c_)
        - [rec_40_prec_40       @ save_sim_dets/ctc](#rec_40_prec_40___save_sim_dets_ct_c_)
    - [simulated_eval       @ ctc](#simulated_eval___ct_c_)
                - [all_seq       @ simulated_eval/ctc](#all_seq___simulated_eval_ctc_)
- [ctmc](#ctm_c_)
    - [syn_3       @ ctmc](#syn_3___ctmc_)
    - [MOT       @ ctmc](#mot___ctmc_)
        - [assoc_method=0       @ MOT/ctmc](#assoc_method_0___mot_ctmc_)
        - [assoc_method=1       @ MOT/ctmc](#assoc_method_1___mot_ctmc_)
    - [save_sim_dets       @ ctmc](#save_sim_dets___ctmc_)
        - [all       @ save_sim_dets/ctmc](#all___save_sim_dets_ctmc_)
        - [rec_100_prec_100       @ save_sim_dets/ctmc](#rec_100_prec_100___save_sim_dets_ctmc_)
        - [rec_80_prec_80       @ save_sim_dets/ctmc](#rec_80_prec_80___save_sim_dets_ctmc_)
        - [rec_60_prec_60       @ save_sim_dets/ctmc](#rec_60_prec_60___save_sim_dets_ctmc_)
        - [rec_40_prec_40       @ save_sim_dets/ctmc](#rec_40_prec_40___save_sim_dets_ctmc_)
    - [simulated_eval       @ ctmc](#simulated_eval___ctmc_)
                - [all_seq       @ simulated_eval/ctmc](#all_seq___simulated_eval_ctm_c_)
- [mot15](#mot15_)
    - [save_sim_dets       @ mot15](#save_sim_dets___mot1_5_)
        - [rec_40_prec_40       @ save_sim_dets/mot15](#rec_40_prec_40___save_sim_dets_mot1_5_)
    - [frcnn,sdp,dpm       @ mot15](#frcnn_sdp_dpm___mot1_5_)
        - [save_sim_dets       @ frcnn,sdp,dpm/mot15](#save_sim_dets___frcnn_sdp_dpm_mot1_5_)
            - [all       @ save_sim_dets/frcnn,sdp,dpm/mot15](#all___save_sim_dets_frcnn_sdp_dpm_mot1_5_)
- [mot17](#mot17_)
    - [frcnn       @ mot17](#frcnn___mot1_7_)
        - [save_sim_dets       @ frcnn/mot17](#save_sim_dets___frcnn_mot1_7_)
            - [rec_79_prec_47       @ save_sim_dets/frcnn/mot17](#rec_79_prec_47___save_sim_dets_frcnn_mot1_7_)
            - [rec_50_prec_50       @ save_sim_dets/frcnn/mot17](#rec_50_prec_50___save_sim_dets_frcnn_mot1_7_)
    - [frcnn,sdp,dpm       @ mot17](#frcnn_sdp_dpm___mot1_7_)
        - [save_sim_dets       @ frcnn,sdp,dpm/mot17](#save_sim_dets___frcnn_sdp_dpm_mot1_7_)
            - [all       @ save_sim_dets/frcnn,sdp,dpm/mot17](#all___save_sim_dets_frcnn_sdp_dpm_mot1_7_)
                - [all_seq       @ all/save_sim_dets/frcnn,sdp,dpm/mot17](#all_seq___all_save_sim_dets_frcnn_sdp_dpm_mot1_7_)
        - [rec_40_prec_40       @ frcnn,sdp,dpm/mot17](#rec_40_prec_40___frcnn_sdp_dpm_mot1_7_)
            - [recs_100       @ rec_40_prec_40/frcnn,sdp,dpm/mot17](#recs_100___rec_40_prec_40_frcnn_sdp_dpm_mot17_)
            - [rec_50_prec_50       @ rec_40_prec_40/frcnn,sdp,dpm/mot17](#rec_50_prec_50___rec_40_prec_40_frcnn_sdp_dpm_mot17_)
            - [rec_79_prec_37       @ rec_40_prec_40/frcnn,sdp,dpm/mot17](#rec_79_prec_37___rec_40_prec_40_frcnn_sdp_dpm_mot17_)
            - [rec_50_prec_60       @ rec_40_prec_40/frcnn,sdp,dpm/mot17](#rec_50_prec_60___rec_40_prec_40_frcnn_sdp_dpm_mot17_)
            - [rec_50_prec_70       @ rec_40_prec_40/frcnn,sdp,dpm/mot17](#rec_50_prec_70___rec_40_prec_40_frcnn_sdp_dpm_mot17_)
            - [rec_70_prec_60       @ rec_40_prec_40/frcnn,sdp,dpm/mot17](#rec_70_prec_60___rec_40_prec_40_frcnn_sdp_dpm_mot17_)
            - [rec_60_prec_70       @ rec_40_prec_40/frcnn,sdp,dpm/mot17](#rec_60_prec_70___rec_40_prec_40_frcnn_sdp_dpm_mot17_)
            - [rec_90_prec_100       @ rec_40_prec_40/frcnn,sdp,dpm/mot17](#rec_90_prec_100___rec_40_prec_40_frcnn_sdp_dpm_mot17_)
        - [simulated_eval       @ frcnn,sdp,dpm/mot17](#simulated_eval___frcnn_sdp_dpm_mot1_7_)
            - [eval_sim       @ simulated_eval/frcnn,sdp,dpm/mot17](#eval_sim___simulated_eval_frcnn_sdp_dpm_mot17_)
                - [all_seq       @ eval_sim/simulated_eval/frcnn,sdp,dpm/mot17](#all_seq___eval_sim_simulated_eval_frcnn_sdp_dpm_mot1_7_)
            - [rec_50_prec_70       @ simulated_eval/frcnn,sdp,dpm/mot17](#rec_50_prec_70___simulated_eval_frcnn_sdp_dpm_mot17_)
            - [rec_70_prec_60       @ simulated_eval/frcnn,sdp,dpm/mot17](#rec_70_prec_60___simulated_eval_frcnn_sdp_dpm_mot17_)
            - [rec_60_prec_70       @ simulated_eval/frcnn,sdp,dpm/mot17](#rec_60_prec_70___simulated_eval_frcnn_sdp_dpm_mot17_)
            - [rec_79_prec_47       @ simulated_eval/frcnn,sdp,dpm/mot17](#rec_79_prec_47___simulated_eval_frcnn_sdp_dpm_mot17_)
            - [rec_79_prec_37       @ simulated_eval/frcnn,sdp,dpm/mot17](#rec_79_prec_37___simulated_eval_frcnn_sdp_dpm_mot17_)
            - [rec_50_prec_90       @ simulated_eval/frcnn,sdp,dpm/mot17](#rec_50_prec_90___simulated_eval_frcnn_sdp_dpm_mot17_)
            - [rec_80_prec_80       @ simulated_eval/frcnn,sdp,dpm/mot17](#rec_80_prec_80___simulated_eval_frcnn_sdp_dpm_mot17_)
            - [rec_90_prec_100       @ simulated_eval/frcnn,sdp,dpm/mot17](#rec_90_prec_100___simulated_eval_frcnn_sdp_dpm_mot17_)
    - [sdp       @ mot17](#sdp___mot1_7_)
    - [dpm       @ mot17](#dpm___mot1_7_)

<!-- /MarkdownTOC -->

<a id="detra_c_"></a>
# detrac

python36 mAP.py det_path_list_file=/data/DETRAC/det_list.txt img_path_list_file=/data/DETRAC/img_list.txt gt_path_list_file=/data/DETRAC/ann_list.txt labels_path=../labelling_tool/data//predefined_classes_vehicle.txt show_anim=0  score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1

<a id="save_sim_dets___detrac_"></a>
## save_sim_dets       @ detrac-->mAP_sim

python36 mAP.py img_root_dir=/data/DETRAC/Images img_path_list_file=lists/detrac.txt labels_path=../labelling_tool/data//predefined_classes_vehicle.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 save_sim_dets=1

<a id="rec_40_prec_40___save_sim_dets_detrac_"></a>
### rec_40_prec_40       @ save_sim_dets/detrac-->mAP_sim

python36 mAP.py img_root_dir=/data/DETRAC/Images img_path_list_file=lists/detrac.txt labels_path=../labelling_tool/data//predefined_classes_vehicle.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 save_sim_dets=1 sim_precs=0.4 sim_recs=0.4


<a id="eval_sim___detrac_"></a>
## eval_sim       @ detrac-->mAP_sim

python36 mAP.py img_root_dir=/data/DETRAC/Images img_path_list_file=lists/detrac.txt labels_path=../labelling_tool/data//predefined_classes_vehicle.txt score_thresholds=0 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 save_sim_dets=0 show_sim=0 eval_sim=1

<a id="gra_m_"></a>
# gram

python36 mAP.py img_root_dir=/data/GRAM/Images img_path_list_file=lists/gram.txt labels_path=../labelling_tool/data//predefined_classes_vehicle.txt show_anim=0  score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1

<a id="idot___gram_"></a>
## idot       @ gram-->mAP_sim

python36 mAP.py img_root_dir=/data/GRAM/Images img_path_list_file=lists/gram.txt labels_path=../labelling_tool/data/predefined_classes_vehicle.txt show_anim=0  score_thresholds=0:1:0.001 check_seq_name=0 start_id=3 end_id=-1

<a id="no_idot___gram_"></a>
## no_idot       @ gram-->mAP_sim

python36 mAP.py img_root_dir=/data/GRAM/Images img_path_list_file=lists/gram.txt labels_path=../labelling_tool/data//predefined_classes_vehicle.txt show_anim=0  score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=2

<a id="ctc_"></a>
# ctc

<a id="syn_3___ct_c_"></a>
## syn_3       @ ctc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTC/Images img_path_list_file=lists/ctc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 assoc_method=0 detection_names=syn_3

<a id="save_sim_dets___ct_c_"></a>
## save_sim_dets       @ ctc-->mAP_sim

<a id="all___save_sim_dets_ct_c_"></a>
### all       @ save_sim_dets/ctc-->mAP_sim
python36 mAP.py img_root_dir=/data/CTC/Images img_path_list_file=lists/ctc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1

<a id="rec_100_prec_100___save_sim_dets_ct_c_"></a>
### rec_100_prec_100       @ save_sim_dets/ctc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTC/Images img_path_list_file=lists/ctc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1  sim_precs=1.0 sim_recs=1.0 

<a id="rec_80_prec_80___save_sim_dets_ct_c_"></a>
### rec_80_prec_80       @ save_sim_dets/ctc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTC/Images img_path_list_file=lists/ctc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1  sim_precs=0.8 sim_recs=0.8

<a id="rec_60_prec_60___save_sim_dets_ct_c_"></a>
### rec_60_prec_60       @ save_sim_dets/ctc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTC/Images img_path_list_file=lists/ctc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1  sim_precs=0.6 sim_recs=0.6 

<a id="rec_40_prec_40___save_sim_dets_ct_c_"></a>
### rec_40_prec_40       @ save_sim_dets/ctc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTC/Images img_path_list_file=lists/ctc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1  sim_precs=0.4 sim_recs=0.4

<a id="simulated_eval___ct_c_"></a>
## simulated_eval       @ ctc-->mAP_sim
<a id="rec_50_prec_50___simulated_eval_frcnn_mot17_"></a>

python36 mAP.py img_root_dir=/data/CTC/Images img_path_list_file=lists/ctc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0 check_seq_name=0 start_id=0 end_id=0 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=0 show_sim=0 eval_sim=1

<a id="all_seq___simulated_eval_ctc_"></a>
##### all_seq       @ simulated_eval/ctc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTC/Images img_path_list_file=lists/ctc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=0 show_sim=0 eval_sim=1

<a id="ctm_c_"></a>
# ctmc

<a id="syn_3___ctmc_"></a>
## syn_3       @ ctmc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 assoc_method=0 detection_names=syn_3

<a id="mot___ctmc_"></a>
## MOT       @ ctmc-->mAP_sim
<a id="assoc_method_0___mot_ctmc_"></a>
### assoc_method=0       @ MOT/ctmc-->mAP_sim
python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 assoc_method=0 detection_names=MOT

<a id="assoc_method_1___mot_ctmc_"></a>
### assoc_method=1       @ MOT/ctmc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 assoc_method=1 detection_names=MOT

<a id="save_sim_dets___ctmc_"></a>
## save_sim_dets       @ ctmc-->mAP_sim

<a id="all___save_sim_dets_ctmc_"></a>
### all       @ save_sim_dets/ctmc-->mAP_sim
python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1

<a id="rec_100_prec_100___save_sim_dets_ctmc_"></a>
### rec_100_prec_100       @ save_sim_dets/ctmc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1  sim_precs=1.0 sim_recs=1.0 

<a id="rec_80_prec_80___save_sim_dets_ctmc_"></a>
### rec_80_prec_80       @ save_sim_dets/ctmc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1  sim_precs=0.8 sim_recs=0.8

<a id="rec_60_prec_60___save_sim_dets_ctmc_"></a>
### rec_60_prec_60       @ save_sim_dets/ctmc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1  sim_precs=0.6 sim_recs=0.6 

<a id="rec_40_prec_40___save_sim_dets_ctmc_"></a>
### rec_40_prec_40       @ save_sim_dets/ctmc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=1  sim_precs=0.4 sim_recs=0.4

<a id="simulated_eval___ctmc_"></a>
## simulated_eval       @ ctmc-->mAP_sim
<a id="rec_50_prec_50___simulated_eval_frcnn_mot17_"></a>

python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0 check_seq_name=0 start_id=0 end_id=0 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=0 show_sim=0 eval_sim=1

<a id="all_seq___simulated_eval_ctm_c_"></a>
##### all_seq       @ simulated_eval/ctmc-->mAP_sim

python36 mAP.py img_root_dir=/data/CTMC/Images img_path_list_file=lists/ctmc_train.txt labels_path=../labelling_tool/data//predefined_classes_cell.txt score_thresholds=0 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=syn_3 save_sim_dets=0 show_sim=0 eval_sim=1


<a id="mot15_"></a>
# mot15

python36 mAP.py img_root_dir=/data/MOT2015/Images img_path_list_file=lists/mot15.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1

<a id="save_sim_dets___mot1_5_"></a>
## save_sim_dets       @ mot15-->mAP_sim

python36 mAP.py img_root_dir=/data/MOT2015/Images img_path_list_file=lists/mot15.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 save_sim_dets=1

<a id="rec_40_prec_40___save_sim_dets_mot1_5_"></a>
### rec_40_prec_40       @ save_sim_dets/mot15-->mAP_sim

python36 mAP.py img_root_dir=/data/MOT2015/Images img_path_list_file=lists/mot15.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 save_sim_dets=1 sim_precs=0.4 sim_recs=0.4

<a id="frcnn_sdp_dpm___mot1_5_"></a>
## frcnn,sdp,dpm       @ mot15-->mAP_sim

<a id="save_sim_dets___frcnn_sdp_dpm_mot1_5_"></a>
### save_sim_dets       @ frcnn,sdp,dpm/mot15-->mAP_sim

<a id="all___save_sim_dets_frcnn_sdp_dpm_mot1_5_"></a>
#### all       @ save_sim_dets/frcnn,sdp,dpm/mot15-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1

<a id="mot17_"></a>
# mot17

<a id="frcnn___mot1_7_"></a>
## frcnn       @ mot17-->mAP_sim

python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 assoc_method=0

<a id="save_sim_dets___frcnn_mot1_7_"></a>
### save_sim_dets       @ frcnn/mot17-->mAP_sim


<a id="rec_79_prec_47___save_sim_dets_frcnn_mot1_7_"></a>
#### rec_79_prec_47       @ save_sim_dets/frcnn/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn save_sim_dets=1 sim_precs=0.47 sim_recs=0.79 show_sim=1

<a id="rec_50_prec_50___save_sim_dets_frcnn_mot1_7_"></a>
#### rec_50_prec_50       @ save_sim_dets/frcnn/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn save_sim_dets=1 sim_precs=0.5 sim_recs=0.5 show_sim=0

python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn save_sim_dets=1 show_sim=0

<a id="frcnn_sdp_dpm___mot1_7_"></a>
## frcnn,sdp,dpm       @ mot17-->mAP_sim

<a id="save_sim_dets___frcnn_sdp_dpm_mot1_7_"></a>
### save_sim_dets       @ frcnn,sdp,dpm/mot17-->mAP_sim

<a id="all___save_sim_dets_frcnn_sdp_dpm_mot1_7_"></a>
#### all       @ save_sim_dets/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1

<a id="all_seq___all_save_sim_dets_frcnn_sdp_dpm_mot1_7_"></a>
##### all_seq       @ all/save_sim_dets/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1

<a id="rec_40_prec_40___frcnn_sdp_dpm_mot1_7_"></a>
### rec_40_prec_40       @ frcnn,sdp,dpm/mot17-->mAP_sim

python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 sim_precs=0.4 sim_recs=0.4


<a id="recs_100___rec_40_prec_40_frcnn_sdp_dpm_mot17_"></a>
#### recs_100       @ rec_40_prec_40/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 sim_recs=1.0

<a id="rec_50_prec_50___rec_40_prec_40_frcnn_sdp_dpm_mot17_"></a>
#### rec_50_prec_50       @ rec_40_prec_40/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 show_sim=1 sim_precs=0.5 sim_recs=0.5

<a id="rec_79_prec_37___rec_40_prec_40_frcnn_sdp_dpm_mot17_"></a>
#### rec_79_prec_37       @ rec_40_prec_40/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 show_sim=0 sim_precs=0.37 sim_recs=0.79

<a id="rec_50_prec_60___rec_40_prec_40_frcnn_sdp_dpm_mot17_"></a>
#### rec_50_prec_60       @ rec_40_prec_40/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 show_sim=0 sim_recs=0.5  sim_precs=0.6 

<a id="rec_50_prec_70___rec_40_prec_40_frcnn_sdp_dpm_mot17_"></a>
#### rec_50_prec_70       @ rec_40_prec_40/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 show_sim=0 sim_recs=0.5  sim_precs=0.7

<a id="rec_70_prec_60___rec_40_prec_40_frcnn_sdp_dpm_mot17_"></a>
#### rec_70_prec_60       @ rec_40_prec_40/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 show_sim=0 sim_recs=0.7  sim_precs=0.6

<a id="rec_60_prec_70___rec_40_prec_40_frcnn_sdp_dpm_mot17_"></a>
#### rec_60_prec_70       @ rec_40_prec_40/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 show_sim=0 sim_recs=0.6  sim_precs=0.7

<a id="rec_90_prec_100___rec_40_prec_40_frcnn_sdp_dpm_mot17_"></a>
#### rec_90_prec_100       @ rec_40_prec_40/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=1 show_sim=0 sim_recs=0.9 sim_precs=1.0


<a id="simulated_eval___frcnn_sdp_dpm_mot1_7_"></a>
### simulated_eval       @ frcnn,sdp,dpm/mot17-->mAP_sim
<a id="rec_50_prec_50___simulated_eval_frcnn_mot17_"></a>

<a id="eval_sim___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### eval_sim       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim

python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=0 show_sim=0 eval_sim=1

<a id="all_seq___eval_sim_simulated_eval_frcnn_sdp_dpm_mot1_7_"></a>
##### all_seq       @ eval_sim/simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim

python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0 check_seq_name=0 start_id=0 end_id=-1 show_anim=0 show_gt=1 assoc_method=0 detection_names=frcnn,sdp,dpm save_sim_dets=0 show_sim=0 eval_sim=1

<a id="rec_50_prec_70___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### rec_50_prec_70       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=rec_50_prec_70 save_sim_dets=0

<a id="rec_70_prec_60___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### rec_70_prec_60       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=rec_70_prec_60 save_sim_dets=0

<a id="rec_60_prec_70___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### rec_60_prec_70       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=rec_60_prec_70 save_sim_dets=0

<a id="rec_79_prec_47___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### rec_79_prec_47       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=rec_79_prec_47 save_sim_dets=0

<a id="rec_79_prec_37___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### rec_79_prec_37       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=rec_79_prec_37 save_sim_dets=0

<a id="rec_50_prec_90___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### rec_50_prec_90       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=rec_50_prec_90 save_sim_dets=0

python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=1 detection_names=rec_60_prec_80 save_sim_dets=0

<a id="rec_80_prec_80___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### rec_80_prec_80       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=rec_80_prec_80 save_sim_dets=0

<a id="rec_90_prec_100___simulated_eval_frcnn_sdp_dpm_mot17_"></a>
#### rec_90_prec_100       @ simulated_eval/frcnn,sdp,dpm/mot17-->mAP_sim
python36 mAP.py img_root_dir=/data/MOT2017/Images img_path_list_file=lists/mot17_frcnn.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=5 end_id=5 show_anim=0 show_gt=1 assoc_method=0 detection_names=rec_90_prec_100 save_sim_dets=0

<a id="sdp___mot1_7_"></a>
## sdp       @ mot17-->mAP_sim

python36 mAP.py img_root_dir=/data/MOT2017_SDP/Images img_path_list_file=lists/mot17_sdp.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1

<a id="dpm___mot1_7_"></a>
## dpm       @ mot17-->mAP_sim

python36 mAP.py img_root_dir=/data/MOT2017_DPM/Images img_path_list_file=lists/mot17_dpm.txt labels_path=../labelling_tool/data//predefined_classes_person.txt score_thresholds=0:1:0.001 check_seq_name=0 start_id=0 end_id=-1



