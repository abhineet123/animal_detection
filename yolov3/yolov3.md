<!-- MarkdownTOC -->

- [train](#train_)
    - [1_per_seq_6_class_vid_67       @ train](#1_per_seq_6_class_vid_67___trai_n_)
        - [rt_for_tbx       @ 1_per_seq_6_class_vid_67/train](#rt_for_tbx___1_per_seq_6_class_vid_67_train_)
    - [2_per_seq_6_class_vid_67       @ train](#2_per_seq_6_class_vid_67___trai_n_)
    - [5_per_seq_6_class_vid_67       @ train](#5_per_seq_6_class_vid_67___trai_n_)
    - [10_per_seq_6_class_vid_67       @ train](#10_per_seq_6_class_vid_67___trai_n_)
    - [acamp20k6_5       @ train](#acamp20k6_5___trai_n_)
        - [rt       @ acamp20k6_5/train](#rt___acamp20k6_5_trai_n_)
        - [multi-scale       @ acamp20k6_5/train](#multi_scale___acamp20k6_5_trai_n_)
        - [transfer       @ acamp20k6_5/train](#transfer___acamp20k6_5_trai_n_)
    - [20k6_5_no_spp       @ train](#20k6_5_no_spp___trai_n_)
    - [20k6_60       @ train](#20k6_60___trai_n_)
        - [win       @ 20k6_60/train](#win___20k6_60_trai_n_)
    - [20k6_60_no_spp       @ train](#20k6_60_no_spp___trai_n_)
    - [10k6_entire_seq       @ train](#10k6_entire_seq___trai_n_)
        - [mixed_precision       @ 10k6_entire_seq/train](#mixed_precision___10k6_entire_seq_trai_n_)
    - [15k6_entire_seq       @ train](#15k6_entire_seq___trai_n_)
    - [1k6_vid_entire_seq_mp       @ train](#1k6_vid_entire_seq_mp___trai_n_)
        - [mp       @ 1k6_vid_entire_seq_mp/train](#mp___1k6_vid_entire_seq_mp_trai_n_)
    - [10k6_vid_entire_seq_mp       @ train](#10k6_vid_entire_seq_mp___trai_n_)
        - [1_per_seq_val       @ 10k6_vid_entire_seq_mp/train](#1_per_seq_val___10k6_vid_entire_seq_mp_train_)
    - [15k6_vid_entire_seq       @ train](#15k6_vid_entire_seq___trai_n_)
        - [1_per_seq_val       @ 15k6_vid_entire_seq/train](#1_per_seq_val___15k6_vid_entire_seq_trai_n_)
    - [1K_static3a_sampled       @ train](#1k_static3a_sampled___trai_n_)
        - [mixed_precision       @ 1K_static3a_sampled/train](#mixed_precision___1k_static3a_sampled_trai_n_)
            - [rt_tbx       @ mixed_precision/1K_static3a_sampled/train](#rt_tbx___mixed_precision_1k_static3a_sampled_trai_n_)
    - [500_static3a       @ train](#500_static3a___trai_n_)
    - [200_static3a       @ train](#200_static3a___trai_n_)
        - [win       @ 200_static3a/train](#win___200_static3a_train_)
    - [20K_vid3a       @ train](#20k_vid3a___trai_n_)
    - [p1_a_h_no_mask_bear_3_frg_per_bkg       @ train](#p1_a_h_no_mask_bear_3_frg_per_bkg___trai_n_)
    - [p1_a_h_no_mask_bear_3_frg_per_bkg       @ train](#p1_a_h_no_mask_bear_3_frg_per_bkg___trai_n__1)
    - [p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ train](#p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___trai_n_)
    - [20K_vid4a       @ train](#20k_vid4a___trai_n_)
    - [p1_4_class_a_h       @ train](#p1_4_class_a_h___trai_n_)
    - [p1_a_h_3_class_30       @ train](#p1_a_h_3_class_30___trai_n_)
    - [p1_a_h_4_class_30       @ train](#p1_a_h_4_class_30___trai_n_)
        - [rt       @ p1_a_h_4_class_30/train](#rt___p1_a_h_4_class_30_trai_n_)
    - [p1_a_h_no_mask_3_class_30       @ train](#p1_a_h_no_mask_3_class_30___trai_n_)
    - [p1_3_class_a_h_3_frg_per_bkg       @ train](#p1_3_class_a_h_3_frg_per_bkg___trai_n_)
    - [p1_4_class_a_h_3_frg_per_bkg       @ train](#p1_4_class_a_h_3_frg_per_bkg___trai_n_)
    - [p1_3_class_a_h_no_mask_3_frg_per_bkg       @ train](#p1_3_class_a_h_no_mask_3_frg_per_bkg___trai_n_)
    - [p1_4_class_a_h_no_mask_3_frg_per_bkg       @ train](#p1_4_class_a_h_no_mask_3_frg_per_bkg___trai_n_)
    - [p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ train](#p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___trai_n_)
    - [p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ train](#p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___trai_n_)
    - [p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ train](#p1_3_class_a_h_siam_mask_3_frg_per_bkg___trai_n_)
    - [p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ train](#p1_4_class_a_h_siam_mask_3_frg_per_bkg___trai_n_)
    - [p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg       @ train](#p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg___trai_n_)
    - [p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg       @ train](#p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg___trai_n_)
    - [p1_a_h_no_mask_3_class_30       @ train](#p1_a_h_no_mask_3_class_30___trai_n__1)
    - [p1_a_h_no_mask_4_class_30       @ train](#p1_a_h_no_mask_4_class_30___trai_n_)
    - [acamp1k8_vid       @ train](#acamp1k8_vid___trai_n_)
    - [acamp1k8_vid_entire_seq       @ train](#acamp1k8_vid_entire_seq___trai_n_)
    - [1k8_vid_even_min_1       @ train](#1k8_vid_even_min_1___trai_n_)
- [test](#tes_t_)
    - [acamp1K_static3a       @ test](#acamp1k_static3a___test_)
        - [inverted       @ acamp1K_static3a/test](#inverted___acamp1k_static3a_tes_t_)
            - [best_91       @ inverted/acamp1K_static3a/test](#best_91___inverted_acamp1k_static3a_test_)
                - [all_vid       @ best_91/inverted/acamp1K_static3a/test](#all_vid___best_91_inverted_acamp1k_static3a_test_)
            - [latest       @ inverted/acamp1K_static3a/test](#latest___inverted_acamp1k_static3a_test_)
            - [p1_3_class_a_h_3_frg_per_bkg_inv       @ inverted/acamp1K_static3a/test](#p1_3_class_a_h_3_frg_per_bkg_inv___inverted_acamp1k_static3a_test_)
                - [all_vid       @ p1_3_class_a_h_3_frg_per_bkg_inv/inverted/acamp1K_static3a/test](#all_vid___p1_3_class_a_h_3_frg_per_bkg_inv_inverted_acamp1k_static3a_tes_t_)
    - [acamp200_static3a       @ test](#acamp200_static3a___test_)
        - [inverted       @ acamp200_static3a/test](#inverted___acamp200_static3a_test_)
            - [best_91       @ inverted/acamp200_static3a/test](#best_91___inverted_acamp200_static3a_tes_t_)
                - [nvme       @ best_91/inverted/acamp200_static3a/test](#nvme___best_91_inverted_acamp200_static3a_tes_t_)
            - [latest       @ inverted/acamp200_static3a/test](#latest___inverted_acamp200_static3a_tes_t_)
    - [acamp500_static3a       @ test](#acamp500_static3a___test_)
        - [best_377       @ acamp500_static3a/test](#best_377___acamp500_static3a_test_)
        - [latest       @ acamp500_static3a/test](#latest___acamp500_static3a_test_)
            - [class_agnostic       @ latest/acamp500_static3a/test](#class_agnostic___latest_acamp500_static3a_tes_t_)
    - [acamp20K_vid3a       @ test](#acamp20k_vid3a___test_)
        - [acamp200_static3a_sampled_train_inverted       @ acamp20K_vid3a/test](#acamp200_static3a_sampled_train_inverted___acamp20k_vid3a_tes_t_)
            - [best_206       @ acamp200_static3a_sampled_train_inverted/acamp20K_vid3a/test](#best_206___acamp200_static3a_sampled_train_inverted_acamp20k_vid3a_test_)
            - [latest       @ acamp200_static3a_sampled_train_inverted/acamp20K_vid3a/test](#latest___acamp200_static3a_sampled_train_inverted_acamp20k_vid3a_test_)
        - [all_static       @ acamp20K_vid3a/test](#all_static___acamp20k_vid3a_tes_t_)
            - [best_206       @ all_static/acamp20K_vid3a/test](#best_206___all_static_acamp20k_vid3a_test_)
            - [latest       @ all_static/acamp20K_vid3a/test](#latest___all_static_acamp20k_vid3a_test_)
                - [class_agnostic       @ latest/all_static/acamp20K_vid3a/test](#class_agnostic___latest_all_static_acamp20k_vid3a_tes_t_)
    - [20K_vid4a       @ test](#20k_vid4a___test_)
        - [p1_4_class_a_h_3_frg_per_bkg_inv       @ 20K_vid4a/test](#p1_4_class_a_h_3_frg_per_bkg_inv___20k_vid4a_test_)
    - [1_per_seq_6_class_vid_67       @ test](#1_per_seq_6_class_vid_67___test_)
        - [1_5_10_per_seq_6_class_vid_67_inverted       @ 1_per_seq_6_class_vid_67/test](#1_5_10_per_seq_6_class_vid_67_inverted___1_per_seq_6_class_vid_67_tes_t_)
            - [bear_1_1       @ 1_5_10_per_seq_6_class_vid_67_inverted/1_per_seq_6_class_vid_67/test](#bear_1_1___1_5_10_per_seq_6_class_vid_67_inverted_1_per_seq_6_class_vid_67_test_)
            - [nms_type_SOFT       @ 1_5_10_per_seq_6_class_vid_67_inverted/1_per_seq_6_class_vid_67/test](#nms_type_soft___1_5_10_per_seq_6_class_vid_67_inverted_1_per_seq_6_class_vid_67_test_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 1_per_seq_6_class_vid_67/test](#1_2_5_10_per_seq_6_class_vid_67_inverted___1_per_seq_6_class_vid_67_tes_t_)
    - [2_per_seq_6_class_vid_67       @ test](#2_per_seq_6_class_vid_67___test_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 2_per_seq_6_class_vid_67/test](#1_2_5_10_per_seq_6_class_vid_67_inverted___2_per_seq_6_class_vid_67_tes_t_)
    - [5_per_seq_6_class_vid_67       @ test](#5_per_seq_6_class_vid_67___test_)
        - [1_5_10_per_seq_6_class_vid_67_inverted       @ 5_per_seq_6_class_vid_67/test](#1_5_10_per_seq_6_class_vid_67_inverted___5_per_seq_6_class_vid_67_tes_t_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 5_per_seq_6_class_vid_67/test](#1_2_5_10_per_seq_6_class_vid_67_inverted___5_per_seq_6_class_vid_67_tes_t_)
    - [10_per_seq_6_class_vid_67       @ test](#10_per_seq_6_class_vid_67___test_)
        - [1_5_10_per_seq_6_class_vid_67_inverted       @ 10_per_seq_6_class_vid_67/test](#1_5_10_per_seq_6_class_vid_67_inverted___10_per_seq_6_class_vid_67_test_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 10_per_seq_6_class_vid_67/test](#1_2_5_10_per_seq_6_class_vid_67_inverted___10_per_seq_6_class_vid_67_test_)
    - [acamp20k6_5       @ test](#acamp20k6_5___test_)
        - [inverted       @ acamp20k6_5/test](#inverted___acamp20k6_5_test_)
            - [only_video       @ inverted/acamp20k6_5/test](#only_video___inverted_acamp20k6_5_tes_t_)
            - [nms_type_or       @ inverted/acamp20k6_5/test](#nms_type_or___inverted_acamp20k6_5_tes_t_)
            - [coyote_jesse_1_1       @ inverted/acamp20k6_5/test](#coyote_jesse_1_1___inverted_acamp20k6_5_tes_t_)
        - [acamp20k6_1_from_end       @ acamp20k6_5/test](#acamp20k6_1_from_end___acamp20k6_5_test_)
    - [acamp20k6_5_tracking       @ test](#acamp20k6_5_tracking___test_)
        - [inverted_video       @ acamp20k6_5_tracking/test](#inverted_video___acamp20k6_5_tracking_tes_t_)
        - [bear_10_3       @ acamp20k6_5_tracking/test](#bear_10_3___acamp20k6_5_tracking_tes_t_)
        - [bear_1_1       @ acamp20k6_5_tracking/test](#bear_1_1___acamp20k6_5_tracking_tes_t_)
            - [no_tracking       @ bear_1_1/acamp20k6_5_tracking/test](#no_tracking___bear_1_1_acamp20k6_5_tracking_test_)
    - [acamp20k6_5_rt       @ test](#acamp20k6_5_rt___test_)
        - [inverted       @ acamp20k6_5_rt/test](#inverted___acamp20k6_5_rt_tes_t_)
            - [best_84       @ inverted/acamp20k6_5_rt/test](#best_84___inverted_acamp20k6_5_rt_test_)
            - [latest.pt       @ inverted/acamp20k6_5_rt/test](#latest_pt___inverted_acamp20k6_5_rt_test_)
    - [acamp20k6_5_4_ms       @ test](#acamp20k6_5_4_ms___test_)
        - [inverted       @ acamp20k6_5_4_ms/test](#inverted___acamp20k6_5_4_ms_tes_t_)
            - [best       @ inverted/acamp20k6_5_4_ms/test](#best___inverted_acamp20k6_5_4_ms_test_)
    - [acamp20k6_5_no_spp       @ test](#acamp20k6_5_no_spp___test_)
        - [inverted       @ acamp20k6_5_no_spp/test](#inverted___acamp20k6_5_no_spp_tes_t_)
            - [best_264       @ inverted/acamp20k6_5_no_spp/test](#best_264___inverted_acamp20k6_5_no_spp_test_)
            - [latest       @ inverted/acamp20k6_5_no_spp/test](#latest___inverted_acamp20k6_5_no_spp_test_)
            - [acamp_1_per_class_6_class       @ inverted/acamp20k6_5_no_spp/test](#acamp_1_per_class_6_class___inverted_acamp20k6_5_no_spp_test_)
    - [acamp20k6_60       @ test](#acamp20k6_60___test_)
        - [inverted       @ acamp20k6_60/test](#inverted___acamp20k6_60_tes_t_)
            - [show_img       @ inverted/acamp20k6_60/test](#show_img___inverted_acamp20k6_60_test_)
    - [acamp20k6_60_no_spp       @ test](#acamp20k6_60_no_spp___test_)
        - [inverted       @ acamp20k6_60_no_spp/test](#inverted___acamp20k6_60_no_spp_test_)
            - [show_img       @ inverted/acamp20k6_60_no_spp/test](#show_img___inverted_acamp20k6_60_no_spp_tes_t_)
    - [acamp10k6_entire_seq       @ test](#acamp10k6_entire_seq___test_)
        - [standard       @ acamp10k6_entire_seq/test](#standard___acamp10k6_entire_seq_tes_t_)
            - [inverted       @ standard/acamp10k6_entire_seq/test](#inverted___standard_acamp10k6_entire_seq_test_)
        - [mixed_precision       @ acamp10k6_entire_seq/test](#mixed_precision___acamp10k6_entire_seq_tes_t_)
            - [inverted       @ mixed_precision/acamp10k6_entire_seq/test](#inverted___mixed_precision_acamp10k6_entire_seq_tes_t_)
    - [acamp15k6_entire_seq       @ test](#acamp15k6_entire_seq___test_)
        - [inverted       @ acamp15k6_entire_seq/test](#inverted___acamp15k6_entire_seq_tes_t_)
    - [acamp10k6_vid_entire_seq_mp       @ test](#acamp10k6_vid_entire_seq_mp___test_)
        - [latest       @ acamp10k6_vid_entire_seq_mp/test](#latest___acamp10k6_vid_entire_seq_mp_test_)
        - [best_56       @ acamp10k6_vid_entire_seq_mp/test](#best_56___acamp10k6_vid_entire_seq_mp_test_)
            - [SiamFC       @ best_56/acamp10k6_vid_entire_seq_mp/test](#siamfc___best_56_acamp10k6_vid_entire_seq_mp_test_)
            - [DaSiamRPN       @ best_56/acamp10k6_vid_entire_seq_mp/test](#dasiamrpn___best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_1       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_1___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_2       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_2___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_3       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_3___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_4       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_4___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_1_ua_filter_0       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_1_ua_filter_0___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_2_ua_filter_0       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_2_ua_filter_0___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_1_ua_filter_0_ua_thresh_0       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_1_ua_filter_0_ua_thresh_0___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_2_ua_filter_0_ua_thresh_0       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_2_ua_filter_0_ua_thresh_0___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_)
            - [class_agnostic       @ best_56/acamp10k6_vid_entire_seq_mp/test](#class_agnostic___best_56_acamp10k6_vid_entire_seq_mp_test_)
                - [max_trackers_3_ua_filter_0_ua_thresh_0       @ class_agnostic/best_56/acamp10k6_vid_entire_seq_mp/test](#max_trackers_3_ua_filter_0_ua_thresh_0___class_agnostic_best_56_acamp10k6_vid_entire_seq_mp_tes_t_)
            - [class_agnostic       @ best_56/acamp10k6_vid_entire_seq_mp/test](#class_agnostic___best_56_acamp10k6_vid_entire_seq_mp_test__1)
    - [acamp1k6_vid_entire_seq       @ test](#acamp1k6_vid_entire_seq___test_)
        - [latest       @ acamp1k6_vid_entire_seq/test](#latest___acamp1k6_vid_entire_seq_test_)
                - [max_trackers_1       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_1___latest_acamp1k6_vid_entire_seq_tes_t_)
                - [max_trackers_2       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_2___latest_acamp1k6_vid_entire_seq_tes_t_)
                - [max_trackers_3       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_3___latest_acamp1k6_vid_entire_seq_tes_t_)
                - [max_trackers_4       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_4___latest_acamp1k6_vid_entire_seq_tes_t_)
                - [max_trackers_1_ua_filter_0       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_1_ua_filter_0___latest_acamp1k6_vid_entire_seq_tes_t_)
                - [max_trackers_2_ua_filter_0       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_2_ua_filter_0___latest_acamp1k6_vid_entire_seq_tes_t_)
                - [max_trackers_1_ua_filter_0_ua_thresh_0       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_1_ua_filter_0_ua_thresh_0___latest_acamp1k6_vid_entire_seq_tes_t_)
                - [max_trackers_2_ua_filter_0_ua_thresh_0       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_2_ua_filter_0_ua_thresh_0___latest_acamp1k6_vid_entire_seq_tes_t_)
                - [max_trackers_3_ua_filter_0_ua_thresh_0       @ latest/acamp1k6_vid_entire_seq/test](#max_trackers_3_ua_filter_0_ua_thresh_0___latest_acamp1k6_vid_entire_seq_tes_t_)
    - [acamp10k6_vid_entire_seq_1_per_seq_val_mp       @ test](#acamp10k6_vid_entire_seq_1_per_seq_val_mp___test_)
        - [latest       @ acamp10k6_vid_entire_seq_1_per_seq_val_mp/test](#latest___acamp10k6_vid_entire_seq_1_per_seq_val_mp_test_)
        - [best_45       @ acamp10k6_vid_entire_seq_1_per_seq_val_mp/test](#best_45___acamp10k6_vid_entire_seq_1_per_seq_val_mp_test_)
    - [acamp15k6_vid_entire_seq       @ test](#acamp15k6_vid_entire_seq___test_)
        - [latest       @ acamp15k6_vid_entire_seq/test](#latest___acamp15k6_vid_entire_seq_tes_t_)
        - [best_17       @ acamp15k6_vid_entire_seq/test](#best_17___acamp15k6_vid_entire_seq_tes_t_)
    - [acamp15k6_vid_entire_seq_1_per_seq_val       @ test](#acamp15k6_vid_entire_seq_1_per_seq_val___test_)
        - [latest       @ acamp15k6_vid_entire_seq_1_per_seq_val/test](#latest___acamp15k6_vid_entire_seq_1_per_seq_val_tes_t_)
        - [best_14       @ acamp15k6_vid_entire_seq_1_per_seq_val/test](#best_14___acamp15k6_vid_entire_seq_1_per_seq_val_tes_t_)
    - [acamp1k8_vid       @ test](#acamp1k8_vid___test_)
        - [best_354       @ acamp1k8_vid/test](#best_354___acamp1k8_vid_tes_t_)
        - [latest       @ acamp1k8_vid/test](#latest___acamp1k8_vid_tes_t_)
    - [acamp1k8_vid_entire_seq       @ test](#acamp1k8_vid_entire_seq___test_)
        - [latest       @ acamp1k8_vid_entire_seq/test](#latest___acamp1k8_vid_entire_seq_test_)
                - [max_trackers_1_ua_filter_0       @ latest/acamp1k8_vid_entire_seq/test](#max_trackers_1_ua_filter_0___latest_acamp1k8_vid_entire_seq_tes_t_)
                - [max_trackers_2_ua_filter_0       @ latest/acamp1k8_vid_entire_seq/test](#max_trackers_2_ua_filter_0___latest_acamp1k8_vid_entire_seq_tes_t_)
                - [max_trackers_1_ua_filter_0_ua_thresh_0       @ latest/acamp1k8_vid_entire_seq/test](#max_trackers_1_ua_filter_0_ua_thresh_0___latest_acamp1k8_vid_entire_seq_tes_t_)
                - [max_trackers_2_ua_filter_0_ua_thresh_0       @ latest/acamp1k8_vid_entire_seq/test](#max_trackers_2_ua_filter_0_ua_thresh_0___latest_acamp1k8_vid_entire_seq_tes_t_)
                - [max_trackers_3_ua_filter_0_ua_thresh_0       @ latest/acamp1k8_vid_entire_seq/test](#max_trackers_3_ua_filter_0_ua_thresh_0___latest_acamp1k8_vid_entire_seq_tes_t_)
        - [best_1       @ acamp1k8_vid_entire_seq/test](#best_1___acamp1k8_vid_entire_seq_test_)
    - [1k8_vid_even_min_1       @ test](#1k8_vid_even_min_1___test_)
        - [latest       @ 1k8_vid_even_min_1/test](#latest___1k8_vid_even_min_1_tes_t_)
    - [p1_a_h_no_mask_3_class_30       @ test](#p1_a_h_no_mask_3_class_30___test_)
        - [best_362       @ p1_a_h_no_mask_3_class_30/test](#best_362___p1_a_h_no_mask_3_class_30_test_)
            - [p1_a_h_3_class_30_inverted       @ best_362/p1_a_h_no_mask_3_class_30/test](#p1_a_h_3_class_30_inverted___best_362_p1_a_h_no_mask_3_class_30_tes_t_)
        - [latest       @ p1_a_h_no_mask_3_class_30/test](#latest___p1_a_h_no_mask_3_class_30_test_)
            - [p1_a_h_3_class_30_inverted       @ latest/p1_a_h_no_mask_3_class_30/test](#p1_a_h_3_class_30_inverted___latest_p1_a_h_no_mask_3_class_30_tes_t_)
    - [p1_a_h_no_mask_4_class_30       @ test](#p1_a_h_no_mask_4_class_30___test_)
        - [best_375       @ p1_a_h_no_mask_4_class_30/test](#best_375___p1_a_h_no_mask_4_class_30_test_)
            - [p1_a_h_4_class_30_inverted       @ best_375/p1_a_h_no_mask_4_class_30/test](#p1_a_h_4_class_30_inverted___best_375_p1_a_h_no_mask_4_class_30_tes_t_)
        - [latest       @ p1_a_h_no_mask_4_class_30/test](#latest___p1_a_h_no_mask_4_class_30_test_)
            - [p1_a_h_4_class_30_inverted       @ latest/p1_a_h_no_mask_4_class_30/test](#p1_a_h_4_class_30_inverted___latest_p1_a_h_no_mask_4_class_30_tes_t_)
    - [p1_a_h_3_class_30       @ test](#p1_a_h_3_class_30___test_)
        - [best_556       @ p1_a_h_3_class_30/test](#best_556___p1_a_h_3_class_30_test_)
        - [latest       @ p1_a_h_3_class_30/test](#latest___p1_a_h_3_class_30_test_)
    - [p1_a_h_4_class_30       @ test](#p1_a_h_4_class_30___test_)
        - [best_1740       @ p1_a_h_4_class_30/test](#best_1740___p1_a_h_4_class_30_test_)
        - [latest       @ p1_a_h_4_class_30/test](#latest___p1_a_h_4_class_30_test_)
    - [p1_a_h_4_class_30_rt       @ test](#p1_a_h_4_class_30_rt___test_)
        - [best_454       @ p1_a_h_4_class_30_rt/test](#best_454___p1_a_h_4_class_30_rt_tes_t_)
        - [latest       @ p1_a_h_4_class_30_rt/test](#latest___p1_a_h_4_class_30_rt_tes_t_)
    - [p1_a_h_no_mask_bear_3_frg_per_bkg       @ test](#p1_a_h_no_mask_bear_3_frg_per_bkg___test_)
        - [best_874       @ p1_a_h_no_mask_bear_3_frg_per_bkg/test](#best_874___p1_a_h_no_mask_bear_3_frg_per_bkg_test_)
        - [latest       @ p1_a_h_no_mask_bear_3_frg_per_bkg/test](#latest___p1_a_h_no_mask_bear_3_frg_per_bkg_test_)
    - [p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ test](#p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___test_)
        - [best_685       @ p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/test](#best_685___p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg_tes_t_)
        - [latest       @ p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/test](#latest___p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg_tes_t_)
    - [p1_3_class_a_h_3_frg_per_bkg       @ test](#p1_3_class_a_h_3_frg_per_bkg___test_)
        - [best_927       @ p1_3_class_a_h_3_frg_per_bkg/test](#best_927___p1_3_class_a_h_3_frg_per_bkg_tes_t_)
        - [latest       @ p1_3_class_a_h_3_frg_per_bkg/test](#latest___p1_3_class_a_h_3_frg_per_bkg_tes_t_)
    - [p1_4_class_a_h_3_frg_per_bkg       @ test](#p1_4_class_a_h_3_frg_per_bkg___test_)
        - [best_909       @ p1_4_class_a_h_3_frg_per_bkg/test](#best_909___p1_4_class_a_h_3_frg_per_bkg_tes_t_)
        - [latest       @ p1_4_class_a_h_3_frg_per_bkg/test](#latest___p1_4_class_a_h_3_frg_per_bkg_tes_t_)
    - [p1_3_class_a_h_no_mask_3_frg_per_bkg       @ test](#p1_3_class_a_h_no_mask_3_frg_per_bkg___test_)
        - [latest_on_p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_no_mask_3_frg_per_bkg/test](#latest_on_p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_no_mask_3_frg_per_bkg_tes_t_)
    - [p1_4_class_a_h_no_mask_3_frg_per_bkg       @ test](#p1_4_class_a_h_no_mask_3_frg_per_bkg___test_)
        - [latest_on_p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_no_mask_3_frg_per_bkg/test](#latest_on_p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_no_mask_3_frg_per_bkg_tes_t_)
    - [p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ test](#p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___test_)
        - [latest_on_p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/test](#latest_on_p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tes_t_)
    - [p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ test](#p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___test_)
        - [latest_on_p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/test](#latest_on_p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tes_t_)
    - [p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ test](#p1_3_class_a_h_siam_mask_3_frg_per_bkg___test_)
        - [latest_on_p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_siam_mask_3_frg_per_bkg/test](#latest_on_p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_siam_mask_3_frg_per_bkg_tes_t_)
    - [p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ test](#p1_4_class_a_h_siam_mask_3_frg_per_bkg___test_)
        - [latest_on_p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_siam_mask_3_frg_per_bkg/test](#latest_on_p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_siam_mask_3_frg_per_bkg_tes_t_)

<!-- /MarkdownTOC -->

<a id="train_"></a>
# train

<a id="1_per_seq_6_class_vid_67___trai_n_"></a>
## 1_per_seq_6_class_vid_67       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/1_per_seq_6_class_vid_67.txt --net_cfg=cfg/1_per_seq_6_class_vid_67.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/1_per_seq_6_class_vid_67_24 --epochs=1000

<a id="rt_for_tbx___1_per_seq_6_class_vid_67_train_"></a>
### rt_for_tbx       @ 1_per_seq_6_class_vid_67/train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/1_per_seq_6_class_vid_67.txt --net_cfg=cfg/1_per_seq_6_class_vid_67.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/1_per_seq_6_class_vid_67_24_rt --epochs=1000

<a id="2_per_seq_6_class_vid_67___trai_n_"></a>
## 2_per_seq_6_class_vid_67       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/2_per_seq_6_class_vid_67.txt --net_cfg=cfg/2_per_seq_6_class_vid_67.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/2_per_seq_6_class_vid_67_24 --epochs=1000

<a id="5_per_seq_6_class_vid_67___trai_n_"></a>
## 5_per_seq_6_class_vid_67       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/5_per_seq_6_class_vid_67.txt --net_cfg=cfg/5_per_seq_6_class_vid_67.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/5_per_seq_6_class_vid_67_24 --epochs=1000

<a id="10_per_seq_6_class_vid_67___trai_n_"></a>
## 10_per_seq_6_class_vid_67       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/10_per_seq_6_class_vid_67.txt --net_cfg=cfg/10_per_seq_6_class_vid_67.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/10_per_seq_6_class_vid_67_24 --epochs=1000

<a id="acamp20k6_5___trai_n_"></a>
## acamp20k6_5       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20k6_5_24

<a id="rt___acamp20k6_5_trai_n_"></a>
### rt       @ acamp20k6_5/train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20k6_5_24_rt

<a id="multi_scale___acamp20k6_5_trai_n_"></a>
### multi-scale       @ acamp20k6_5/train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=4 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20k6_5_4_ms --multi-scale --resume


<a id="transfer___acamp20k6_5_trai_n_"></a>
### transfer       @ acamp20k6_5/train

__nan__

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=32 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20k6_5_32_tr --transfer

<a id="20k6_5_no_spp___trai_n_"></a>
## 20k6_5_no_spp       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/acamp20k6_5_no_spp.txt --net_cfg=cfg/acamp20k6_5_no_spp.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20k6_5_no_spp_24


<a id="20k6_60___trai_n_"></a>
## 20k6_60       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/acamp20k6_60.txt --net_cfg=cfg/acamp20k6_60.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20k6_60_24

<a id="win___20k6_60_trai_n_"></a>
### win       @ 20k6_60/train

python3 yolov3_train.py --data_cfg=cfg/acamp20k6_60.txt --net_cfg=cfg/acamp20k6_60.cfg --batch_size=12 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20k6_60_24

<a id="20k6_60_no_spp___trai_n_"></a>
## 20k6_60_no_spp       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/acamp20k6_60_no_spp.txt --net_cfg=cfg/acamp20k6_60_no_spp.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20k6_60_no_spp_24

<a id="10k6_entire_seq___trai_n_"></a>
## 10k6_entire_seq       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp10k6_entire_seq.txt --net_cfg=cfg/acamp10k6_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp10k6_entire_seq_24

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp10k6_entire_seq_24")'

<a id="mixed_precision___10k6_entire_seq_trai_n_"></a>
### mixed_precision       @ 10k6_entire_seq/train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp10k6_entire_seq.txt --net_cfg=cfg/acamp10k6_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp10k6_entire_seq_mp_24 --mixed_precision=1

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp10k6_entire_seq_mp_24")'

<a id="15k6_entire_seq___trai_n_"></a>
## 15k6_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/acamp15k6_entire_seq.txt --net_cfg=cfg/acamp15k6_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp15k6_entire_seq_24

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp15k6_entire_seq_24")'

<a id="1k6_vid_entire_seq_mp___trai_n_"></a>
## 1k6_vid_entire_seq_mp       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp1k6_vid_entire_seq_24 --mixed_precision=0

<a id="mp___1k6_vid_entire_seq_mp_trai_n_"></a>
### mp       @ 1k6_vid_entire_seq_mp/train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp1k6_vid_entire_seq_24_mp --mixed_precision=1

<a id="10k6_vid_entire_seq_mp___trai_n_"></a>
## 10k6_vid_entire_seq_mp       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24 --mixed_precision=1

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp10k6_vid_entire_seq_mp_24")'

<a id="1_per_seq_val___10k6_vid_entire_seq_mp_train_"></a>
### 1_per_seq_val       @ 10k6_vid_entire_seq_mp/train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp10k6_vid_entire_seq_1_per_seq_val.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_1_per_seq_val_24 --mixed_precision=1

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp10k6_vid_entire_seq_mp_1_per_seq_val_24")'


<a id="15k6_vid_entire_seq___trai_n_"></a>
## 15k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/acamp15k6_vid_entire_seq.txt --net_cfg=cfg/acamp15k6_vid_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp15k6_vid_entire_seq_24

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp15k6_vid_entire_seq_24")'

<a id="1_per_seq_val___15k6_vid_entire_seq_trai_n_"></a>
### 1_per_seq_val       @ 15k6_vid_entire_seq/train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/acamp15k6_vid_entire_seq_1_per_seq_val.txt --net_cfg=cfg/acamp15k6_vid_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp15k6_vid_entire_seq_1_per_seq_val_24

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp15k6_vid_entire_seq_1_per_seq_val_24")'

<a id="1k_static3a_sampled___trai_n_"></a>
## 1K_static3a_sampled       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp1K_static3a.txt --net_cfg=cfg/acamp1K_static3a.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp1K_static3a

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp1K_static3a")'

<a id="mixed_precision___1k_static3a_sampled_trai_n_"></a>
### mixed_precision       @ 1K_static3a_sampled/train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp1K_static3a.txt --net_cfg=cfg/acamp1K_static3a.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp1K_static3a_mp --mixed_precision=1

python3 -c "import torch;print(torch.__version__);print(torch.version.cuda)"

<a id="rt_tbx___mixed_precision_1k_static3a_sampled_trai_n_"></a>
#### rt_tbx       @ mixed_precision/1K_static3a_sampled/train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp1K_static3a.txt --net_cfg=cfg/acamp1K_static3a.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp1K_static3a_rt --mixed_precision=0

<a id="500_static3a___trai_n_"></a>
## 500_static3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/acamp500_static3a.txt --net_cfg=cfg/acamp500_static3a.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp500_static3a --epochs=1000

<a id="200_static3a___trai_n_"></a>
## 200_static3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/acamp200_static3a.txt --net_cfg=cfg/acamp200_static3a.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp200_static3a --resume --epochs=400

<a id="win___200_static3a_train_"></a>
### win       @ 200_static3a/train

python3 yolov3_train.py --data_cfg=cfg/acamp200_static3a.txt --net_cfg=cfg/acamp200_static3a.cfg --batch_size=12 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp200_static3a --epochs=400

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp200_static3a")'

<a id="20k_vid3a___trai_n_"></a>
## 20K_vid3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/acamp20K_vid3a.txt --net_cfg=cfg/acamp20K_vid3a.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20K_vid3a --epochs=400

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp20K_vid3a")'

<a id="p1_a_h_no_mask_bear_3_frg_per_bkg___trai_n_"></a>
## p1_a_h_no_mask_bear_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/p1_a_h_no_mask_bear_3_frg_per_bkg.txt --net_cfg=cfg/p1_a_h_no_mask_bear_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_a_h_no_mask_bear_3_frg_per_bkg --epochs=1000

<a id="p1_a_h_no_mask_bear_3_frg_per_bkg___trai_n__1"></a>
## p1_a_h_no_mask_bear_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/p1_a_h_no_mask_bear_3_frg_per_bkg.txt --net_cfg=cfg/p1_a_h_no_mask_bear_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_a_h_no_mask_bear_3_frg_per_bkg --epochs=1000

<a id="p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___trai_n_"></a>
## p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --net_cfg=cfg/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg --epochs=1000

<a id="20k_vid4a___trai_n_"></a>
## 20K_vid4a       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/acamp20K_vid4a.txt --net_cfg=cfg/acamp20K_vid4a.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp20K_vid4a_mp --epochs=400 --mixed_precision=1

<a id="p1_4_class_a_h___trai_n_"></a>
## p1_4_class_a_h       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/p1_4_class_a_h.txt --net_cfg=cfg/p1_4_class_a_h.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_4_class_a_h_mp --epochs=1000 --mixed_precision=1 --load_sep=1

<a id="p1_a_h_3_class_30___trai_n_"></a>
## p1_a_h_3_class_30       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/p1_a_h_3_class_30.txt --net_cfg=cfg/p1_a_h_3_class_30.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_a_h_3_class_30 --epochs=400

<a id="p1_a_h_4_class_30___trai_n_"></a>
## p1_a_h_4_class_30       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_a_h_4_class_30.txt --net_cfg=cfg/p1_a_h_4_class_30.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_a_h_4_class_30 --epochs=400

<a id="rt___p1_a_h_4_class_30_trai_n_"></a>
### rt       @ p1_a_h_4_class_30/train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_a_h_4_class_30.txt --net_cfg=cfg/p1_a_h_4_class_30.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_a_h_4_class_30_rt --epochs=400

<a id="p1_a_h_no_mask_3_class_30___trai_n_"></a>
## p1_a_h_no_mask_3_class_30       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_a_h_no_mask_3_class_30.txt --net_cfg=cfg/p1_a_h_no_mask_3_class_30.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_a_h_no_mask_3_class_30 --epochs=600 --resume

<a id="p1_3_class_a_h_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/p1_3_class_a_h_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_3_class_a_h_3_frg_per_bkg --epochs=1000

<a id="p1_4_class_a_h_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_4_class_a_h_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_4_class_a_h_3_frg_per_bkg --epochs=1000

<a id="p1_3_class_a_h_no_mask_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_no_mask_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/p1_3_class_a_h_no_mask_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_no_mask_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_3_class_a_h_no_mask_3_frg_per_bkg --epochs=1000

<a id="p1_4_class_a_h_no_mask_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_no_mask_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_4_class_a_h_no_mask_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_no_mask_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_4_class_a_h_no_mask_3_frg_per_bkg --epochs=1000

<a id="p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg --epochs=1000

<a id="p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg --epochs=1000

<a id="p1_3_class_a_h_siam_mask_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/p1_3_class_a_h_siam_mask_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_siam_mask_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_3_class_a_h_siam_mask_3_frg_per_bkg --epochs=1000

<a id="p1_4_class_a_h_siam_mask_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_4_class_a_h_siam_mask_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_siam_mask_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_4_class_a_h_siam_mask_3_frg_per_bkg --epochs=1000

<a id="p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg --epochs=1000

<a id="p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg --epochs=1000

<a id="p1_a_h_no_mask_3_class_30___trai_n__1"></a>
## p1_a_h_no_mask_3_class_30       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/p1_a_h_no_mask_3_class_30.txt --net_cfg=cfg/p1_a_h_no_mask_3_class_30.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_a_h_no_mask_3_class_30 --epochs=600 --resume

<a id="p1_a_h_no_mask_4_class_30___trai_n_"></a>
## p1_a_h_no_mask_4_class_30       @ train

CUDA_VISIBLE_DEVICES=0 python3 yolov3_train.py --data_cfg=cfg/p1_a_h_no_mask_4_class_30.txt --net_cfg=cfg/p1_a_h_no_mask_4_class_30.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/p1_a_h_no_mask_4_class_30 --epochs=400

<a id="acamp1k8_vid___trai_n_"></a>
## acamp1k8_vid       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/acamp1k8_vid.txt --net_cfg=cfg/acamp1k8_vid.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp1k8_vid_24  --resume --epochs=400

python3 -c 'from yolov3_utils import utils; utils.plot_results(folder="yolov3_weights/acamp1k8_vid_24")'

<a id="acamp1k8_vid_entire_seq___trai_n_"></a>
## acamp1k8_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/acamp1k8_vid_entire_seq.txt --net_cfg=cfg/acamp1k8_vid_entire_seq.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp1k8_vid_entire_seq_24 --epochs=400  --mixed_precision=0

<a id="1k8_vid_even_min_1___trai_n_"></a>
## 1k8_vid_even_min_1       @ train

CUDA_VISIBLE_DEVICES=1 python3 yolov3_train.py --data_cfg=cfg/1k8_vid_even_min_1.txt --net_cfg=cfg/1k8_vid_even_min_1.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/1k8_vid_even_min_1_24_mp --epochs=400  --mixed_precision=1

<a id="tes_t_"></a>
# test

<a id="acamp1k_static3a___test_"></a>
## acamp1K_static3a       @ test

<a id="inverted___acamp1k_static3a_tes_t_"></a>
### inverted       @ acamp1K_static3a/test

<a id="best_91___inverted_acamp1k_static3a_test_"></a>
#### best_91       @ inverted/acamp1K_static3a/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp1K_static3a.txt --net_cfg=cfg/acamp1K_static3a.cfg --batch_size=48 --weights=yolov3_weights/acamp1K_static3a/best_91.pt --test_path=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1K_static3a/best_91_on_acamp1K_static3a_sampled_train_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted score_thresholds=0:1:0.001

<a id="all_vid___best_91_inverted_acamp1k_static3a_test_"></a>
##### all_vid       @ best_91/inverted/acamp1K_static3a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1K_static3a.txt --net_cfg=cfg/acamp1K_static3a.cfg --batch_size=32 --weights=yolov3_weights/acamp1K_static3a/best_91.pt --test_path=/data/acamp/acamp20k/all_vid_3_class.txt

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_video_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1K_static3a/best_91_on_all_vid_3_class/ score_thresholds=0:1:0.001


<a id="latest___inverted_acamp1k_static3a_test_"></a>
#### latest       @ inverted/acamp1K_static3a/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp1K_static3a.txt --net_cfg=cfg/acamp1K_static3a.cfg --batch_size=48 --weights=yolov3_weights/acamp1K_static3a/latest.pt --test_path=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1K_static3a/latest_on_acamp1K_static3a_sampled_train_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_3_frg_per_bkg_inv___inverted_acamp1k_static3a_test_"></a>
#### p1_3_class_a_h_3_frg_per_bkg_inv       @ inverted/acamp1K_static3a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1K_static3a.txt --net_cfg=cfg/acamp1K_static3a.cfg --batch_size=32 --weights=yolov3_weights/acamp1K_static3a/latest.pt --test_path=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1K_static3a/latest_on_p1_3_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="all_vid___p1_3_class_a_h_3_frg_per_bkg_inv_inverted_acamp1k_static3a_tes_t_"></a>
##### all_vid       @ p1_3_class_a_h_3_frg_per_bkg_inv/inverted/acamp1K_static3a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1K_static3a.txt --net_cfg=cfg/acamp1K_static3a.cfg --batch_size=32 --weights=yolov3_weights/acamp1K_static3a/latest.pt --test_path=/data/acamp/acamp20k/all_vid_3_class.txt

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_video_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1K_static3a/latest_on_all_vid_3_class/ score_thresholds=0:1:0.001

<a id="acamp200_static3a___test_"></a>
## acamp200_static3a       @ test

<a id="inverted___acamp200_static3a_test_"></a>
### inverted       @ acamp200_static3a/test

<a id="best_91___inverted_acamp200_static3a_tes_t_"></a>
#### best_91       @ inverted/acamp200_static3a/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp200_static3a.txt --net_cfg=cfg/acamp200_static3a.cfg --batch_size=48 --weights=yolov3_weights/acamp200_static3a/best_94.pt --test_path=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp200_static3a/best_94_on_acamp200_static3a_sampled_train_inverted/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted score_thresholds=0:1:0.001

<a id="nvme___best_91_inverted_acamp200_static3a_tes_t_"></a>
##### nvme       @ best_91/inverted/acamp200_static3a/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp200_static3a.txt --net_cfg=cfg/acamp200_static3a.cfg --batch_size=48 --weights=yolov3_weights/acamp200_static3a/best_94.pt --test_path=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted_nvme.txt


<a id="latest___inverted_acamp200_static3a_tes_t_"></a>
#### latest       @ inverted/acamp200_static3a/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp200_static3a.txt --net_cfg=cfg/acamp200_static3a.cfg --batch_size=32 --weights=yolov3_weights/acamp200_static3a/latest.pt --test_path=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp200_static3a/latest_on_acamp200_static3a_sampled_train_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted score_thresholds=0:1:0.001

<a id="acamp500_static3a___test_"></a>
## acamp500_static3a       @ test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_train.py --data_cfg=cfg/acamp500_static3a.txt --net_cfg=cfg/acamp500_static3a.cfg --batch_size=24 --pretrained_weights=pretrained_weights --weights=yolov3_weights/acamp500_static3a --epochs=1000

<a id="best_377___acamp500_static3a_test_"></a>
### best_377       @ acamp500_static3a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp500_static3a.txt --net_cfg=cfg/acamp500_static3a.cfg --batch_size=32 --weights=yolov3_weights/acamp500_static3a/best_377.pt --test_path=/data/acamp/acamp20k/acamp500_static3a_train_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp500_static3a/best_377_on_acamp500_static3a_train_inverted/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted score_thresholds=0:1:0.001 n_threads=1

<a id="latest___acamp500_static3a_test_"></a>
### latest       @ acamp500_static3a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp500_static3a.txt --net_cfg=cfg/acamp500_static3a.cfg --batch_size=32 --weights=yolov3_weights/acamp500_static3a/latest.pt --test_path=/data/acamp/acamp20k/acamp500_static3a_train_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp500_static3a/latest_on_acamp500_static3a_train_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted score_thresholds=0:1:0.001

<a id="class_agnostic___latest_acamp500_static3a_tes_t_"></a>
#### class_agnostic       @ latest/acamp500_static3a/test

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp500_static3a/latest_on_acamp500_static3a_train_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted score_thresholds=0:1:0.001 class_agnostic=1

<a id="acamp20k_vid3a___test_"></a>
## acamp20K_vid3a       @ test

<a id="acamp200_static3a_sampled_train_inverted___acamp20k_vid3a_tes_t_"></a>
### acamp200_static3a_sampled_train_inverted       @ acamp20K_vid3a/test

<a id="best_206___acamp200_static3a_sampled_train_inverted_acamp20k_vid3a_test_"></a>
#### best_206       @ acamp200_static3a_sampled_train_inverted/acamp20K_vid3a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20K_vid3a.txt --net_cfg=cfg/acamp20K_vid3a.cfg --batch_size=32 --weights=yolov3_weights/acamp20K_vid3a/best_206.pt --test_path=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted.txt

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20K_vid3a/best_206_on_acamp200_static3a_sampled_train_inverted/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted score_thresholds=0:1:0.001
<a id="latest___acamp200_static3a_sampled_train_inverted_acamp20k_vid3a_test_"></a>
#### latest       @ acamp200_static3a_sampled_train_inverted/acamp20K_vid3a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20K_vid3a.txt --net_cfg=cfg/acamp20K_vid3a.cfg --batch_size=32 --weights=yolov3_weights/acamp20K_vid3a/latest.pt --test_path=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted.txt

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20K_vid3a/latest_on_acamp200_static3a_sampled_train_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted score_thresholds=0:1:0.001

<a id="all_static___acamp20k_vid3a_tes_t_"></a>
### all_static       @ acamp20K_vid3a/test

<a id="best_206___all_static_acamp20k_vid3a_test_"></a>
#### best_206       @ all_static/acamp20K_vid3a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20K_vid3a.txt --net_cfg=cfg/acamp20K_vid3a.cfg --batch_size=32 --weights=yolov3_weights/acamp20K_vid3a/best_206.pt --test_path=/data/acamp/acamp20k/all_static_3_class.txt

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20K_vid3a/best_206_on_all_static_3_class/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/all_static_3_class score_thresholds=0:1:0.001

<a id="latest___all_static_acamp20k_vid3a_test_"></a>
#### latest       @ all_static/acamp20K_vid3a/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp20K_vid3a.txt --net_cfg=cfg/acamp20K_vid3a.cfg --batch_size=24 --weights=yolov3_weights/acamp20K_vid3a/latest.pt --test_path=/data/acamp/acamp20k/all_static_3_class.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20K_vid3a/latest_on_all_static_3_class/ load_samples=1 load_samples_root=/data/acamp/acamp20k/all_static_3_class score_thresholds=0:1:0.001

<a id="class_agnostic___latest_all_static_acamp20k_vid3a_tes_t_"></a>
##### class_agnostic       @ latest/all_static/acamp20K_vid3a/test

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_static_3_class.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20K_vid3a/latest_on_all_static_3_class/ load_samples=1 load_samples_root=/data/acamp/acamp20k/all_static_3_class score_thresholds=0:1:0.001 class_agnostic=1

<a id="20k_vid4a___test_"></a>
## 20K_vid4a       @ test

<a id="p1_4_class_a_h_3_frg_per_bkg_inv___20k_vid4a_test_"></a>
### p1_4_class_a_h_3_frg_per_bkg_inv       @ 20K_vid4a/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20K_vid4a.txt --net_cfg=cfg/acamp20K_vid4a.cfg --batch_size=32 --weights=yolov3_weights/acamp20K_vid4a_mp/latest.pt --test_path=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20K_vid4a_mp/latest_on_p1_4_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001


<a id="1_per_seq_6_class_vid_67___test_"></a>
## 1_per_seq_6_class_vid_67       @ test

<a id="1_5_10_per_seq_6_class_vid_67_inverted___1_per_seq_6_class_vid_67_tes_t_"></a>
### 1_5_10_per_seq_6_class_vid_67_inverted       @ 1_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/1_per_seq_6_class_vid_67.txt --net_cfg=cfg/1_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/1_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video_67.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/1_per_seq_6_class_vid_67_24/latest_on_1_5_10_per_seq_6_class_vid_67_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="bear_1_1___1_5_10_per_seq_6_class_vid_67_inverted_1_per_seq_6_class_vid_67_test_"></a>
#### bear_1_1       @ 1_5_10_per_seq_6_class_vid_67_inverted/1_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/1_per_seq_6_class_vid_67.txt --net_cfg=cfg/1_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/1_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_5_10_per_seq_6_class_bear_1_1_inverted.txt --vis=1

<a id="nms_type_soft___1_5_10_per_seq_6_class_vid_67_inverted_1_per_seq_6_class_vid_67_test_"></a>
#### nms_type_SOFT       @ 1_5_10_per_seq_6_class_vid_67_inverted/1_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/1_per_seq_6_class_vid_67.txt --net_cfg=cfg/1_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/1_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted.txt --nms_type=SOFT --out_suffix=nms_type_SOFT

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video_67.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/1_per_seq_6_class_vid_67_24/latest_nms_type_SOFT_on_1_5_10_per_seq_6_class_vid_67_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___1_per_seq_6_class_vid_67_tes_t_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 1_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/1_per_seq_6_class_vid_67.txt --net_cfg=cfg/1_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/1_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video_67.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/1_per_seq_6_class_vid_67_24/latest_on_1_2_5_10_per_seq_6_class_vid_67_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="2_per_seq_6_class_vid_67___test_"></a>
## 2_per_seq_6_class_vid_67       @ test

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___2_per_seq_6_class_vid_67_tes_t_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 2_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/2_per_seq_6_class_vid_67.txt --net_cfg=cfg/2_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/2_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted.txt

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video_67.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/2_per_seq_6_class_vid_67_24/latest_on_1_2_5_10_per_seq_6_class_vid_67_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001


<a id="5_per_seq_6_class_vid_67___test_"></a>
## 5_per_seq_6_class_vid_67       @ test

<a id="1_5_10_per_seq_6_class_vid_67_inverted___5_per_seq_6_class_vid_67_tes_t_"></a>
### 1_5_10_per_seq_6_class_vid_67_inverted       @ 5_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/5_per_seq_6_class_vid_67.txt --net_cfg=cfg/5_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/5_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video_67.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/5_per_seq_6_class_vid_67_24/latest_on_1_5_10_per_seq_6_class_vid_67_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___5_per_seq_6_class_vid_67_tes_t_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 5_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/5_per_seq_6_class_vid_67.txt --net_cfg=cfg/5_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/5_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted.txt

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video_67.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/5_per_seq_6_class_vid_67_24/latest_on_1_2_5_10_per_seq_6_class_vid_67_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="10_per_seq_6_class_vid_67___test_"></a>
## 10_per_seq_6_class_vid_67       @ test

<a id="1_5_10_per_seq_6_class_vid_67_inverted___10_per_seq_6_class_vid_67_test_"></a>
### 1_5_10_per_seq_6_class_vid_67_inverted       @ 10_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/10_per_seq_6_class_vid_67.txt --net_cfg=cfg/10_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/10_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video_67.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/10_per_seq_6_class_vid_67_24/latest_on_1_5_10_per_seq_6_class_vid_67_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___10_per_seq_6_class_vid_67_test_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 10_per_seq_6_class_vid_67/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/10_per_seq_6_class_vid_67.txt --net_cfg=cfg/10_per_seq_6_class_vid_67.cfg --batch_size=24 --weights=yolov3_weights/10_per_seq_6_class_vid_67_24/latest.pt --test_path=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video_67.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/10_per_seq_6_class_vid_67_24/latest_on_1_2_5_10_per_seq_6_class_vid_67_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="acamp20k6_5___test_"></a>
## acamp20k6_5       @ test

<a id="inverted___acamp20k6_5_test_"></a>
### inverted       @ acamp20k6_5/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=48 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_24/backup270_on_acamp20k6_5_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted score_thresholds=0:1:0.001 n_threads=8

<a id="only_video___inverted_acamp20k6_5_tes_t_"></a>
#### only_video       @ inverted/acamp20k6_5/test

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_24/backup270_on_acamp20k6_5_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid out_postfix=inv_only_vid score_thresholds=0:1:0.001

<a id="nms_type_or___inverted_acamp20k6_5_tes_t_"></a>
#### nms_type_or       @ inverted/acamp20k6_5/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=1 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt nms_type='or' save_dir=backup270_nms_type_or_on_20k6_5_inverted

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_24/backup270_nms_type_or_on_20k6_5_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted score_thresholds=0:1:0.001


<a id="coyote_jesse_1_1___inverted_acamp20k6_5_tes_t_"></a>
#### coyote_jesse_1_1       @ inverted/acamp20k6_5/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=24 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/coyote_jesse_1_1_5_inverted_yolov3_pt.txt

<a id="acamp20k6_1_from_end___acamp20k6_5_test_"></a>
### acamp20k6_1_from_end       @ acamp20k6_5/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=24 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_1_from_end.txt

<a id="acamp20k6_5_tracking___test_"></a>
## acamp20k6_5_tracking       @ test

<a id="inverted_video___acamp20k6_5_tracking_tes_t_"></a>
### inverted_video       @ acamp20k6_5_tracking/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=1 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_video_inverted_yolov3_pt.txt --use_tracking=1 --conf_thresh=0.03 --track_thresh=0.5 --assoc_thresh=0.1 --nms_thresh=0.1

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=1 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_video_inverted_yolov3_pt.txt --use_tracking=1 --conf_thresh=0.03 --track_thresh=0.5 --assoc_thresh=0.1 --nms_thresh=0.1

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_24/backup270_on_acamp20k6_5_video_inverted_yolov3_pt load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_video_inverted score_thresholds=0:1:0.001


<a id="bear_10_3___acamp20k6_5_tracking_tes_t_"></a>
### bear_10_3       @ acamp20k6_5_tracking/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=1 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_bear_10_3_inverted_yolov3_pt.txt --use_tracking=1 --conf_thresh=0.001 --track_thresh=0.5 --assoc_thresh=0.1 --nms_thresh=0.1 --unassoc_thresh=10 --track_diff=10 --vis=1

<a id="bear_1_1___acamp20k6_5_tracking_tes_t_"></a>
### bear_1_1       @ acamp20k6_5_tracking/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=1 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_bear_1_1_inverted_yolov3_pt.txt --use_tracking=1 --conf_thresh=0.03 --track_thresh=0.5 --assoc_thresh=0.1 --nms_thresh=0.1

<a id="no_tracking___bear_1_1_acamp20k6_5_tracking_test_"></a>
#### no_tracking       @ bear_1_1/acamp20k6_5_tracking/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=1 --weights=yolov3_weights/acamp20k6_5_24/backup270.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_bear_1_1_inverted_yolov3_pt.txt --conf_thresh=0.03 --track_thresh=0.5 --assoc_thresh=0.1 --nms_thresh=0.1 --use_tracking=0


<a id="acamp20k65rt__test"></a>
<a id="acamp20k6_5_rt___test_"></a>
## acamp20k6_5_rt       @ test

<a id="inverted___acamp20k6_5_rt_tes_t_"></a>
### inverted       @ acamp20k6_5_rt/test

<a id="best_84___inverted_acamp20k6_5_rt_test_"></a>
#### best_84       @ inverted/acamp20k6_5_rt/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=24 --weights=yolov3_weights/acamp20k6_5_24_rt/best_84.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_24_rt/best_84_on_acamp20k6_5_inverted_yolov3_pt/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted

<a id="latest_pt___inverted_acamp20k6_5_rt_test_"></a>
#### latest.pt       @ inverted/acamp20k6_5_rt/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=24 --weights=yolov3_weights/acamp20k6_5_24_rt/latest.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_24_rt/latest_on_acamp20k6_5_inverted_yolov3_pt/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted  --show_animation=1 --save_animation=1 --save_file_res=1280x720

<a id="acamp20k6_5_4_ms___test_"></a>
## acamp20k6_5_4_ms       @ test

<a id="inverted___acamp20k6_5_4_ms_tes_t_"></a>
### inverted       @ acamp20k6_5_4_ms/test

<a id="best___inverted_acamp20k6_5_4_ms_test_"></a>
#### best       @ inverted/acamp20k6_5_4_ms/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5.txt --net_cfg=cfg/acamp20k6_5.cfg --batch_size=24 --weights=yolov3_weights/acamp20k6_5_4_ms/best.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_4_ms/best_on_acamp20k6_5_inverted_yolov3_pt/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted


<a id="acamp20k6_5_no_spp___test_"></a>
## acamp20k6_5_no_spp       @ test

<a id="inverted___acamp20k6_5_no_spp_tes_t_"></a>
### inverted       @ acamp20k6_5_no_spp/test

<a id="best_264___inverted_acamp20k6_5_no_spp_test_"></a>
#### best_264       @ inverted/acamp20k6_5_no_spp/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5_no_spp.txt --net_cfg=cfg/acamp20k6_5_no_spp.cfg --batch_size=24 --weights=yolov3_weights/acamp20k6_5_no_spp_24/best_264.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=0 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_no_spp_24/best_264_on_acamp20k6_5_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted score_thresholds=0:1:0.001

<a id="latest___inverted_acamp20k6_5_no_spp_test_"></a>
#### latest       @ inverted/acamp20k6_5_no_spp/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_5_no_spp.txt --net_cfg=cfg/acamp20k6_5_no_spp.cfg --batch_size=24 --weights=yolov3_weights/acamp20k6_5_no_spp_24/latest.pt --test_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=0 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_no_spp_24/latest_on_acamp20k6_5_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted score_thresholds=0:1:0.001

<a id="acamp_1_per_class_6_class___inverted_acamp20k6_5_no_spp_test_"></a>
#### acamp_1_per_class_6_class       @ inverted/acamp20k6_5_no_spp/test

CUDA_VISIBLE_DEVICES=0 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_1_per_class_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_no_spp_24/best_264_on_acamp20k6_5_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted


CUDA_VISIBLE_DEVICES=0 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_1_per_class_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_5_no_spp_24/best_264_on_acamp20k6_5_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inverted combine_sequences=1


<a id="acamp20k6_60___test_"></a>
## acamp20k6_60       @ test

<a id="inverted___acamp20k6_60_tes_t_"></a>
### inverted       @ acamp20k6_60/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_60.txt --net_cfg=cfg/acamp20k6_60.cfg --batch_size=48 --weights=yolov3_weights/acamp20k6_60_24/best_91.pt --test_path=/data/acamp/acamp20k/acamp20k6_60_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_60_24/best_91_on_acamp20k6_60_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_inverted

<a id="show_img___inverted_acamp20k6_60_test_"></a>
#### show_img       @ inverted/acamp20k6_60/test

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_60_24/best_91_on_acamp20k6_60_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_inverted show_img=0 eval_every=0 show_animation=0 

<a id="acamp20k6_60_no_spp___test_"></a>
## acamp20k6_60_no_spp       @ test

<a id="inverted___acamp20k6_60_no_spp_test_"></a>
### inverted       @ acamp20k6_60_no_spp/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp20k6_60_no_spp.txt --net_cfg=cfg/acamp20k6_60_no_spp.cfg --batch_size=40 --weights=yolov3_weights/acamp20k6_60_no_spp_24/best_51.pt --test_path=/data/acamp/acamp20k/acamp20k6_60_inverted_yolov3_pt.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_60_no_spp_24/best_51_on_acamp20k6_60_inverted_yolov3_pt/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp20k6_60_inverted score_thresholds=0:1:0.001

<a id="show_img___inverted_acamp20k6_60_no_spp_tes_t_"></a>
#### show_img       @ inverted/acamp20k6_60_no_spp/test

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_60_no_spp_24/best_51_on_acamp20k6_60_inverted_yolov3_pt/ load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_inverted show_img=1 eval_every=-1

<a id="acamp10k6_entire_seq___test_"></a>
## acamp10k6_entire_seq       @ test

<a id="standard___acamp10k6_entire_seq_tes_t_"></a>
### standard       @ acamp10k6_entire_seq/test

<a id="inverted___standard_acamp10k6_entire_seq_test_"></a>
#### inverted       @ standard/acamp10k6_entire_seq/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_entire_seq.txt --net_cfg=cfg/acamp10k6_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp10k6_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp10k6_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_60_no_spp_24/best_51_on_acamp20k6_60_inverted_yolov3_pt/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp20k6_60_inverted score_thresholds=0:1:0.001

<a id="mixed_precision___acamp10k6_entire_seq_tes_t_"></a>
### mixed_precision       @ acamp10k6_entire_seq/test

<a id="inverted___mixed_precision_acamp10k6_entire_seq_tes_t_"></a>
#### inverted       @ mixed_precision/acamp10k6_entire_seq/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_entire_seq.txt --net_cfg=cfg/acamp10k6_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp10k6_entire_seq_mp_24/latest.pt --test_path=/data/acamp/acamp20k/acamp10k6_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_60_no_spp_24/best_51_on_acamp20k6_60_inverted_yolov3_pt/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp20k6_60_inverted score_thresholds=0:1:0.001

<a id="acamp15k6_entire_seq___test_"></a>
## acamp15k6_entire_seq       @ test

<a id="inverted___acamp15k6_entire_seq_tes_t_"></a>
### inverted       @ acamp15k6_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp15k6_entire_seq.txt --net_cfg=cfg/acamp15k6_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp15k6_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp15k6_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp20k6_60_no_spp_24/best_51_on_acamp20k6_60_inverted_yolov3_pt/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp20k6_60_inverted score_thresholds=0:1:0.001

<a id="acamp10k6_vid_entire_seq_mp___test_"></a>
## acamp10k6_vid_entire_seq_mp       @ test

<a id="latest___acamp10k6_vid_entire_seq_mp_test_"></a>
### latest       @ acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/latest.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/latest_on_acamp10k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="best_56___acamp10k6_vid_entire_seq_mp_test_"></a>
### best_56       @ acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="siamfc___best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
#### SiamFC       @ best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=fc max_trackers=5 vis=1

<a id="dasiamrpn___best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
#### DaSiamRPN       @ best_56/acamp10k6_vid_entire_seq_mp/test

<a id="max_trackers_1___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
##### max_trackers_1       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=1 vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_1_rt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_1_rt/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_2___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
##### max_trackers_2       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=2 vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_2_rt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_2_rt/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_3___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
##### max_trackers_3       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=3 vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_3_rt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_3_rt/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1


<a id="max_trackers_4___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
##### max_trackers_4       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=4  vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_4_rt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_4_rt/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_1_ua_filter_0___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
##### max_trackers_1_ua_filter_0       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=1 vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_1_rt_ua_filter_0 filter_unassociated=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_1_rt_ua_filter_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_2_ua_filter_0___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
##### max_trackers_2_ua_filter_0       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=2 vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0 filter_unassociated=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_1_ua_filter_0_ua_thresh_0___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
##### max_trackers_1_ua_filter_0_ua_thresh_0       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=1 vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_1_rt_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_1_rt_ua_filter_0_ua_thresh_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_2_ua_filter_0_ua_thresh_0___dasiamrpn_best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
##### max_trackers_2_ua_filter_0_ua_thresh_0       @ DaSiamRPN/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=2 vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0_ua_thresh_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="class_agnostic___best_56_acamp10k6_vid_entire_seq_mp_test_"></a>
#### class_agnostic       @ best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0_ua_thresh_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1 class_agnostic=1

<a id="max_trackers_3_ua_filter_0_ua_thresh_0___class_agnostic_best_56_acamp10k6_vid_entire_seq_mp_tes_t_"></a>
##### max_trackers_3_ua_filter_0_ua_thresh_0       @ class_agnostic/best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=3 vis=0 da_siam_rpn.gpu_id=0 save_dir=best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_3_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_3_ua_filter_0_ua_thresh_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="class_agnostic___best_56_acamp10k6_vid_entire_seq_mp_test__1"></a>
#### class_agnostic       @ best_56/acamp10k6_vid_entire_seq_mp/test

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_24/best_56_on_acamp10k6_vid_entire_seq_inv_rpn_max_trackers_3_ua_filter_0_ua_thresh_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1 class_agnostic=1

<a id="acamp1k6_vid_entire_seq___test_"></a>
## acamp1k6_vid_entire_seq       @ test

<a id="latest___acamp1k6_vid_entire_seq_test_"></a>
### latest       @ acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_1___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_1       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=1 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_1

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_1 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_2___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_2       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=2 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_2

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_2 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_3___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_3       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=3 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_3

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_3 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_4___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_4       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=4 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_4

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_4 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_1_ua_filter_0___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_1_ua_filter_0       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=1 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_1_rt_ua_filter_0 filter_unassociated=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_1_rt_ua_filter_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_2_ua_filter_0___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_2_ua_filter_0       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=2 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0 filter_unassociated=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_1_ua_filter_0_ua_thresh_0___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_1_ua_filter_0_ua_thresh_0       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=1 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_1_rt_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_1_rt_ua_filter_0_ua_thresh_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_2_ua_filter_0_ua_thresh_0___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_2_ua_filter_0_ua_thresh_0       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=2 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0_ua_thresh_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_3_ua_filter_0_ua_thresh_0___latest_acamp1k6_vid_entire_seq_tes_t_"></a>
##### max_trackers_3_ua_filter_0_ua_thresh_0       @ latest/acamp1k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k6_vid_entire_seq.txt --net_cfg=cfg/acamp1k6_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=3 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_3_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k6_vid_entire_seq_24/latest_on_acamp1k6_vid_entire_seq_inv_rpn_max_trackers_3_ua_filter_0_ua_thresh_0 load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1


<a id="acamp10k6_vid_entire_seq_1_per_seq_val_mp___test_"></a>
## acamp10k6_vid_entire_seq_1_per_seq_val_mp       @ test

<a id="latest___acamp10k6_vid_entire_seq_1_per_seq_val_mp_test_"></a>
### latest       @ acamp10k6_vid_entire_seq_1_per_seq_val_mp/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq_1_per_seq_val.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_1_per_seq_val_24/latest.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_1_per_seq_val_24/latest_on_acamp10k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score

<a id="best_45___acamp10k6_vid_entire_seq_1_per_seq_val_mp_test_"></a>
### best_45       @ acamp10k6_vid_entire_seq_1_per_seq_val_mp/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp10k6_vid_entire_seq_1_per_seq_val.txt --net_cfg=cfg/acamp10k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp10k6_vid_entire_seq_mp_1_per_seq_val_24/best_45.pt --test_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/ac1amp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp10k6_vid_entire_seq_mp_1_per_seq_val_24/best_45_on_acamp10k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="acamp15k6_vid_entire_seq___test_"></a>
## acamp15k6_vid_entire_seq       @ test

<a id="latest___acamp15k6_vid_entire_seq_tes_t_"></a>
### latest       @ acamp15k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp15k6_vid_entire_seq.txt --net_cfg=cfg/acamp15k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp15k6_vid_entire_seq_24/latest.pt --test_path=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp15k6_vid_entire_seq_24/latest_on_acamp15k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="best_17___acamp15k6_vid_entire_seq_tes_t_"></a>
### best_17       @ acamp15k6_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp15k6_vid_entire_seq.txt --net_cfg=cfg/acamp15k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp15k6_vid_entire_seq_24/best_17.pt --test_path=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp15k6_vid_entire_seq_24/best_17_on_acamp15k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="acamp15k6_vid_entire_seq_1_per_seq_val___test_"></a>
## acamp15k6_vid_entire_seq_1_per_seq_val       @ test

<a id="latest___acamp15k6_vid_entire_seq_1_per_seq_val_tes_t_"></a>
### latest       @ acamp15k6_vid_entire_seq_1_per_seq_val/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp15k6_vid_entire_seq_1_per_seq_val.txt --net_cfg=cfg/acamp15k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp15k6_vid_entire_seq_1_per_seq_val_24/latest.pt --test_path=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp15k6_vid_entire_seq_1_per_seq_val_24/latest_on_acamp15k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="best_14___acamp15k6_vid_entire_seq_1_per_seq_val_tes_t_"></a>
### best_14       @ acamp15k6_vid_entire_seq_1_per_seq_val/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/acamp15k6_vid_entire_seq_1_per_seq_val.txt --net_cfg=cfg/acamp15k6_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp15k6_vid_entire_seq_1_per_seq_val_24/best_14.pt --test_path=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k6.pbtxt n_frames=0 show_img=0 n_classes=6 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_6_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp15k6_vid_entire_seq_1_per_seq_val_24/best_14_on_acamp15k6_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="acamp1k8_vid___test_"></a>
## acamp1k8_vid       @ test

<a id="best_354___acamp1k8_vid_tes_t_"></a>
### best_354       @ acamp1k8_vid/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid.txt --net_cfg=cfg/acamp1k8_vid.cfg --batch_size=32 --weights=yolov3_weights/acamp1k8_vid_24/best_354.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_24/best_354_on_acamp1k8_vid_inverted/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_inverted score_thresholds=0:1:0.001

<a id="latest___acamp1k8_vid_tes_t_"></a>
### latest       @ acamp1k8_vid/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid.txt --net_cfg=cfg/acamp1k8_vid.cfg --batch_size=32 --weights=yolov3_weights/acamp1k8_vid_24/latest.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_24/latest_on_acamp1k8_vid_inverted/ --load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_inverted score_thresholds=0:1:0.001

<a id="acamp1k8_vid_entire_seq___test_"></a>
## acamp1k8_vid_entire_seq       @ test

<a id="latest___acamp1k8_vid_entire_seq_test_"></a>
### latest       @ acamp1k8_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid_entire_seq.txt --net_cfg=cfg/acamp1k8_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_1_ua_filter_0___latest_acamp1k8_vid_entire_seq_tes_t_"></a>
##### max_trackers_1_ua_filter_0       @ latest/acamp1k8_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid_entire_seq.txt --net_cfg=cfg/acamp1k8_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=1 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_1_ua_filter_0 filter_unassociated=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_1_ua_filter_0/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_2_ua_filter_0___latest_acamp1k8_vid_entire_seq_tes_t_"></a>
##### max_trackers_2_ua_filter_0       @ latest/acamp1k8_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid_entire_seq.txt --net_cfg=cfg/acamp1k8_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=2 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0 filter_unassociated=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_1_ua_filter_0_ua_thresh_0___latest_acamp1k8_vid_entire_seq_tes_t_"></a>
##### max_trackers_1_ua_filter_0_ua_thresh_0       @ latest/acamp1k8_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid_entire_seq.txt --net_cfg=cfg/acamp1k8_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=1 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_1_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_1_ua_filter_0_ua_thresh_0/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1


<a id="max_trackers_2_ua_filter_0_ua_thresh_0___latest_acamp1k8_vid_entire_seq_tes_t_"></a>
##### max_trackers_2_ua_filter_0_ua_thresh_0       @ latest/acamp1k8_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid_entire_seq.txt --net_cfg=cfg/acamp1k8_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=2 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_2_ua_filter_0_ua_thresh_0/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="max_trackers_3_ua_filter_0_ua_thresh_0___latest_acamp1k8_vid_entire_seq_tes_t_"></a>
##### max_trackers_3_ua_filter_0_ua_thresh_0       @ latest/acamp1k8_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid_entire_seq.txt --net_cfg=cfg/acamp1k8_vid_entire_seq.cfg --batch_size=1 --weights=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv.txt tracker_type=rpn max_trackers=3 vis=0 da_siam_rpn.gpu_id=0 save_dir=latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_3_ua_filter_0_ua_thresh_0 filter_unassociated=0 unassoc_thresh=0

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/latest_on_acamp1k8_vid_entire_seq_inv_rpn_max_trackers_3_ua_filter_0_ua_thresh_0/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1


<a id="best_1___acamp1k8_vid_entire_seq_test_"></a>
### best_1       @ acamp1k8_vid_entire_seq/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/acamp1k8_vid_entire_seq.txt --net_cfg=cfg/acamp1k8_vid_entire_seq.cfg --batch_size=24 --weights=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/best_1.pt --test_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/acamp1k8_vid_entire_seq_24_mp/best_1_on_acamp1k8_vid_entire_seq_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="1k8_vid_even_min_1___test_"></a>
## 1k8_vid_even_min_1       @ test

<a id="latest___1k8_vid_even_min_1_tes_t_"></a>
### latest       @ 1k8_vid_even_min_1/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/1k8_vid_even_min_1.txt --net_cfg=cfg/1k8_vid_even_min_1.cfg --batch_size=4 --weights=yolov3_weights/1k8_vid_even_min_1_24_mp/latest.pt --test_path=/data/acamp/acamp20k/1k8_vid_even_min_1_inv.txt

CUDA_VISIBLE_DEVICES=2 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_4k8.pbtxt n_frames=0 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=../tf_api/acamp_all_8_class_video.txt  random_sampling=0 sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/1k8_vid_even_min_1_24_mp/latest_on_1k8_vid_even_min_1_inv/ load_samples=seq_to_samples.txt load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv score_thresholds=0:1:0.001 allow_seq_skipping=1 n_threads=1


<a id="p1_a_h_no_mask_3_class_30___test_"></a>
## p1_a_h_no_mask_3_class_30       @ test

<a id="best_362___p1_a_h_no_mask_3_class_30_test_"></a>
### best_362       @ p1_a_h_no_mask_3_class_30/test

<a id="p1_a_h_3_class_30_inverted___best_362_p1_a_h_no_mask_3_class_30_tes_t_"></a>
#### p1_a_h_3_class_30_inverted       @ best_362/p1_a_h_no_mask_3_class_30/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_no_mask_3_class_30.txt --net_cfg=cfg/p1_a_h_no_mask_3_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_no_mask_3_class_30/best_362.pt --test_path=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_no_mask_3_class_30/best_362_on_p1_a_h_3_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted score_thresholds=0:1:0.001

<a id="latest___p1_a_h_no_mask_3_class_30_test_"></a>
### latest       @ p1_a_h_no_mask_3_class_30/test

<a id="p1_a_h_3_class_30_inverted___latest_p1_a_h_no_mask_3_class_30_tes_t_"></a>
#### p1_a_h_3_class_30_inverted       @ latest/p1_a_h_no_mask_3_class_30/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_no_mask_3_class_30.txt --net_cfg=cfg/p1_a_h_no_mask_3_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_no_mask_3_class_30/latest.pt --test_path=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_no_mask_3_class_30/latest_on_p1_a_h_3_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted score_thresholds=0:1:0.001

<a id="p1_a_h_no_mask_4_class_30___test_"></a>
## p1_a_h_no_mask_4_class_30       @ test

<a id="best_375___p1_a_h_no_mask_4_class_30_test_"></a>
### best_375       @ p1_a_h_no_mask_4_class_30/test

<a id="p1_a_h_4_class_30_inverted___best_375_p1_a_h_no_mask_4_class_30_tes_t_"></a>
#### p1_a_h_4_class_30_inverted       @ best_375/p1_a_h_no_mask_4_class_30/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_no_mask_4_class_30.txt --net_cfg=cfg/p1_a_h_no_mask_4_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_no_mask_4_class_30/best_375.pt --test_path=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_no_mask_4_class_30/best_375_on_p1_a_h_4_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted score_thresholds=0:1:0.001

<a id="latest___p1_a_h_no_mask_4_class_30_test_"></a>
### latest       @ p1_a_h_no_mask_4_class_30/test

<a id="p1_a_h_4_class_30_inverted___latest_p1_a_h_no_mask_4_class_30_tes_t_"></a>
#### p1_a_h_4_class_30_inverted       @ latest/p1_a_h_no_mask_4_class_30/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_no_mask_4_class_30.txt --net_cfg=cfg/p1_a_h_no_mask_4_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_no_mask_4_class_30/latest.pt --test_path=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_no_mask_4_class_30/latest_on_p1_a_h_4_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted score_thresholds=0:1:0.001


<a id="p1_a_h_3_class_30___test_"></a>
## p1_a_h_3_class_30       @ test

<a id="best_556___p1_a_h_3_class_30_test_"></a>
### best_556       @ p1_a_h_3_class_30/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_3_class_30.txt --net_cfg=cfg/p1_a_h_3_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_3_class_30/best_556.pt --test_path=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_3_class_30/best_556_on_p1_a_h_3_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted score_thresholds=0:1:0.001

<a id="latest___p1_a_h_3_class_30_test_"></a>
### latest       @ p1_a_h_3_class_30/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_3_class_30.txt --net_cfg=cfg/p1_a_h_3_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_3_class_30/latest.pt --test_path=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_3_class_30/latest_on_p1_a_h_3_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted score_thresholds=0:1:0.001


<a id="p1_a_h_4_class_30___test_"></a>
## p1_a_h_4_class_30       @ test

<a id="best_1740___p1_a_h_4_class_30_test_"></a>
### best_1740       @ p1_a_h_4_class_30/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_4_class_30.txt --net_cfg=cfg/p1_a_h_4_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_4_class_30/best_1740.pt --test_path=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_4_class_30/best_1740_on_p1_a_h_4_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted score_thresholds=0:1:0.001 show_img=1

<a id="latest___p1_a_h_4_class_30_test_"></a>
### latest       @ p1_a_h_4_class_30/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_4_class_30.txt --net_cfg=cfg/p1_a_h_4_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_4_class_30/latest.pt --test_path=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_4_class_30/latest_on_p1_a_h_4_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted score_thresholds=0:1:0.001


<a id="p1_a_h_4_class_30_rt___test_"></a>
## p1_a_h_4_class_30_rt       @ test

<a id="best_454___p1_a_h_4_class_30_rt_tes_t_"></a>
### best_454       @ p1_a_h_4_class_30_rt/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_4_class_30.txt --net_cfg=cfg/p1_a_h_4_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_4_class_30_rt/best_454.pt --test_path=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_4_class_30_rt/best_454_on_p1_a_h_4_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted score_thresholds=0:1:0.001

<a id="latest___p1_a_h_4_class_30_rt_tes_t_"></a>
### latest       @ p1_a_h_4_class_30_rt/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_4_class_30.txt --net_cfg=cfg/p1_a_h_4_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_4_class_30_rt/latest.pt --test_path=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_4_class_30_rt/latest_on_p1_a_h_4_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted score_thresholds=0:1:0.001

<a id="p1_a_h_no_mask_bear_3_frg_per_bkg___test_"></a>
## p1_a_h_no_mask_bear_3_frg_per_bkg       @ test

<a id="best_874___p1_a_h_no_mask_bear_3_frg_per_bkg_test_"></a>
### best_874       @ p1_a_h_no_mask_bear_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_no_mask_bear_3_frg_per_bkg.txt --net_cfg=cfg/p1_a_h_no_mask_bear_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_no_mask_bear_3_frg_per_bkg/best_874.pt --test_path=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_bear.pbtxt n_frames=0 show_img=0 n_classes=1 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_bear.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_no_mask_bear_3_frg_per_bkg/best_874_on_p1_a_h_bear_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv score_thresholds=0:1:0.001 show_img=0

<a id="latest___p1_a_h_no_mask_bear_3_frg_per_bkg_test_"></a>
### latest       @ p1_a_h_no_mask_bear_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/p1_a_h_no_mask_bear_3_frg_per_bkg.txt --net_cfg=cfg/p1_a_h_no_mask_bear_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_no_mask_bear_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_bear.pbtxt n_frames=0 show_img=0 n_classes=1 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_bear.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_no_mask_bear_3_frg_per_bkg/latest_on_p1_a_h_bear_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___test_"></a>
## p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ test

<a id="best_685___p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg_tes_t_"></a>
### best_685       @ p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --net_cfg=cfg/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/best_685.pt --test_path=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_bear.pbtxt n_frames=0 show_img=0 n_classes=1 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_bear.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/best_685_on_p1_a_h_bear_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv score_thresholds=0:1:0.001 show_img=0

<a id="latest___p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg_tes_t_"></a>
### latest       @ p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --net_cfg=cfg/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_bear.pbtxt n_frames=0 show_img=0 n_classes=1 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_bear.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/latest_on_p1_a_h_bear_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv score_thresholds=0:1:0.001


<a id="p1_3_class_a_h_3_frg_per_bkg___test_"></a>
## p1_3_class_a_h_3_frg_per_bkg       @ test

<a id="best_927___p1_3_class_a_h_3_frg_per_bkg_tes_t_"></a>
### best_927       @ p1_3_class_a_h_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_3_class_a_h_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_3_class_a_h_3_frg_per_bkg/best_927.pt --test_path=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_3_class_a_h_3_frg_per_bkg/best_927_on_p1_3_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="latest___p1_3_class_a_h_3_frg_per_bkg_tes_t_"></a>
### latest       @ p1_3_class_a_h_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_3_class_a_h_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_3_class_a_h_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_3_class_a_h_3_frg_per_bkg/latest_on_p1_3_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001


<a id="p1_4_class_a_h_3_frg_per_bkg___test_"></a>
## p1_4_class_a_h_3_frg_per_bkg       @ test

<a id="best_909___p1_4_class_a_h_3_frg_per_bkg_tes_t_"></a>
### best_909       @ p1_4_class_a_h_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_4_class_a_h_3_frg_per_bkg.txt --net_cfg=cfg/p1_a_h_4_class_30.cfg --batch_size=32 --weights=yolov3_weights/p1_a_h_4_class_30/best_1740.pt --test_path=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inverted.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_a_h_4_class_30/best_1740_on_p1_a_h_4_class_30_inverted/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted score_thresholds=0:1:0.001 show_img=1

<a id="latest___p1_4_class_a_h_3_frg_per_bkg_tes_t_"></a>
### latest       @ p1_4_class_a_h_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/p1_4_class_a_h_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_4_class_a_h_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_4_class_a_h_3_frg_per_bkg/latest_on_p1_4_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_no_mask_3_frg_per_bkg___test_"></a>
## p1_3_class_a_h_no_mask_3_frg_per_bkg       @ test

<a id="latest_on_p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_no_mask_3_frg_per_bkg_tes_t_"></a>
### latest_on_p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_no_mask_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_3_class_a_h_no_mask_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_no_mask_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_3_class_a_h_no_mask_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_3_class_a_h_no_mask_3_frg_per_bkg/latest_on_p1_3_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="p1_4_class_a_h_no_mask_3_frg_per_bkg___test_"></a>
## p1_4_class_a_h_no_mask_3_frg_per_bkg       @ test

<a id="latest_on_p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_no_mask_3_frg_per_bkg_tes_t_"></a>
### latest_on_p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_no_mask_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=0 python3 yolov3_test.py --data_cfg=cfg/p1_4_class_a_h_no_mask_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_no_mask_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_4_class_a_h_no_mask_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_4_class_a_h_no_mask_3_frg_per_bkg/latest_on_p1_4_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___test_"></a>
## p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ test

<a id="latest_on_p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tes_t_"></a>
### latest_on_p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/latest_on_p1_3_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___test_"></a>
## p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ test

<a id="latest_on_p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tes_t_"></a>
### latest_on_p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/latest_on_p1_4_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_siam_mask_3_frg_per_bkg___test_"></a>
## p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ test

<a id="latest_on_p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_siam_mask_3_frg_per_bkg_tes_t_"></a>
### latest_on_p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_siam_mask_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=2 python3 yolov3_test.py --data_cfg=cfg/p1_3_class_a_h_siam_mask_3_frg_per_bkg.txt --net_cfg=cfg/p1_3_class_a_h_siam_mask_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_3_class_a_h_siam_mask_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k3a.pbtxt n_frames=0 show_img=0 n_classes=3 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_3_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_3_class_a_h_siam_mask_3_frg_per_bkg/latest_on_p1_3_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="p1_4_class_a_h_siam_mask_3_frg_per_bkg___test_"></a>
## p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ test

<a id="latest_on_p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_siam_mask_3_frg_per_bkg_tes_t_"></a>
### latest_on_p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_siam_mask_3_frg_per_bkg/test

CUDA_VISIBLE_DEVICES=1 python3 yolov3_test.py --data_cfg=cfg/p1_4_class_a_h_siam_mask_3_frg_per_bkg.txt --net_cfg=cfg/p1_4_class_a_h_siam_mask_3_frg_per_bkg.cfg --batch_size=32 --weights=yolov3_weights/p1_4_class_a_h_siam_mask_3_frg_per_bkg/latest.pt --test_path=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv.txt

CUDA_VISIBLE_DEVICES=1 python3 ../tf_api/tf_api_eval.py labels_path=../tf_api/data/wildlife_label_map_20k4.pbtxt n_frames=0 show_img=0 n_classes=4 eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=../tf_api/p1_a_h_4_class.txt sleep_time=10 write_summary=1 save_det=1 load_det=1 load_dir=yolov3_weights/p1_4_class_a_h_siam_mask_3_frg_per_bkg/latest_on_p1_4_class_a_h_3_frg_per_bkg_inv/ load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001








