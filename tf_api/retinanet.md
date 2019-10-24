<!-- MarkdownTOC -->

- [train](#train_)
    - [p1_a_h_no_mask_bear_3_frg_per_bkg       @ train](#p1_a_h_no_mask_bear_3_frg_per_bkg___trai_n_)
    - [p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ train](#p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___trai_n_)
    - [20k_bear       @ train](#20k_bear___trai_n_)
    - [20k_deer       @ train](#20k_deer___trai_n_)
    - [20k_coyote       @ train](#20k_coyote___trai_n_)
    - [20k_moose       @ train](#20k_moose___trai_n_)
    - [20k_elk       @ train](#20k_elk___trai_n_)
    - [20k_bison       @ train](#20k_bison___trai_n_)
    - [no_static__bear       @ train](#no_static_bear___trai_n_)
    - [no_static__deer       @ train](#no_static_deer___trai_n_)
    - [no_static_coyote       @ train](#no_static_coyote___trai_n_)
    - [40K_3       @ train](#40k_3___trai_n_)
    - [1K_3a       @ train](#1k_3a___trai_n_)
    - [1K_3a_sampled       @ train](#1k_3a_sampled___trai_n_)
        - [mp       @ 1K_3a_sampled/train](#mp___1k_3a_sampled_trai_n_)
    - [1K_3a_sampled_score_threshold_0       @ train](#1k_3a_sampled_score_threshold_0___trai_n_)
    - [500_static3a       @ train](#500_static3a___trai_n_)
        - [windows_b32       @ 500_static3a/train](#windows_b32___500_static3a_train_)
    - [200_static3a       @ train](#200_static3a___trai_n_)
        - [win       @ 200_static3a/train](#win___200_static3a_train_)
    - [no_static3a       @ train](#no_static3a___trai_n_)
    - [1600_static3       @ train](#1600_static3___trai_n_)
    - [20K_vid3a       @ train](#20k_vid3a___trai_n_)
    - [40K_3a       @ train](#40k_3a___trai_n_)
        - [rt       @ 40K_3a/train](#rt___40k_3a_train_)
    - [p1_3_class_a_h_3_frg_per_bkg       @ train](#p1_3_class_a_h_3_frg_per_bkg___trai_n_)
    - [p1_3_class_a_h_no_mask_3_frg_per_bkg       @ train](#p1_3_class_a_h_no_mask_3_frg_per_bkg___trai_n_)
    - [p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ train](#p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___trai_n_)
    - [p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ train](#p1_3_class_a_h_siam_mask_3_frg_per_bkg___trai_n_)
    - [p1_a_h_3_class_30       @ train](#p1_a_h_3_class_30___trai_n_)
    - [p1_a_h_3_class_70       @ train](#p1_a_h_3_class_70___trai_n_)
    - [p1_a_h_3_class_100       @ train](#p1_a_h_3_class_100___trai_n_)
    - [p1_a_h_no_mask_3_class_10       @ train](#p1_a_h_no_mask_3_class_10___trai_n_)
    - [p1_a_h_no_mask_3_class_30       @ train](#p1_a_h_no_mask_3_class_30___trai_n_)
    - [p1_a_h_no_mask_3_class_70       @ train](#p1_a_h_no_mask_3_class_70___trai_n_)
    - [20k4       @ train](#20k4___trai_n_)
    - [20K_vid4a       @ train](#20k_vid4a___trai_n_)
    - [p1_4_class_a_h       @ train](#p1_4_class_a_h___trai_n_)
    - [p1_4_class_a_h_3_frg_per_bkg       @ train](#p1_4_class_a_h_3_frg_per_bkg___trai_n_)
    - [p1_4_class_a_h_no_mask_3_frg_per_bkg       @ train](#p1_4_class_a_h_no_mask_3_frg_per_bkg___trai_n_)
    - [p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ train](#p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___trai_n_)
    - [p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ train](#p1_4_class_a_h_siam_mask_3_frg_per_bkg___trai_n_)
    - [p1_a_h_4_class_30       @ train](#p1_a_h_4_class_30___trai_n_)
    - [p1_a_h_4_class_50       @ train](#p1_a_h_4_class_50___trai_n_)
    - [p1_a_h_4_class_70       @ train](#p1_a_h_4_class_70___trai_n_)
    - [p1_a_h_no_mask_4_class_30       @ train](#p1_a_h_no_mask_4_class_30___trai_n_)
    - [p1_a_h_no_mask_4_class_10       @ train](#p1_a_h_no_mask_4_class_10___trai_n_)
    - [p1_a_h_no_mask_4_class_70       @ train](#p1_a_h_no_mask_4_class_70___trai_n_)
    - [20K6       @ train](#20k6___trai_n_)
        - [320x320       @ 20K6/train](#320x320___20k6_train_)
    - [20K6_5       @ train](#20k6_5___trai_n_)
    - [20K6_60       @ train](#20k6_60___trai_n_)
    - [10k6_entire_seq       @ train](#10k6_entire_seq___trai_n_)
    - [15k6_entire_seq       @ train](#15k6_entire_seq___trai_n_)
    - [1k6_vid_entire_seq       @ train](#1k6_vid_entire_seq___trai_n_)
    - [10k6_vid_entire_seq       @ train](#10k6_vid_entire_seq___trai_n_)
    - [15k6_vid_entire_seq       @ train](#15k6_vid_entire_seq___trai_n_)
    - [1_per_seq_6_class_vid_67_b32       @ train](#1_per_seq_6_class_vid_67_b32___trai_n_)
    - [2_per_seq_6_class_vid_67_b32       @ train](#2_per_seq_6_class_vid_67_b32___trai_n_)
    - [5_per_seq_6_class_vid_67_b32       @ train](#5_per_seq_6_class_vid_67_b32___trai_n_)
    - [10_per_seq_6_class_vid_67_b32       @ train](#10_per_seq_6_class_vid_67_b32___trai_n_)
    - [acamp_switch_6_class       @ train](#acamp_switch_6_class___trai_n_)
        - [100_per_class       @ acamp_switch_6_class/train](#100_per_class___acamp_switch_6_class_train_)
            - [individual       @ 100_per_class/acamp_switch_6_class/train](#individual___100_per_class_acamp_switch_6_class_train_)
            - [combined       @ 100_per_class/acamp_switch_6_class/train](#combined___100_per_class_acamp_switch_6_class_train_)
        - [20_per_class       @ acamp_switch_6_class/train](#20_per_class___acamp_switch_6_class_train_)
            - [individual       @ 20_per_class/acamp_switch_6_class/train](#individual___20_per_class_acamp_switch_6_class_trai_n_)
            - [combined       @ 20_per_class/acamp_switch_6_class/train](#combined___20_per_class_acamp_switch_6_class_trai_n_)
    - [4k8       @ train](#4k8___trai_n_)
    - [1k8_vid       @ train](#1k8_vid___trai_n_)
    - [1k8_vid_even_min_1       @ train](#1k8_vid_even_min_1___trai_n_)
    - [1k8_vid_entire_seq       @ train](#1k8_vid_entire_seq___trai_n_)
- [eval](#eva_l_)
    - [1K_3a       @ eval](#1k_3a___eval_)
- [tf_api_eval](#tf_api_eval_)
    - [1600_static_3       @ tf_api_eval](#1600_static_3___tf_api_eva_l_)
        - [on_train       @ 1600_static_3/tf_api_eval](#on_train___1600_static_3_tf_api_eva_l_)
            - [person_coco17_1600       @ on_train/1600_static_3/tf_api_eval](#person_coco17_1600___on_train_1600_static_3_tf_api_eval_)
        - [acamp_no_static_2_class       @ 1600_static_3/tf_api_eval](#acamp_no_static_2_class___1600_static_3_tf_api_eva_l_)
    - [40k3       @ tf_api_eval](#40k3___tf_api_eva_l_)
        - [on_train       @ 40k3/tf_api_eval](#on_train___40k3_tf_api_eval_)
            - [no_human       @ on_train/40k3/tf_api_eval](#no_human___on_train_40k3_tf_api_eva_l_)
        - [acamp_no_static_2_class       @ 40k3/tf_api_eval](#acamp_no_static_2_class___40k3_tf_api_eval_)
    - [p1_a_h_no_mask_bear_3_frg_per_bkg       @ tf_api_eval](#p1_a_h_no_mask_bear_3_frg_per_bkg___tf_api_eva_l_)
    - [p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ tf_api_eval](#p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___tf_api_eva_l_)
    - [p1_3_class_a_h_3_frg_per_bkg       @ tf_api_eval](#p1_3_class_a_h_3_frg_per_bkg___tf_api_eva_l_)
        - [inverted       @ p1_3_class_a_h_3_frg_per_bkg/tf_api_eval](#inverted___p1_3_class_a_h_3_frg_per_bkg_tf_api_eval_)
    - [p1_3_class_a_h_no_mask_3_frg_per_bkg       @ tf_api_eval](#p1_3_class_a_h_no_mask_3_frg_per_bkg___tf_api_eva_l_)
        - [p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_no_mask_3_frg_per_bkg/tf_api_eval](#p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_no_mask_3_frg_per_bkg_tf_api_eval_)
    - [p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ tf_api_eval](#p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___tf_api_eva_l_)
        - [p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/tf_api_eval](#p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tf_api_eval_)
    - [p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ tf_api_eval](#p1_3_class_a_h_siam_mask_3_frg_per_bkg___tf_api_eva_l_)
        - [p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_siam_mask_3_frg_per_bkg/tf_api_eval](#p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_siam_mask_3_frg_per_bkg_tf_api_eval_)
    - [p1_a_h_no_mask_3_class_30       @ tf_api_eval](#p1_a_h_no_mask_3_class_30___tf_api_eva_l_)
        - [p1_a_h_3_class_30_inverted       @ p1_a_h_no_mask_3_class_30/tf_api_eval](#p1_a_h_3_class_30_inverted___p1_a_h_no_mask_3_class_30_tf_api_eva_l_)
    - [p1_a_h_3_class_30       @ tf_api_eval](#p1_a_h_3_class_30___tf_api_eva_l_)
        - [p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_a_h_3_class_30/tf_api_eval](#p1_3_class_a_h_3_frg_per_bkg_inv___p1_a_h_3_class_30_tf_api_eva_l_)
        - [on_train_100       @ p1_a_h_3_class_30/tf_api_eval](#on_train_100___p1_a_h_3_class_30_tf_api_eva_l_)
        - [acamp1K_static3a_test       @ p1_a_h_3_class_30/tf_api_eval](#acamp1k_static3a_test___p1_a_h_3_class_30_tf_api_eva_l_)
        - [acamp_no_static_3_class       @ p1_a_h_3_class_30/tf_api_eval](#acamp_no_static_3_class___p1_a_h_3_class_30_tf_api_eva_l_)
        - [on_inverted       @ p1_a_h_3_class_30/tf_api_eval](#on_inverted___p1_a_h_3_class_30_tf_api_eva_l_)
    - [p1_a_h_3_class_70       @ tf_api_eval](#p1_a_h_3_class_70___tf_api_eva_l_)
        - [on_inverted       @ p1_a_h_3_class_70/tf_api_eval](#on_inverted___p1_a_h_3_class_70_tf_api_eva_l_)
    - [p1_a_h_3_class_100       @ tf_api_eval](#p1_a_h_3_class_100___tf_api_eva_l_)
        - [acamp1K_static3a_test       @ p1_a_h_3_class_100/tf_api_eval](#acamp1k_static3a_test___p1_a_h_3_class_100_tf_api_eval_)
        - [on_train       @ p1_a_h_3_class_100/tf_api_eval](#on_train___p1_a_h_3_class_100_tf_api_eval_)
        - [acamp_no_static_3_class       @ p1_a_h_3_class_100/tf_api_eval](#acamp_no_static_3_class___p1_a_h_3_class_100_tf_api_eval_)
    - [1K_3a       @ tf_api_eval](#1k_3a___tf_api_eva_l_)
        - [acamp1K_static3a_test       @ 1K_3a/tf_api_eval](#acamp1k_static3a_test___1k_3a_tf_api_eva_l_)
        - [acamp_no_static_3_class       @ 1K_3a/tf_api_eval](#acamp_no_static_3_class___1k_3a_tf_api_eva_l_)
        - [p1_h_3_class       @ 1K_3a/tf_api_eval](#p1_h_3_class___1k_3a_tf_api_eva_l_)
        - [p1_a_3_class       @ 1K_3a/tf_api_eval](#p1_a_3_class___1k_3a_tf_api_eva_l_)
        - [p1_a_h_3_class_30_inverted       @ 1K_3a/tf_api_eval](#p1_a_h_3_class_30_inverted___1k_3a_tf_api_eva_l_)
        - [p1_a_h_3_class_70_inverted       @ 1K_3a/tf_api_eval](#p1_a_h_3_class_70_inverted___1k_3a_tf_api_eva_l_)
        - [p1_a_h_3_class_100       @ 1K_3a/tf_api_eval](#p1_a_h_3_class_100___1k_3a_tf_api_eva_l_)
    - [1K_3a_sampled       @ tf_api_eval](#1k_3a_sampled___tf_api_eva_l_)
        - [inverted       @ 1K_3a_sampled/tf_api_eval](#inverted___1k_3a_sampled_tf_api_eva_l_)
            - [map_plotting       @ inverted/1K_3a_sampled/tf_api_eval](#map_plotting___inverted_1k_3a_sampled_tf_api_eval_)
            - [win       @ inverted/1K_3a_sampled/tf_api_eval](#win___inverted_1k_3a_sampled_tf_api_eval_)
        - [all_vid       @ 1K_3a_sampled/tf_api_eval](#all_vid___1k_3a_sampled_tf_api_eva_l_)
        - [p1_3_class_a_h_3_frg_per_bkg_inv       @ 1K_3a_sampled/tf_api_eval](#p1_3_class_a_h_3_frg_per_bkg_inv___1k_3a_sampled_tf_api_eva_l_)
    - [1K_3a_sampled_score_threshold_0       @ tf_api_eval](#1k_3a_sampled_score_threshold_0___tf_api_eva_l_)
        - [inverted       @ 1K_3a_sampled_score_threshold_0/tf_api_eval](#inverted___1k_3a_sampled_score_threshold_0_tf_api_eva_l_)
    - [500_static3a       @ tf_api_eval](#500_static3a___tf_api_eva_l_)
        - [class_agnostic       @ 500_static3a/tf_api_eval](#class_agnostic___500_static3a_tf_api_eval_)
        - [nms_thresh_0       @ 500_static3a/tf_api_eval](#nms_thresh_0___500_static3a_tf_api_eval_)
            - [bgr_to_rgb_0       @ nms_thresh_0/500_static3a/tf_api_eval](#bgr_to_rgb_0___nms_thresh_0_500_static3a_tf_api_eva_l_)
    - [200_static3a       @ tf_api_eval](#200_static3a___tf_api_eva_l_)
        - [inverted       @ 200_static3a/tf_api_eval](#inverted___200_static3a_tf_api_eval_)
            - [inclusive       @ inverted/200_static3a/tf_api_eval](#inclusive___inverted_200_static3a_tf_api_eva_l_)
            - [map_plotting       @ inverted/200_static3a/tf_api_eval](#map_plotting___inverted_200_static3a_tf_api_eva_l_)
            - [win       @ inverted/200_static3a/tf_api_eval](#win___inverted_200_static3a_tf_api_eva_l_)
    - [20K_vid3a       @ tf_api_eval](#20k_vid3a___tf_api_eva_l_)
        - [acamp200_static3a_inverted       @ 20K_vid3a/tf_api_eval](#acamp200_static3a_inverted___20k_vid3a_tf_api_eva_l_)
        - [all_static       @ 20K_vid3a/tf_api_eval](#all_static___20k_vid3a_tf_api_eva_l_)
            - [class_agnostic       @ all_static/20K_vid3a/tf_api_eval](#class_agnostic___all_static_20k_vid3a_tf_api_eval_)
    - [no_static3a       @ tf_api_eval](#no_static3a___tf_api_eva_l_)
        - [acamp_all_static_3_class       @ no_static3a/tf_api_eval](#acamp_all_static_3_class___no_static3a_tf_api_eva_l_)
        - [p1_h_3_class       @ no_static3a/tf_api_eval](#p1_h_3_class___no_static3a_tf_api_eva_l_)
        - [p1_a_3_class       @ no_static3a/tf_api_eval](#p1_a_3_class___no_static3a_tf_api_eva_l_)
    - [40k3a_rt       @ tf_api_eval](#40k3a_rt___tf_api_eva_l_)
        - [on_train       @ 40k3a_rt/tf_api_eval](#on_train___40k3a_rt_tf_api_eval_)
    - [40k3a       @ tf_api_eval](#40k3a___tf_api_eva_l_)
        - [on_train       @ 40k3a/tf_api_eval](#on_train___40k3a_tf_api_eva_l_)
        - [acamp_no_static_3_class       @ 40k3a/tf_api_eval](#acamp_no_static_3_class___40k3a_tf_api_eva_l_)
        - [acamp_all_static_3_class       @ 40k3a/tf_api_eval](#acamp_all_static_3_class___40k3a_tf_api_eva_l_)
        - [p1_h_3_class       @ 40k3a/tf_api_eval](#p1_h_3_class___40k3a_tf_api_eva_l_)
        - [p1_a_3_class       @ 40k3a/tf_api_eval](#p1_a_3_class___40k3a_tf_api_eva_l_)
    - [no_static__bear       @ tf_api_eval](#no_static_bear___tf_api_eva_l_)
        - [on_train       @ no_static__bear/tf_api_eval](#on_train___no_static_bear_tf_api_eval_)
        - [acamp_static_bear       @ no_static__bear/tf_api_eval](#acamp_static_bear___no_static_bear_tf_api_eval_)
        - [acamp_static_3_class       @ no_static__bear/tf_api_eval](#acamp_static_3_class___no_static_bear_tf_api_eval_)
    - [no_static__deer       @ tf_api_eval](#no_static_deer___tf_api_eva_l_)
        - [on_train       @ no_static__deer/tf_api_eval](#on_train___no_static_deer_tf_api_eval_)
        - [acamp_static_deer       @ no_static__deer/tf_api_eval](#acamp_static_deer___no_static_deer_tf_api_eval_)
        - [acamp_static_3_class       @ no_static__deer/tf_api_eval](#acamp_static_3_class___no_static_deer_tf_api_eval_)
    - [no_static_coyote       @ tf_api_eval](#no_static_coyote___tf_api_eva_l_)
        - [on_train       @ no_static_coyote/tf_api_eval](#on_train___no_static_coyote_tf_api_eval_)
        - [acamp_static_coyote       @ no_static_coyote/tf_api_eval](#acamp_static_coyote___no_static_coyote_tf_api_eval_)
        - [acamp_static_3_class       @ no_static_coyote/tf_api_eval](#acamp_static_3_class___no_static_coyote_tf_api_eval_)
    - [20k_bear       @ tf_api_eval](#20k_bear___tf_api_eva_l_)
        - [20k4_inverted       @ 20k_bear/tf_api_eval](#20k4_inverted___20k_bear_tf_api_eval_)
        - [acamp_all_static_3_class       @ 20k_bear/tf_api_eval](#acamp_all_static_3_class___20k_bear_tf_api_eval_)
        - [20k6_inverted       @ 20k_bear/tf_api_eval](#20k6_inverted___20k_bear_tf_api_eval_)
            - [only_video       @ 20k6_inverted/20k_bear/tf_api_eval](#only_video___20k6_inverted_20k_bear_tf_api_eval_)
    - [20k_deer       @ tf_api_eval](#20k_deer___tf_api_eva_l_)
        - [on_inverted       @ 20k_deer/tf_api_eval](#on_inverted___20k_deer_tf_api_eval_)
        - [20k4_inverted       @ 20k_deer/tf_api_eval](#20k4_inverted___20k_deer_tf_api_eval_)
        - [acamp_all_static_3_class       @ 20k_deer/tf_api_eval](#acamp_all_static_3_class___20k_deer_tf_api_eval_)
        - [20k6_inverted       @ 20k_deer/tf_api_eval](#20k6_inverted___20k_deer_tf_api_eval_)
            - [only_video       @ 20k6_inverted/20k_deer/tf_api_eval](#only_video___20k6_inverted_20k_deer_tf_api_eval_)
    - [20k_coyote       @ tf_api_eval](#20k_coyote___tf_api_eva_l_)
        - [on_inverted       @ 20k_coyote/tf_api_eval](#on_inverted___20k_coyote_tf_api_eval_)
        - [20k4_inverted       @ 20k_coyote/tf_api_eval](#20k4_inverted___20k_coyote_tf_api_eval_)
        - [acamp_all_static_3_class       @ 20k_coyote/tf_api_eval](#acamp_all_static_3_class___20k_coyote_tf_api_eval_)
        - [20k6_inverted       @ 20k_coyote/tf_api_eval](#20k6_inverted___20k_coyote_tf_api_eval_)
            - [only_video       @ 20k6_inverted/20k_coyote/tf_api_eval](#only_video___20k6_inverted_20k_coyote_tf_api_eval_)
    - [20k_moose       @ tf_api_eval](#20k_moose___tf_api_eva_l_)
        - [on_inverted       @ 20k_moose/tf_api_eval](#on_inverted___20k_moose_tf_api_eva_l_)
        - [20k4_inverted       @ 20k_moose/tf_api_eval](#20k4_inverted___20k_moose_tf_api_eva_l_)
        - [20k6_inverted       @ 20k_moose/tf_api_eval](#20k6_inverted___20k_moose_tf_api_eva_l_)
            - [only_video       @ 20k6_inverted/20k_moose/tf_api_eval](#only_video___20k6_inverted_20k_moose_tf_api_eva_l_)
    - [20k_elk       @ tf_api_eval](#20k_elk___tf_api_eva_l_)
        - [on_inverted       @ 20k_elk/tf_api_eval](#on_inverted___20k_elk_tf_api_eva_l_)
        - [20k6_inverted       @ 20k_elk/tf_api_eval](#20k6_inverted___20k_elk_tf_api_eva_l_)
            - [only_video       @ 20k6_inverted/20k_elk/tf_api_eval](#only_video___20k6_inverted_20k_elk_tf_api_eva_l_)
    - [20k_bison       @ tf_api_eval](#20k_bison___tf_api_eva_l_)
        - [on_inverted       @ 20k_bison/tf_api_eval](#on_inverted___20k_bison_tf_api_eva_l_)
        - [20k6_inverted       @ 20k_bison/tf_api_eval](#20k6_inverted___20k_bison_tf_api_eva_l_)
            - [only_video       @ 20k6_inverted/20k_bison/tf_api_eval](#only_video___20k6_inverted_20k_bison_tf_api_eva_l_)
    - [20k4       @ tf_api_eval](#20k4___tf_api_eva_l_)
        - [on_inverted       @ 20k4/tf_api_eval](#on_inverted___20k4_tf_api_eval_)
        - [on_train       @ 20k4/tf_api_eval](#on_train___20k4_tf_api_eval_)
        - [p1_a_h_4_class_30_inverted       @ 20k4/tf_api_eval](#p1_a_h_4_class_30_inverted___20k4_tf_api_eval_)
        - [p1_a_h_4_class_50_inverted       @ 20k4/tf_api_eval](#p1_a_h_4_class_50_inverted___20k4_tf_api_eval_)
        - [p1_4_class_a_h_3_frg_per_bkg_inv       @ 20k4/tf_api_eval](#p1_4_class_a_h_3_frg_per_bkg_inv___20k4_tf_api_eval_)
        - [prototype_1_vid_4_class       @ 20k4/tf_api_eval](#prototype_1_vid_4_class___20k4_tf_api_eval_)
    - [20K_vid4a       @ tf_api_eval](#20k_vid4a___tf_api_eva_l_)
        - [p1_4_class_a_h_3_frg_per_bkg_inv       @ 20K_vid4a/tf_api_eval](#p1_4_class_a_h_3_frg_per_bkg_inv___20k_vid4a_tf_api_eva_l_)
    - [p1_4_class_a_h_3_frg_per_bkg       @ tf_api_eval](#p1_4_class_a_h_3_frg_per_bkg___tf_api_eva_l_)
        - [inverted       @ p1_4_class_a_h_3_frg_per_bkg/tf_api_eval](#inverted___p1_4_class_a_h_3_frg_per_bkg_tf_api_eval_)
        - [prototype_1_vid_4_class       @ p1_4_class_a_h_3_frg_per_bkg/tf_api_eval](#prototype_1_vid_4_class___p1_4_class_a_h_3_frg_per_bkg_tf_api_eval_)
            - [vis       @ prototype_1_vid_4_class/p1_4_class_a_h_3_frg_per_bkg/tf_api_eval](#vis___prototype_1_vid_4_class_p1_4_class_a_h_3_frg_per_bkg_tf_api_eval_)
        - [prototype_1_vid_4_class_no_moving_bkg       @ p1_4_class_a_h_3_frg_per_bkg/tf_api_eval](#prototype_1_vid_4_class_no_moving_bkg___p1_4_class_a_h_3_frg_per_bkg_tf_api_eval_)
    - [p1_4_class_a_h_no_mask_3_frg_per_bkg       @ tf_api_eval](#p1_4_class_a_h_no_mask_3_frg_per_bkg___tf_api_eva_l_)
        - [p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_no_mask_3_frg_per_bkg/tf_api_eval](#p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_no_mask_3_frg_per_bkg_tf_api_eval_)
    - [p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ tf_api_eval](#p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___tf_api_eva_l_)
        - [p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/tf_api_eval](#p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tf_api_eval_)
        - [p1_a_h_4_class_70_inverted       @ p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/tf_api_eval](#p1_a_h_4_class_70_inverted___p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tf_api_eval_)
    - [p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ tf_api_eval](#p1_4_class_a_h_siam_mask_3_frg_per_bkg___tf_api_eva_l_)
        - [p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_siam_mask_3_frg_per_bkg/tf_api_eval](#p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_siam_mask_3_frg_per_bkg_tf_api_eval_)
    - [p1_a_h_no_mask_4_class_30       @ tf_api_eval](#p1_a_h_no_mask_4_class_30___tf_api_eva_l_)
        - [p1_a_h_4_class_30_inverted       @ p1_a_h_no_mask_4_class_30/tf_api_eval](#p1_a_h_4_class_30_inverted___p1_a_h_no_mask_4_class_30_tf_api_eva_l_)
    - [p1_a_h_4_class_30       @ tf_api_eval](#p1_a_h_4_class_30___tf_api_eva_l_)
        - [on_inverted       @ p1_a_h_4_class_30/tf_api_eval](#on_inverted___p1_a_h_4_class_30_tf_api_eva_l_)
        - [p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_a_h_4_class_30/tf_api_eval](#p1_4_class_a_h_3_frg_per_bkg_inv___p1_a_h_4_class_30_tf_api_eva_l_)
    - [p1_a_h_4_class_50       @ tf_api_eval](#p1_a_h_4_class_50___tf_api_eva_l_)
        - [on_inverted       @ p1_a_h_4_class_50/tf_api_eval](#on_inverted___p1_a_h_4_class_50_tf_api_eva_l_)
    - [p1_a_h_4_class_70       @ tf_api_eval](#p1_a_h_4_class_70___tf_api_eva_l_)
        - [on_inverted       @ p1_a_h_4_class_70/tf_api_eval](#on_inverted___p1_a_h_4_class_70_tf_api_eva_l_)
    - [20k6       @ tf_api_eval](#20k6___tf_api_eva_l_)
        - [on_train       @ 20k6/tf_api_eval](#on_train___20k6_tf_api_eval_)
        - [custom_folder       @ 20k6/tf_api_eval](#custom_folder___20k6_tf_api_eval_)
    - [20k6_320       @ tf_api_eval](#20k6_320___tf_api_eva_l_)
        - [acamp_all_coyote       @ 20k6_320/tf_api_eval](#acamp_all_coyote___20k6_320_tf_api_eval_)
        - [acamp_all_deer       @ 20k6_320/tf_api_eval](#acamp_all_deer___20k6_320_tf_api_eval_)
    - [20k6_5       @ tf_api_eval](#20k6_5___tf_api_eva_l_)
        - [inverted       @ 20k6_5/tf_api_eval](#inverted___20k6_5_tf_api_eval_)
            - [combine_sequences_640x360       @ inverted/20k6_5/tf_api_eval](#combine_sequences_640x360___inverted_20k6_5_tf_api_eva_l_)
            - [combine_sequences_416x416       @ inverted/20k6_5/tf_api_eval](#combine_sequences_416x416___inverted_20k6_5_tf_api_eva_l_)
            - [score_thresholds_0       @ inverted/20k6_5/tf_api_eval](#score_thresholds_0___inverted_20k6_5_tf_api_eva_l_)
        - [acamp_all_coyote       @ 20k6_5/tf_api_eval](#acamp_all_coyote___20k6_5_tf_api_eval_)
        - [only_video       @ 20k6_5/tf_api_eval](#only_video___20k6_5_tf_api_eval_)
            - [nms_thresh_0       @ only_video/20k6_5/tf_api_eval](#nms_thresh_0___only_video_20k6_5_tf_api_eva_l_)
                - [bgr       @ nms_thresh_0/only_video/20k6_5/tf_api_eval](#bgr___nms_thresh_0_only_video_20k6_5_tf_api_eval_)
    - [20k6_60       @ tf_api_eval](#20k6_60___tf_api_eva_l_)
        - [inverted       @ 20k6_60/tf_api_eval](#inverted___20k6_60_tf_api_eva_l_)
        - [vis       @ 20k6_60/tf_api_eval](#vis___20k6_60_tf_api_eva_l_)
            - [map_plotting       @ vis/20k6_60/tf_api_eval](#map_plotting___vis_20k6_60_tf_api_eva_l_)
        - [combine_sequences_640x360       @ 20k6_60/tf_api_eval](#combine_sequences_640x360___20k6_60_tf_api_eva_l_)
        - [acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval](#acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_)
    - [10k6_entire_seq       @ tf_api_eval](#10k6_entire_seq___tf_api_eva_l_)
    - [15k6_entire_seq       @ tf_api_eval](#15k6_entire_seq___tf_api_eva_l_)
        - [inverted       @ 15k6_entire_seq/tf_api_eval](#inverted___15k6_entire_seq_tf_api_eva_l_)
    - [1k6_vid_entire_seq       @ tf_api_eval](#1k6_vid_entire_seq___tf_api_eva_l_)
        - [inverted       @ 1k6_vid_entire_seq/tf_api_eval](#inverted___1k6_vid_entire_seq_tf_api_eval_)
    - [10k6_vid_entire_seq       @ tf_api_eval](#10k6_vid_entire_seq___tf_api_eva_l_)
        - [inverted       @ 10k6_vid_entire_seq/tf_api_eval](#inverted___10k6_vid_entire_seq_tf_api_eva_l_)
        - [nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval](#nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_)
    - [15k6_vid_entire_seq       @ tf_api_eval](#15k6_vid_entire_seq___tf_api_eva_l_)
        - [inverted       @ 15k6_vid_entire_seq/tf_api_eval](#inverted___15k6_vid_entire_seq_tf_api_eva_l_)
    - [1_per_seq_6_class_vid_67_b32       @ tf_api_eval](#1_per_seq_6_class_vid_67_b32___tf_api_eva_l_)
        - [inverted       @ 1_per_seq_6_class_vid_67_b32/tf_api_eval](#inverted___1_per_seq_6_class_vid_67_b32_tf_api_eval_)
        - [acamp_all_6_class_video_67       @ 1_per_seq_6_class_vid_67_b32/tf_api_eval](#acamp_all_6_class_video_67___1_per_seq_6_class_vid_67_b32_tf_api_eval_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 1_per_seq_6_class_vid_67_b32/tf_api_eval](#1_2_5_10_per_seq_6_class_vid_67_inverted___1_per_seq_6_class_vid_67_b32_tf_api_eval_)
    - [2_per_seq_6_class_vid_67_b32       @ tf_api_eval](#2_per_seq_6_class_vid_67_b32___tf_api_eva_l_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 2_per_seq_6_class_vid_67_b32/tf_api_eval](#1_2_5_10_per_seq_6_class_vid_67_inverted___2_per_seq_6_class_vid_67_b32_tf_api_eval_)
    - [5_per_seq_6_class_vid_67_b32       @ tf_api_eval](#5_per_seq_6_class_vid_67_b32___tf_api_eva_l_)
        - [inverted       @ 5_per_seq_6_class_vid_67_b32/tf_api_eval](#inverted___5_per_seq_6_class_vid_67_b32_tf_api_eval_)
        - [acamp_all_6_class_video_67       @ 5_per_seq_6_class_vid_67_b32/tf_api_eval](#acamp_all_6_class_video_67___5_per_seq_6_class_vid_67_b32_tf_api_eval_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 5_per_seq_6_class_vid_67_b32/tf_api_eval](#1_2_5_10_per_seq_6_class_vid_67_inverted___5_per_seq_6_class_vid_67_b32_tf_api_eval_)
    - [10_per_seq_6_class_vid_67_b32       @ tf_api_eval](#10_per_seq_6_class_vid_67_b32___tf_api_eva_l_)
        - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ 10_per_seq_6_class_vid_67_b32/tf_api_eval](#1_2_5_10_per_seq_6_class_vid_67_inverted___10_per_seq_6_class_vid_67_b32_tf_api_eva_l_)
    - [acamp_switch_6_class       @ tf_api_eval](#acamp_switch_6_class___tf_api_eva_l_)
        - [100_per_class       @ acamp_switch_6_class/tf_api_eval](#100_per_class___acamp_switch_6_class_tf_api_eval_)
            - [individual       @ 100_per_class/acamp_switch_6_class/tf_api_eval](#individual___100_per_class_acamp_switch_6_class_tf_api_eval_)
            - [1       @ 100_per_class/acamp_switch_6_class/tf_api_eval](#1___100_per_class_acamp_switch_6_class_tf_api_eval_)
            - [2       @ 100_per_class/acamp_switch_6_class/tf_api_eval](#2___100_per_class_acamp_switch_6_class_tf_api_eval_)
            - [3       @ 100_per_class/acamp_switch_6_class/tf_api_eval](#3___100_per_class_acamp_switch_6_class_tf_api_eval_)
            - [4       @ 100_per_class/acamp_switch_6_class/tf_api_eval](#4___100_per_class_acamp_switch_6_class_tf_api_eval_)
            - [5       @ 100_per_class/acamp_switch_6_class/tf_api_eval](#5___100_per_class_acamp_switch_6_class_tf_api_eval_)
            - [1_to_5       @ 100_per_class/acamp_switch_6_class/tf_api_eval](#1_to_5___100_per_class_acamp_switch_6_class_tf_api_eval_)
            - [combined       @ 100_per_class/acamp_switch_6_class/tf_api_eval](#combined___100_per_class_acamp_switch_6_class_tf_api_eval_)
        - [20_per_class       @ acamp_switch_6_class/tf_api_eval](#20_per_class___acamp_switch_6_class_tf_api_eval_)
            - [individual       @ 20_per_class/acamp_switch_6_class/tf_api_eval](#individual___20_per_class_acamp_switch_6_class_tf_api_eva_l_)
            - [1       @ 20_per_class/acamp_switch_6_class/tf_api_eval](#1___20_per_class_acamp_switch_6_class_tf_api_eva_l_)
            - [2       @ 20_per_class/acamp_switch_6_class/tf_api_eval](#2___20_per_class_acamp_switch_6_class_tf_api_eva_l_)
            - [3       @ 20_per_class/acamp_switch_6_class/tf_api_eval](#3___20_per_class_acamp_switch_6_class_tf_api_eva_l_)
            - [4       @ 20_per_class/acamp_switch_6_class/tf_api_eval](#4___20_per_class_acamp_switch_6_class_tf_api_eva_l_)
            - [5       @ 20_per_class/acamp_switch_6_class/tf_api_eval](#5___20_per_class_acamp_switch_6_class_tf_api_eva_l_)
            - [1_to_5       @ 20_per_class/acamp_switch_6_class/tf_api_eval](#1_to_5___20_per_class_acamp_switch_6_class_tf_api_eva_l_)
            - [combined       @ 20_per_class/acamp_switch_6_class/tf_api_eval](#combined___20_per_class_acamp_switch_6_class_tf_api_eva_l_)
    - [4k8       @ tf_api_eval](#4k8___tf_api_eva_l_)
    - [1k8_vid       @ tf_api_eval](#1k8_vid___tf_api_eva_l_)
    - [1k8_vid_entire_seq       @ tf_api_eval](#1k8_vid_entire_seq___tf_api_eva_l_)
        - [class_agnostic       @ 1k8_vid_entire_seq/tf_api_eval](#class_agnostic___1k8_vid_entire_seq_tf_api_eval_)
    - [1k8_vid_even_min_1       @ tf_api_eval](#1k8_vid_even_min_1___tf_api_eva_l_)
- [train_and_eval](#train_and_eva_l_)
    - [1k3a_static       @ train_and_eval](#1k3a_static___train_and_eval_)
    - [40k3a       @ train_and_eval](#40k3a___train_and_eval_)
        - [model_main       @ 40k3a/train_and_eval](#model_main___40k3a_train_and_eval_)
- [test](#tes_t_)
    - [pretrained       @ test](#pretrained___test_)
        - [camera       @ pretrained/test](#camera___pretrained_tes_t_)
            - [only_person       @ camera/pretrained/test](#only_person___camera_pretrained_test_)

<!-- /MarkdownTOC -->


<a id="train_"></a>
# train

<a id="p1_a_h_no_mask_bear_3_frg_per_bkg___trai_n_"></a>
## p1_a_h_no_mask_bear_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_bear_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_no_mask_bear_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___trai_n_"></a>
## p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="20k_bear___trai_n_"></a>
## 20k_bear       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bear.config train_dir=trained/retinanet_resnet50_v1_20k_bear n_steps=100000 save_interval_secs=600

<a id="20k_deer___trai_n_"></a>
## 20k_deer       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_deer.config train_dir=trained/retinanet_resnet50_v1_20k_deer n_steps=100000 save_interval_secs=600

<a id="20k_coyote___trai_n_"></a>
## 20k_coyote       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_coyote.config train_dir=trained/retinanet_resnet50_v1_20k_coyote n_steps=100000 save_interval_secs=600

<a id="20k_moose___trai_n_"></a>
## 20k_moose       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_moose.config train_dir=trained/retinanet_resnet50_v1_20k_moose n_steps=100000 save_interval_secs=600

<a id="20k_elk___trai_n_"></a>
## 20k_elk       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_elk.config train_dir=trained/retinanet_resnet50_v1_20k_elk n_steps=100000 save_interval_secs=600

<a id="20k_bison___trai_n_"></a>
## 20k_bison       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bison.config train_dir=trained/retinanet_resnet50_v1_20k_bison n_steps=100000 save_interval_secs=600

<a id="no_static_bear___trai_n_"></a>
## no_static__bear       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_bear.config train_dir=trained/retinanet_resnet50_v1_no_static_bear n_steps=100000 save_interval_secs=600

<a id="no_static_deer___trai_n_"></a>
## no_static__deer       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_deer.config train_dir=trained/retinanet_resnet50_v1_no_static_deer n_steps=100000 save_interval_secs=600

<a id="no_static_coyote___trai_n_"></a>
## no_static_coyote       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_coyote.config train_dir=trained/retinanet_resnet50_v1_no_static_coyote n_steps=100000 save_interval_secs=600

<a id="40k_3___trai_n_"></a>
## 40K_3       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3.config train_dir=trained/retinanet_resnet50_v1_40k3 n_steps=100000 save_interval_secs=600

<a id="1k_3a___trai_n_"></a>
## 1K_3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config train_dir=trained/retinanet_resnet50_v1_1k3a n_steps=100000 save_interval_secs=600


<a id="1k_3a_sampled___trai_n_"></a>
## 1K_3a_sampled       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled.config train_dir=trained/retinanet_resnet50_v1_1k3a_sampled n_steps=100000 save_interval_secs=600 


<a id="mp___1k_3a_sampled_trai_n_"></a>
### mp       @ 1K_3a_sampled/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled.config train_dir=trained/retinanet_resnet50_v1_1k3a_sampled_mp n_steps=100000 save_interval_secs=600 enable_mixed_precision=1


<a id="1k_3a_sampled_score_threshold_0___trai_n_"></a>
## 1K_3a_sampled_score_threshold_0       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled_score_threshold_0.config train_dir=trained/retinanet_resnet50_v1_1k3a_sampled_score_threshold_0 n_steps=100000 save_interval_secs=600

<a id="500_static3a___trai_n_"></a>
## 500_static3a       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_500_static3a.config train_dir=trained/retinanet_resnet50_v1_500_static3a n_steps=100000 save_interval_secs=600 

<a id="windows_b32___500_static3a_train_"></a>
### windows_b32       @ 500_static3a/train

python3 /models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_500_static3a_b32.config train_dir=trained/retinanet_resnet50_v1_500_static3a_b32 n_steps=100000 save_interval_secs=600 


<a id="200_static3a___trai_n_"></a>
## 200_static3a       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_200_static3a.config train_dir=trained/retinanet_resnet50_v1_200_static3a n_steps=100000 save_interval_secs=600

<a id="win___200_static3a_train_"></a>
### win       @ 200_static3a/train

CUDA_VISIBLE_DEVICES=2 python3 models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_200_static3a.config train_dir=trained/retinanet_resnet50_v1_200_static3a n_steps=100000 save_interval_secs=600

<a id="no_static3a___trai_n_"></a>
## no_static3a       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static3a.config train_dir=trained/retinanet_resnet50_v1_no_static3a n_steps=100000 save_interval_secs=600

<a id="1600_static3___trai_n_"></a>
## 1600_static3       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1600_static_3.config train_dir=trained/retinanet_resnet50_v1_1600_static_3 n_steps=100000 save_interval_secs=600

<a id="20k_vid3a___trai_n_"></a>
## 20K_vid3a       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_vid3a.config train_dir=trained/retinanet_resnet50_v1_20k_vid3a n_steps=100000 save_interval_secs=600

<a id="40k_3a___trai_n_"></a>
## 40K_3a       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config train_dir=trained/retinanet_resnet50_v1_40k3a n_steps=100000 save_interval_secs=600

<a id="rt___40k_3a_train_"></a>
### rt       @ 40K_3a/train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config train_dir=trained/retinanet_resnet50_v1_40k3a_rt n_steps=100000 save_interval_secs=600

<a id="p1_3_class_a_h_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_3_class_a_h_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_3_class_a_h_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_3_class_a_h_no_mask_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_no_mask_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_3_class_a_h_no_mask_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_3_class_a_h_no_mask_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_3_class_a_h_siam_mask_3_frg_per_bkg___trai_n_"></a>
## p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_3_class_a_h_siam_mask_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_3_class_a_h_siam_mask_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_a_h_3_class_30___trai_n_"></a>
## p1_a_h_3_class_30       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_30.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_3_class_30 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_3_class_70___trai_n_"></a>
## p1_a_h_3_class_70       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_70.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_3_class_70 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_3_class_100___trai_n_"></a>
## p1_a_h_3_class_100       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_100.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_3_class_100 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_no_mask_3_class_10___trai_n_"></a>
## p1_a_h_no_mask_3_class_10       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_3_class_10.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_no_mask_3_class_10 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_no_mask_3_class_30___trai_n_"></a>
## p1_a_h_no_mask_3_class_30       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_3_class_30.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_no_mask_3_class_30 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_no_mask_3_class_70___trai_n_"></a>
## p1_a_h_no_mask_3_class_70       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_3_class_70.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_no_mask_3_class_70 n_steps=100000 save_interval_secs=600


<a id="20k4___trai_n_"></a>
## 20k4       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k4.config train_dir=trained/retinanet_resnet50_v1_20k4 n_steps=100000 save_interval_secs=600

<a id="20k_vid4a___trai_n_"></a>
## 20K_vid4a       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_vid4a.config train_dir=trained/retinanet_resnet50_v1_20k_vid4a n_steps=100000 save_interval_secs=600

<a id="p1_4_class_a_h___trai_n_"></a>
## p1_4_class_a_h       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h.config train_dir=trained/retinanet_resnet50_v1_p1_4_class_a_h n_steps=100000 save_interval_secs=600

<a id="p1_4_class_a_h_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_4_class_a_h_no_mask_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_no_mask_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_no_mask_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_4_class_a_h_no_mask_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_4_class_a_h_siam_mask_3_frg_per_bkg___trai_n_"></a>
## p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_siam_mask_3_frg_per_bkg.config train_dir=trained/retinanet_resnet50_v1_p1_4_class_a_h_siam_mask_3_frg_per_bkg n_steps=100000 save_interval_secs=600

<a id="p1_a_h_4_class_30___trai_n_"></a>
## p1_a_h_4_class_30       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_4_class_30.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_4_class_30 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_4_class_50___trai_n_"></a>
## p1_a_h_4_class_50       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_4_class_50.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_4_class_50 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_4_class_70___trai_n_"></a>
## p1_a_h_4_class_70       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_4_class_70.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_4_class_70 n_steps=100000 save_interval_secs=600


<a id="p1_a_h_no_mask_4_class_30___trai_n_"></a>
## p1_a_h_no_mask_4_class_30       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_4_class_30.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_no_mask_4_class_30 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_no_mask_4_class_10___trai_n_"></a>
## p1_a_h_no_mask_4_class_10       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/p1_a_h_no_mask_4_class_10.config train_dir=trained/p1_a_h_no_mask_4_class_10 n_steps=100000 save_interval_secs=600


<a id="p1_a_h_no_mask_4_class_70___trai_n_"></a>
## p1_a_h_no_mask_4_class_70       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_4_class_70.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_no_mask_4_class_70 n_steps=100000 save_interval_secs=600

<a id="20k6___trai_n_"></a>
## 20K6       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6.config train_dir=trained/retinanet_resnet50_v1_20k6 n_steps=100000 save_interval_secs=600

<a id="320x320___20k6_train_"></a>
### 320x320       @ 20K6/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_320.config train_dir=trained/retinanet_resnet50_v1_20k6_320 n_steps=100000 save_interval_secs=600

<a id="20k6_5___trai_n_"></a>
## 20K6_5       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config train_dir=trained/retinanet_resnet50_v1_20k6_5 n_steps=100000 save_interval_secs=600

<a id="20k6_60___trai_n_"></a>
## 20K6_60       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20K6_60.config train_dir=trained/retinanet_resnet50_v1_20K6_60 n_steps=100000 save_interval_secs=600

<a id="10k6_entire_seq___trai_n_"></a>
## 10k6_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_10k6_entire_seq.config train_dir=trained/retinanet_resnet50_v1_10k6_entire_seq n_steps=100000 save_interval_secs=600

<a id="15k6_entire_seq___trai_n_"></a>
## 15k6_entire_seq       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_15k6_entire_seq.config train_dir=trained/retinanet_resnet50_v1_15k6_entire_seq n_steps=100000 save_interval_secs=600

<a id="1k6_vid_entire_seq___trai_n_"></a>
## 1k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k6_vid_entire_seq.config train_dir=trained/retinanet_resnet50_v1_1k6_vid_entire_seq n_steps=100000 save_interval_secs=600

<a id="10k6_vid_entire_seq___trai_n_"></a>
## 10k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_10k6_vid_entire_seq.config train_dir=trained/retinanet_resnet50_v1_10k6_vid_entire_seq n_steps=100000 save_interval_secs=600

<a id="15k6_vid_entire_seq___trai_n_"></a>
## 15k6_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_15k6_vid_entire_seq.config train_dir=trained/retinanet_resnet50_v1_15k6_vid_entire_seq n_steps=100000 save_interval_secs=600

<a id="1_per_seq_6_class_vid_67_b32___trai_n_"></a>
## 1_per_seq_6_class_vid_67_b32       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1_per_seq_6_class_vid_67_b32.config train_dir=trained/retinanet_resnet50_v1_1_per_seq_6_class_vid_67_b32 n_steps=100000 save_interval_secs=600

<a id="2_per_seq_6_class_vid_67_b32___trai_n_"></a>
## 2_per_seq_6_class_vid_67_b32       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_2_per_seq_6_class_vid_67_b32.config train_dir=trained/retinanet_resnet50_v1_2_per_seq_6_class_vid_67_b32 n_steps=100000 save_interval_secs=600

<a id="5_per_seq_6_class_vid_67_b32___trai_n_"></a>
## 5_per_seq_6_class_vid_67_b32       @ train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_5_per_seq_6_class_vid_67_b32.config train_dir=trained/retinanet_resnet50_v1_5_per_seq_6_class_vid_67_b32 n_steps=100000 save_interval_secs=600

<a id="10_per_seq_6_class_vid_67_b32___trai_n_"></a>
## 10_per_seq_6_class_vid_67_b32       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_10_per_seq_6_class_vid_67_b32.config train_dir=trained/retinanet_resnet50_v1_10_per_seq_6_class_vid_67_b32 n_steps=100000 save_interval_secs=600

<a id="acamp_switch_6_class___trai_n_"></a>
## acamp_switch_6_class       @ train

<a id="100_per_class___acamp_switch_6_class_train_"></a>
### 100_per_class       @ acamp_switch_6_class/train

<a id="individual___100_per_class_acamp_switch_6_class_train_"></a>
#### individual       @ 100_per_class/acamp_switch_6_class/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_1_100.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_1_100 n_steps=100000 save_interval_secs=600

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_2_100.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_2_100 n_steps=100000 save_interval_secs=600

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_3_100.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_3_100 n_steps=100000 save_interval_secs=600

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_4_100.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_4_100 n_steps=100000 save_interval_secs=600

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_5_100.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_5_100 n_steps=100000 save_interval_secs=600

<a id="combined___100_per_class_acamp_switch_6_class_train_"></a>
#### combined       @ 100_per_class/acamp_switch_6_class/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_combined_100.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_combined_100 n_steps=100000 save_interval_secs=600


<a id="20_per_class___acamp_switch_6_class_train_"></a>
### 20_per_class       @ acamp_switch_6_class/train

<a id="individual___20_per_class_acamp_switch_6_class_trai_n_"></a>
#### individual       @ 20_per_class/acamp_switch_6_class/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_1_20.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_1_20 n_steps=100000 save_interval_secs=600

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_2_20.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_2_20 n_steps=100000 save_interval_secs=600

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_3_20.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_3_20 n_steps=100000 save_interval_secs=600

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_4_20.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_4_20 n_steps=100000 save_interval_secs=600

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_5_20.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_5_20 n_steps=100000 save_interval_secs=600

<a id="combined___20_per_class_acamp_switch_6_class_trai_n_"></a>
#### combined       @ 20_per_class/acamp_switch_6_class/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_combined_20.config train_dir=trained/retinanet_resnet50_v1_acamp_switch_6_class_combined_20 n_steps=100000 save_interval_secs=600


<a id="4k8___trai_n_"></a>
## 4k8       @ train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_4k8.config train_dir=trained/retinanet_resnet50_v1_4k8 n_steps=100000 save_interval_secs=600

<a id="1k8_vid___trai_n_"></a>
## 1k8_vid       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k8_vid.config train_dir=trained/retinanet_resnet50_v1_1k8_vid n_steps=1000000 save_interval_secs=600

<a id="1k8_vid_even_min_1___trai_n_"></a>
## 1k8_vid_even_min_1       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k8_vid_even_min_1.config train_dir=trained/retinanet_resnet50_v1_1k8_vid_even_min_1 n_steps=1000000 save_interval_secs=600

<a id="1k8_vid_entire_seq___trai_n_"></a>
## 1k8_vid_entire_seq       @ train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k8_vid_entire_seq.config train_dir=trained/retinanet_resnet50_v1_1k8_vid_entire_seq n_steps=1000000 save_interval_secs=600

<a id="eva_l_"></a>
# eval

<a id="1k_3a___eval_"></a>
## 1K_3a       @ eval

CUDA_VISIBLE_DEVICES=0 python2 ~/models/research/object_detection/legacy/eval.py pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config eval_dir=trained/retinanet_resnet50_v1_1k3a/eval checkpoint_dir=trained/retinanet_resnet50_v1_1k3a run_once=1


<a id="tf_api_eval_"></a>
# tf_api_eval

<a id="1600_static_3___tf_api_eva_l_"></a>
## 1600_static_3       @ tf_api_eval

<a id="on_train___1600_static_3_tf_api_eva_l_"></a>
### on_train       @ 1600_static_3/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1600_static_3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp1600_static3_train.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1600_static_3.config sampling_ratio=1.0 random_sampling=0 start_seq_id=0 write_summary=1 save_det=1 load_det=1 

<a id="person_coco17_1600___on_train_1600_static_3_tf_api_eval_"></a>
#### person_coco17_1600       @ on_train/1600_static_3/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1600_static_3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=person_coco17_1600 pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1600_static_3.config sampling_ratio=1.0 random_sampling=0 start_seq_id=0 write_summary=1 save_det=1 load_det=1

<a id="acamp_no_static_2_class___1600_static_3_tf_api_eva_l_"></a>
### acamp_no_static_2_class       @ 1600_static_3/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1600_static_3 labels_path=data/wildlife_label_map_20k2.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_2_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1600_static_3.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1 save_det=1

<a id="40k3___tf_api_eva_l_"></a>
## 40k3       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  eval_every=10 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3_test.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3.config sampling_ratio=0.1 random_sampling=1 sleep_time=10 write_summary=1 save_det=1

<a id="on_train___40k3_tf_api_eval_"></a>
### on_train       @ 40k3/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3_coco_train.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3.config sampling_ratio=1.0 random_sampling=0 start_seq_id=0 write_summary=1 save_det=1 load_det=0

<a id="no_human___on_train_40k3_tf_api_eva_l_"></a>
#### no_human       @ on_train/40k3/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3 labels_path=data/wildlife_label_map_20k3.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3_coco_train_no_human.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3.config sampling_ratio=1.0 random_sampling=0 start_seq_id=0 write_summary=1 save_det=1 load_det=0

<a id="acamp_no_static_2_class___40k3_tf_api_eval_"></a>
### acamp_no_static_2_class       @ 40k3/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3 labels_path=data/wildlife_label_map_20k2.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_2_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=0 write_summary=1 save_det=1


<a id="p1_a_h_no_mask_bear_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_a_h_no_mask_bear_3_frg_per_bkg       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_no_mask_bear_3_frg_per_bkg labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=1 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_bear.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_bear_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_a_h_bear_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001


<a id="p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=1 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_bear.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_a_h_bear_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_3_class_a_h_3_frg_per_bkg       @ tf_api_eval

<a id="inverted___p1_3_class_a_h_3_frg_per_bkg_tf_api_eval_"></a>
### inverted       @ p1_3_class_a_h_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_3_class_a_h_3_frg_per_bkg labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_3_class_a_h_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_3_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_no_mask_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_3_class_a_h_no_mask_3_frg_per_bkg       @ tf_api_eval

<a id="p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_no_mask_3_frg_per_bkg_tf_api_eval_"></a>
### p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_no_mask_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_3_class_a_h_no_mask_3_frg_per_bkg labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_3_class_a_h_no_mask_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_3_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ tf_api_eval

<a id="p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tf_api_eval_"></a>
### p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_3_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_siam_mask_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ tf_api_eval

<a id="p1_3_class_a_h_3_frg_per_bkg_inv___p1_3_class_a_h_siam_mask_3_frg_per_bkg_tf_api_eval_"></a>
### p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_3_class_a_h_siam_mask_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_3_class_a_h_siam_mask_3_frg_per_bkg labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_3_class_a_h_siam_mask_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_3_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_a_h_no_mask_3_class_30___tf_api_eva_l_"></a>
## p1_a_h_no_mask_3_class_30       @ tf_api_eval

<a id="p1_a_h_3_class_30_inverted___p1_a_h_no_mask_3_class_30_tf_api_eva_l_"></a>
### p1_a_h_3_class_30_inverted       @ p1_a_h_no_mask_3_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_no_mask_3_class_30 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_3_class_30.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_a_h_3_class_30_inverted load_det=0 score_thresholds=0:1:0.001

<a id="p1_a_h_3_class_30___tf_api_eva_l_"></a>
## p1_a_h_3_class_30       @ tf_api_eval

<a id="p1_3_class_a_h_3_frg_per_bkg_inv___p1_a_h_3_class_30_tf_api_eva_l_"></a>
### p1_3_class_a_h_3_frg_per_bkg_inv       @ p1_a_h_3_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_30 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_30.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_3_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001


<a id="on_train_100___p1_a_h_3_class_30_tf_api_eva_l_"></a>
### on_train_100       @ p1_a_h_3_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_30 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_30.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1

<a id="acamp1k_static3a_test___p1_a_h_3_class_30_tf_api_eva_l_"></a>
### acamp1K_static3a_test       @ p1_a_h_3_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_30 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k seq_paths=acamp1K_static3a_test.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_30.config sampling_ratio=1.0 random_sampling=1 sleep_time=10 eval_every=0

<a id="acamp_no_static_3_class___p1_a_h_3_class_30_tf_api_eva_l_"></a>
### acamp_no_static_3_class       @ p1_a_h_3_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_30 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_30.config sampling_ratio=1.0 random_sampling=1 sleep_time=10 eval_every=0

<a id="on_inverted___p1_a_h_3_class_30_tf_api_eva_l_"></a>
### on_inverted       @ p1_a_h_3_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_30 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_30.config sampling_ratio=0.3 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 draw_plot=0

<a id="p1_a_h_3_class_70___tf_api_eva_l_"></a>
## p1_a_h_3_class_70       @ tf_api_eval

<a id="on_inverted___p1_a_h_3_class_70_tf_api_eva_l_"></a>
### on_inverted       @ p1_a_h_3_class_70/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_70 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_70.config sampling_ratio=0.7 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=train_inverted load_det=0


<a id="p1_a_h_3_class_100___tf_api_eva_l_"></a>
## p1_a_h_3_class_100       @ tf_api_eval

<a id="acamp1k_static3a_test___p1_a_h_3_class_100_tf_api_eval_"></a>
### acamp1K_static3a_test       @ p1_a_h_3_class_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_100 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3 root_dir=/data/acamp/acamp20k seq_paths=acamp1K_static3a_test.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_100.config sampling_ratio=1.0 random_sampling=1 sleep_time=10 eval_every=0

<a id="on_train___p1_a_h_3_class_100_tf_api_eval_"></a>
### on_train       @ p1_a_h_3_class_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_100 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_100.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1

<a id="acamp_no_static_3_class___p1_a_h_3_class_100_tf_api_eval_"></a>
### acamp_no_static_3_class       @ p1_a_h_3_class_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_3_class_100 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_3_class_100.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1


<a id="1k_3a___tf_api_eva_l_"></a>
## 1K_3a       @ tf_api_eval

<a id="acamp1k_static3a_test___1k_3a_tf_api_eva_l_"></a>
### acamp1K_static3a_test       @ 1K_3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp1K_static3a_test.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10 eval_every=0

<a id="acamp_no_static_3_class___1k_3a_tf_api_eva_l_"></a>
### acamp_no_static_3_class       @ 1K_3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1

<a id="p1_h_3_class___1k_3a_tf_api_eva_l_"></a>
### p1_h_3_class       @ 1K_3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1

<a id="p1_a_3_class___1k_3a_tf_api_eva_l_"></a>
### p1_a_3_class       @ 1K_3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1

<a id="p1_a_h_3_class_30_inverted___1k_3a_tf_api_eva_l_"></a>
### p1_a_h_3_class_30_inverted       @ 1K_3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config  sampling_ratio=0.3 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_a_h_3_class_30_inverted load_det=0

<a id="p1_a_h_3_class_70_inverted___1k_3a_tf_api_eva_l_"></a>
### p1_a_h_3_class_70_inverted       @ 1K_3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config  sampling_ratio=0.7 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_a_h_3_class_70_inverted load_det=0

<a id="p1_a_h_3_class_100___1k_3a_tf_api_eva_l_"></a>
### p1_a_h_3_class_100       @ 1K_3a/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1

<a id="1k_3a_sampled___tf_api_eva_l_"></a>
## 1K_3a_sampled       @ tf_api_eval

<a id="inverted___1k_3a_sampled_tf_api_eva_l_"></a>
### inverted       @ 1K_3a_sampled/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="map_plotting___inverted_1k_3a_sampled_tf_api_eval_"></a>
#### map_plotting       @ inverted/1K_3a_sampled/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=1 eval_every=0 load_dir=trained/retinanet_resnet50_v1_1k3a_sampled/19507_on_inverted show_animation=0 draw_plot=0 results_path=results/trained_retinanet_resnet50_v1_1k3a_sampled_19507_on_inverted score_thresholds=0:1:0.001


<a id="win___inverted_1k_3a_sampled_tf_api_eval_"></a>
#### win       @ inverted/1K_3a_sampled/tf_api_eval

python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=1 eval_every=0 load_dir=H:\UofA\PhD\Reports\190721_wacv_animal_detection_paper\results\retinanet\trained_retinanet_resnet50_v1_1k3a_sampled_19507_on_inverted_grs_190615_101104 show_animation=0  draw_plot=1 results_path=results/trained_retinanet_resnet50_v1_1k3a_sampled_19507_on_inverted_grs_190615_101104 score_thresholds=0.19

<a id="all_vid___1k_3a_sampled_tf_api_eva_l_"></a>
### all_vid       @ 1K_3a_sampled/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_video_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled.config sleep_time=10 write_summary=1 save_det=1 out_postfix=all_vid load_det=0 score_thresholds=0:1:0.001

<a id="p1_3_class_a_h_3_frg_per_bkg_inv___1k_3a_sampled_tf_api_eva_l_"></a>
### p1_3_class_a_h_3_frg_per_bkg_inv       @ 1K_3a_sampled/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a_sampled labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled.config sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_3_class_a_h_3_frg_per_bkg_inv load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv score_thresholds=0:1:0.001

<a id="1k_3a_sampled_score_threshold_0___tf_api_eva_l_"></a>
## 1K_3a_sampled_score_threshold_0       @ tf_api_eval

<a id="inverted___1k_3a_sampled_score_threshold_0_tf_api_eva_l_"></a>
### inverted       @ 1K_3a_sampled_score_threshold_0/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k3a_sampled_score_threshold_0 labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a_sampled_score_threshold_0.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="500_static3a___tf_api_eva_l_"></a>
## 500_static3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001

<a id="class_agnostic___500_static3a_tf_api_eval_"></a>
### class_agnostic       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 class_agnostic=1

<a id="nms_thresh_0___500_static3a_tf_api_eval_"></a>
### nms_thresh_0       @ 500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="bgr_to_rgb_0___nms_thresh_0_500_static3a_tf_api_eva_l_"></a>
#### bgr_to_rgb_0       @ nms_thresh_0/500_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_500_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_500_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted_bgr load_det=0 score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 bgr_to_rgb=0

<a id="200_static3a___tf_api_eva_l_"></a>
## 200_static3a       @ tf_api_eval

<a id="inverted___200_static3a_tf_api_eval_"></a>
### inverted       @ 200_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_200_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_200_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="inclusive___inverted_200_static3a_tf_api_eva_l_"></a>
#### inclusive       @ inverted/200_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_200_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_200_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=1 eval_every=0 score_thresholds=0:1.01:0.01 load_dir=H:\UofA\PhD\Reports\190721_wacv_animal_detection_paper\results\retinanet\trained_retinanet_resnet50_v1_200_static3a_29925_on_inverted_grs_190615_085631 show_animation=0 draw_plot=0 results_path=results/trained_retinanet_resnet50_v1_200_static3a_29925_on_inverted_grs_190615_085631


<a id="map_plotting___inverted_200_static3a_tf_api_eva_l_"></a>
#### map_plotting       @ inverted/200_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_200_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_200_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=1 eval_every=0 load_dir=trained/retinanet_resnet50_v1_200_static3a/29925_on_inverted show_animation=0 draw_plot=0 results_path=results/trained_retinanet_resnet50_v1_200_static3a_29925_on_inverted score_thresholds=0:1:0.001

<a id="win___inverted_200_static3a_tf_api_eva_l_"></a>
#### win       @ inverted/200_static3a/tf_api_eval

python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_200_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_200_static3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=1 eval_every=0 score_thresholds=0 load_dir=H:\UofA\PhD\Reports\190721_wacv_animal_detection_paper\results\retinanet\trained_retinanet_resnet50_v1_200_static3a_29925_on_inverted_grs_190615_085631 show_animation=0  draw_plot=1 results_path=results/trained_retinanet_resnet50_v1_200_static3a_29925_on_inverted_grs_190615_085631

<a id="20k_vid3a___tf_api_eva_l_"></a>
## 20K_vid3a       @ tf_api_eval

<a id="acamp200_static3a_inverted___20k_vid3a_tf_api_eva_l_"></a>
### acamp200_static3a_inverted       @ 20K_vid3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_vid3a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp200_static3a_inverted load_det=1 load_dir=trained/retinanet_resnet50_v1_20k_vid3a/38575_on_acamp200_static3a_inverted score_thresholds=0:1:0.001

<a id="all_static___20k_vid3a_tf_api_eva_l_"></a>
### all_static       @ 20K_vid3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_vid3a.config sleep_time=10 write_summary=1 save_det=1 out_postfix=all_static score_thresholds=0:1:0.001

<a id="class_agnostic___all_static_20k_vid3a_tf_api_eval_"></a>
#### class_agnostic       @ all_static/20K_vid3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_vid3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_vid3a.config sleep_time=10 write_summary=1 save_det=1 out_postfix=all_static score_thresholds=0:1:0.001 class_agnostic=1 load_det=1

<a id="no_static3a___tf_api_eva_l_"></a>
## no_static3a       @ tf_api_eval

<a id="acamp_all_static_3_class___no_static3a_tf_api_eva_l_"></a>
### acamp_all_static_3_class       @ no_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1 save_det=1

<a id="p1_h_3_class___no_static3a_tf_api_eva_l_"></a>
### p1_h_3_class       @ no_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1 save_det=1

<a id="p1_a_3_class___no_static3a_tf_api_eva_l_"></a>
### p1_a_3_class       @ no_static3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1 save_det=1

<a id="40k3a_rt___tf_api_eva_l_"></a>
## 40k3a_rt       @ tf_api_eval

<a id="on_train___40k3a_rt_tf_api_eval_"></a>
### on_train       @ 40k3a_rt/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3a_rt labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3a_train.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config write_summary=1 save_det=1 out_postfix=train

<a id="40k3a___tf_api_eva_l_"></a>
## 40k3a       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  eval_every=10 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3a_test.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config sampling_ratio=0.1 random_sampling=1 sleep_time=10 write_summary=1 save_det=1

<a id="on_train___40k3a_tf_api_eva_l_"></a>
### on_train       @ 40k3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=3  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp40k3a_train.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config write_summary=1 save_det=1 out_postfix=train

<a id="acamp_no_static_3_class___40k3a_tf_api_eva_l_"></a>
### acamp_no_static_3_class       @ 40k3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1 save_det=1

<a id="acamp_all_static_3_class___40k3a_tf_api_eva_l_"></a>
### acamp_all_static_3_class       @ 40k3a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1 save_det=1

<a id="p1_h_3_class___40k3a_tf_api_eva_l_"></a>
### p1_h_3_class       @ 40k3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_h_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10  eval_every=0 write_summary=1 save_det=1

<a id="p1_a_3_class___40k3a_tf_api_eva_l_"></a>
### p1_a_3_class       @ 40k3a/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_40k3a labels_path=data/wildlife_label_map_20k3a.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=3  root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config sampling_ratio=1.0 random_sampling=1 sleep_time=10 eval_every=0 write_summary=1 save_det=1

<a id="no_static_bear___tf_api_eva_l_"></a>
## no_static__bear       @ tf_api_eval

<a id="on_train___no_static_bear_tf_api_eval_"></a>
### on_train       @ no_static__bear/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_bear labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_bear.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_bear.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_no_static_bear load_det=0

<a id="acamp_static_bear___no_static_bear_tf_api_eval_"></a>
### acamp_static_bear       @ no_static__bear/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_bear labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_bear.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_bear.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_bear load_det=0

<a id="acamp_static_3_class___no_static_bear_tf_api_eval_"></a>
### acamp_static_3_class       @ no_static__bear/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_bear labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_bear.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_3_class load_det=0

<a id="no_static_deer___tf_api_eva_l_"></a>
## no_static__deer       @ tf_api_eval

<a id="on_train___no_static_deer_tf_api_eval_"></a>
### on_train       @ no_static__deer/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_deer labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_deer.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_deer.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_no_static_deer load_det=0

<a id="acamp_static_deer___no_static_deer_tf_api_eval_"></a>
### acamp_static_deer       @ no_static__deer/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_deer labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_deer.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_deer.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_deer load_det=0

<a id="acamp_static_3_class___no_static_deer_tf_api_eval_"></a>
### acamp_static_3_class       @ no_static__deer/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_deer labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_deer.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_3_class load_det=0

<a id="no_static_coyote___tf_api_eva_l_"></a>
## no_static_coyote       @ tf_api_eval

<a id="on_train___no_static_coyote_tf_api_eval_"></a>
### on_train       @ no_static_coyote/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_coyote labels_path=data/wildlife_label_map_coyote.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_no_static_coyote.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_coyote.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_no_static_coyote load_det=0

<a id="acamp_static_coyote___no_static_coyote_tf_api_eval_"></a>
### acamp_static_coyote       @ no_static_coyote/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_coyote labels_path=data/wildlife_label_map_coyote.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_coyote.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_coyote.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_coyote load_det=0

<a id="acamp_static_3_class___no_static_coyote_tf_api_eval_"></a>
### acamp_static_3_class       @ no_static_coyote/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_no_static_coyote labels_path=data/wildlife_label_map_coyote.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_no_static_coyote.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_3_class load_det=0

<a id="20k_bear___tf_api_eva_l_"></a>
## 20k_bear       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_bear labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_bear.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bear.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k_bear_inverted load_det=0

<a id="20k4_inverted___20k_bear_tf_api_eval_"></a>
### 20k4_inverted       @ 20k_bear/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_bear labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bear.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k4_inverted load_det=0

<a id="acamp_all_static_3_class___20k_bear_tf_api_eval_"></a>
### acamp_all_static_3_class       @ 20k_bear/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_bear labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bear.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_3_class load_det=0


<a id="20k6_inverted___20k_bear_tf_api_eval_"></a>
### 20k6_inverted       @ 20k_bear/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_bear labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bear.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_inverted load_det=0

<a id="only_video___20k6_inverted_20k_bear_tf_api_eval_"></a>
#### only_video       @ 20k6_inverted/20k_bear/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_bear labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6 eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bear.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_5_inv_only_vid load_det=0

<a id="20k_deer___tf_api_eva_l_"></a>
## 20k_deer       @ tf_api_eval

<a id="on_inverted___20k_deer_tf_api_eval_"></a>
### on_inverted       @ 20k_deer/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_deer labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_deer.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_deer.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k_deer_inverted load_det=0

<a id="20k4_inverted___20k_deer_tf_api_eval_"></a>
### 20k4_inverted       @ 20k_deer/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_deer labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_deer.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k4_inverted load_det=0

<a id="acamp_all_static_3_class___20k_deer_tf_api_eval_"></a>
### acamp_all_static_3_class       @ 20k_deer/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_deer labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_deer.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_3_class load_det=0

<a id="20k6_inverted___20k_deer_tf_api_eval_"></a>
### 20k6_inverted       @ 20k_deer/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_deer labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_deer.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_inverted load_det=0

<a id="only_video___20k6_inverted_20k_deer_tf_api_eval_"></a>
#### only_video       @ 20k6_inverted/20k_deer/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_deer labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6 eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_deer.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_5_inv_only_vid load_det=0

<a id="20k_coyote___tf_api_eva_l_"></a>
## 20k_coyote       @ tf_api_eval

<a id="on_inverted___20k_coyote_tf_api_eval_"></a>
### on_inverted       @ 20k_coyote/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_coyote labels_path=data/wildlife_label_map_coyote.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_coyote.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_coyote.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k_coyote_inverted load_det=0

<a id="20k4_inverted___20k_coyote_tf_api_eval_"></a>
### 20k4_inverted       @ 20k_coyote/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_coyote labels_path=data/wildlife_label_map_coyote.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_coyote.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k4_inverted load_det=0

<a id="acamp_all_static_3_class___20k_coyote_tf_api_eval_"></a>
### acamp_all_static_3_class       @ 20k_coyote/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_coyote labels_path=data/wildlife_label_map_coyote.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_static_3_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_coyote.config sampling_ratio=1.0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_static_3_class load_det=0

<a id="20k6_inverted___20k_coyote_tf_api_eval_"></a>
### 20k6_inverted       @ 20k_coyote/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_coyote labels_path=data/wildlife_label_map_coyote.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_coyote.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_inverted load_det=0

<a id="only_video___20k6_inverted_20k_coyote_tf_api_eval_"></a>
#### only_video       @ 20k6_inverted/20k_coyote/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_coyote labels_path=data/wildlife_label_map_coyote.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6 eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_coyote.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_5_inv_only_vid load_det=0

<a id="20k_moose___tf_api_eva_l_"></a>
## 20k_moose       @ tf_api_eval

<a id="on_inverted___20k_moose_tf_api_eva_l_"></a>
### on_inverted       @ 20k_moose/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_moose labels_path=data/wildlife_label_map_moose.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_moose.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_moose.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k_moose_inverted load_det=0

<a id="20k4_inverted___20k_moose_tf_api_eva_l_"></a>
### 20k4_inverted       @ 20k_moose/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_moose labels_path=data/wildlife_label_map_moose.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_moose.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k4_inverted load_det=0

<a id="20k6_inverted___20k_moose_tf_api_eva_l_"></a>
### 20k6_inverted       @ 20k_moose/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_moose labels_path=data/wildlife_label_map_moose.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_moose.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_inverted load_det=0

<a id="only_video___20k6_inverted_20k_moose_tf_api_eva_l_"></a>
#### only_video       @ 20k6_inverted/20k_moose/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_moose labels_path=data/wildlife_label_map_moose.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6 eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_moose.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_5_inv_only_vid load_det=0

<a id="20k_elk___tf_api_eva_l_"></a>
## 20k_elk       @ tf_api_eval

<a id="on_inverted___20k_elk_tf_api_eva_l_"></a>
### on_inverted       @ 20k_elk/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_elk labels_path=data/wildlife_label_map_elk.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_elk.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_elk.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k_elk_inverted load_det=0

<a id="20k6_inverted___20k_elk_tf_api_eva_l_"></a>
### 20k6_inverted       @ 20k_elk/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_elk labels_path=data/wildlife_label_map_elk.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_elk.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_inverted load_det=0


<a id="only_video___20k6_inverted_20k_elk_tf_api_eva_l_"></a>
#### only_video       @ 20k6_inverted/20k_elk/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_elk labels_path=data/wildlife_label_map_elk.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6 eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_elk.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_5_inv_only_vid load_det=0

<a id="20k_bison___tf_api_eva_l_"></a>
## 20k_bison       @ tf_api_eval

<a id="on_inverted___20k_bison_tf_api_eva_l_"></a>
### on_inverted       @ 20k_bison/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_bison labels_path=data/wildlife_label_map_bison.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_bison.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bison.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k_bison_inverted load_det=0

<a id="20k6_inverted___20k_bison_tf_api_eva_l_"></a>
### 20k6_inverted       @ 20k_bison/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_bison labels_path=data/wildlife_label_map_bison.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bison.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_inverted load_det=0

<a id="only_video___20k6_inverted_20k_bison_tf_api_eva_l_"></a>
#### only_video       @ 20k6_inverted/20k_bison/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_bison labels_path=data/wildlife_label_map_bison.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6 eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_bison.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=20k6_5_inv_only_vid load_det=0

<a id="20k4___tf_api_eva_l_"></a>
## 20k4       @ tf_api_eval

<a id="on_inverted___20k4_tf_api_eval_"></a>
### on_inverted       @ 20k4/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k4 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k4.config sampling_ratio=0.8 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=20k4_inverted load_det=0

<a id="on_train___20k4_tf_api_eval_"></a>
### on_train       @ 20k4/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k4 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k4.config sampling_ratio=0.8 sleep_time=10 write_summary=1 save_det=1 out_postfix=train load_det=0

<a id="p1_a_h_4_class_30_inverted___20k4_tf_api_eval_"></a>
### p1_a_h_4_class_30_inverted       @ 20k4/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k4 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k4.config sampling_ratio=0.3 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_a_h_4_class_30_inverted load_det=0

<a id="p1_a_h_4_class_50_inverted___20k4_tf_api_eval_"></a>
### p1_a_h_4_class_50_inverted       @ 20k4/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k4 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k4.config sampling_ratio=0.5 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_a_h_4_class_50_inverted load_det=0

<a id="p1_4_class_a_h_3_frg_per_bkg_inv___20k4_tf_api_eval_"></a>
### p1_4_class_a_h_3_frg_per_bkg_inv       @ 20k4/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k4 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k4.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_4_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="prototype_1_vid_4_class___20k4_tf_api_eval_"></a>
### prototype_1_vid_4_class       @ 20k4/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k4 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1_vid seq_paths=prototype_1_vid_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k4.config sleep_time=10 write_summary=1 save_det=1 out_postfix=prototype_1_vid_4_class load_det=0 score_thresholds=0:1:0.001

<a id="20k_vid4a___tf_api_eva_l_"></a>
## 20K_vid4a       @ tf_api_eval

<a id="p1_4_class_a_h_3_frg_per_bkg_inv___20k_vid4a_tf_api_eva_l_"></a>
### p1_4_class_a_h_3_frg_per_bkg_inv       @ 20K_vid4a/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k_vid4a labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k_vid4a.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_4_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_4_class_a_h_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_4_class_a_h_3_frg_per_bkg       @ tf_api_eval

<a id="inverted___p1_4_class_a_h_3_frg_per_bkg_tf_api_eval_"></a>
### inverted       @ p1_4_class_a_h_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_4_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="prototype_1_vid_4_class___p1_4_class_a_h_3_frg_per_bkg_tf_api_eval_"></a>
### prototype_1_vid_4_class       @ p1_4_class_a_h_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1_vid seq_paths=prototype_1_vid_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg.config sleep_time=10 write_summary=1 save_det=1 out_postfix=prototype_1_vid_4_class load_det=0 score_thresholds=0:1:0.001

<a id="vis___prototype_1_vid_4_class_p1_4_class_a_h_3_frg_per_bkg_tf_api_eval_"></a>
#### vis       @ prototype_1_vid_4_class/p1_4_class_a_h_3_frg_per_bkg/tf_api_eval

python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1_vid seq_paths=prototype_1_vid_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg.config sleep_time=10 write_summary=1 save_det=1 out_postfix=prototype_1_vid_4_class load_det=1 show_img=1 eval_every=-1

<a id="prototype_1_vid_4_class_no_moving_bkg___p1_4_class_a_h_3_frg_per_bkg_tf_api_eval_"></a>
### prototype_1_vid_4_class_no_moving_bkg       @ p1_4_class_a_h_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1_vid seq_paths=prototype_1_vid_4_class_no_moving_bkg.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_3_frg_per_bkg.config sleep_time=10 write_summary=1 save_det=1 out_postfix=prototype_1_vid_4_class load_det=1 score_thresholds=0:1:0.001


<a id="p1_4_class_a_h_no_mask_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_4_class_a_h_no_mask_3_frg_per_bkg       @ tf_api_eval

<a id="p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_no_mask_3_frg_per_bkg_tf_api_eval_"></a>
### p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_no_mask_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_4_class_a_h_no_mask_3_frg_per_bkg labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_no_mask_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_4_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ tf_api_eval

<a id="p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tf_api_eval_"></a>
### p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_4_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001


<a id="p1_a_h_4_class_70_inverted___p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_tf_api_eval_"></a>
### p1_a_h_4_class_70_inverted       @ p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k4 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k4.config sampling_ratio=0.7 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_a_h_4_class_70_inverted load_det=0

<a id="p1_4_class_a_h_siam_mask_3_frg_per_bkg___tf_api_eva_l_"></a>
## p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ tf_api_eval

<a id="p1_4_class_a_h_3_frg_per_bkg_inv___p1_4_class_a_h_siam_mask_3_frg_per_bkg_tf_api_eval_"></a>
### p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_4_class_a_h_siam_mask_3_frg_per_bkg/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_4_class_a_h_siam_mask_3_frg_per_bkg labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_4_class_a_h_siam_mask_3_frg_per_bkg.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_4_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_a_h_no_mask_4_class_30___tf_api_eva_l_"></a>
## p1_a_h_no_mask_4_class_30       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_4_class_30.config train_dir=trained/retinanet_resnet50_v1_p1_a_h_no_mask_4_class_30 n_steps=100000 save_interval_secs=600

<a id="p1_a_h_4_class_30_inverted___p1_a_h_no_mask_4_class_30_tf_api_eva_l_"></a>
### p1_a_h_4_class_30_inverted       @ p1_a_h_no_mask_4_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_no_mask_4_class_30 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_no_mask_4_class_30.config  sleep_time=10 write_summary=1 load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted save_det=1 out_postfix=p1_a_h_4_class_30_inverted load_det=0 score_thresholds=0:1:0.001


<a id="p1_a_h_4_class_30___tf_api_eva_l_"></a>
## p1_a_h_4_class_30       @ tf_api_eval

<a id="on_inverted___p1_a_h_4_class_30_tf_api_eva_l_"></a>
### on_inverted       @ p1_a_h_4_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_4_class_30 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_4_class_30.config sampling_ratio=0.3 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="p1_4_class_a_h_3_frg_per_bkg_inv___p1_a_h_4_class_30_tf_api_eva_l_"></a>
### p1_4_class_a_h_3_frg_per_bkg_inv       @ p1_a_h_4_class_30/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_4_class_30 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_4_class_30.config load_samples=1 load_samples_root=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=p1_4_class_a_h_3_frg_per_bkg_inv load_det=0 score_thresholds=0:1:0.001

<a id="p1_a_h_4_class_50___tf_api_eva_l_"></a>
## p1_a_h_4_class_50       @ tf_api_eval

<a id="on_inverted___p1_a_h_4_class_50_tf_api_eva_l_"></a>
### on_inverted       @ p1_a_h_4_class_50/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_4_class_50 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_4_class_50.config sampling_ratio=0.5 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=train_inverted load_det=0

<a id="p1_a_h_4_class_70___tf_api_eva_l_"></a>
## p1_a_h_4_class_70       @ tf_api_eval

<a id="on_inverted___p1_a_h_4_class_70_tf_api_eva_l_"></a>
### on_inverted       @ p1_a_h_4_class_70/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_p1_a_h_4_class_70 labels_path=data/wildlife_label_map_20k4.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=4  eval_every=0 root_dir=/data/acamp/acamp20k/prototype_1 seq_paths=p1_a_h_4_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_p1_a_h_4_class_70.config sampling_ratio=0.7 even_sampling=1 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=train_inverted load_det=0

<a id="20k6___tf_api_eva_l_"></a>
## 20k6       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6.config sampling_ratio=-0.2 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_all_6_class_20_from_end load_det=0

<a id="on_train___20k6_tf_api_eval_"></a>
### on_train       @ 20k6/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6.config sampling_ratio=0.8 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=train

<a id="custom_folder___20k6_tf_api_eval_"></a>
### custom_folder       @ 20k6/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=1 n_classes=6 eval_every=-1 seq_paths=<custom_folder_path> pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6.config sleep_time=10 write_summary=1 save_det=1 out_postfix=custom_folder load_det=0


<a id="20k6_320___tf_api_eva_l_"></a>
## 20k6_320       @ tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_320 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_320.config sampling_ratio=-0.2 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_all_6_class_20_from_end load_det=1 save_dir=trained/retinanet_resnet50_v1_20k6_320/50782_on_acamp_all_6_class_20_from_end

<a id="acamp_all_coyote___20k6_320_tf_api_eval_"></a>
### acamp_all_coyote       @ 20k6_320/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_320 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_coyote.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_320.config sampling_ratio=-0.2 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_all_6_class_20_from_end load_det=1 save_dir=trained/retinanet_resnet50_v1_20k6_320/50782_on_acamp_all_6_class_20_from_end

<a id="acamp_all_deer___20k6_320_tf_api_eval_"></a>
### acamp_all_deer       @ 20k6_320/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_320 labels_path=data/wildlife_label_map_deer.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_deer.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_320.config sampling_ratio=-0.2 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=acamp_all_6_class_20_from_end load_det=1 save_dir=trained/retinanet_resnet50_v1_20k6_320/50782_on_acamp_all_6_class_20_from_end


<a id="20k6_5___tf_api_eva_l_"></a>
## 20k6_5       @ tf_api_eval

<a id="inverted___20k6_5_tf_api_eval_"></a>
### inverted       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 load_dir=trained/retinanet_resnet50_v1_20k6_5/8320_on_inverted score_thresholds=0:1:0.001 

<a id="combine_sequences_640x360___inverted_20k6_5_tf_api_eva_l_"></a>
#### combine_sequences_640x360       @ inverted/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=112 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 load_dir=trained/retinanet_resnet50_v1_20k6_5/8320_on_inverted combine_sequences=1 input_size=640x360

<a id="combine_sequences_416x416___inverted_20k6_5_tf_api_eva_l_"></a>
#### combine_sequences_416x416       @ inverted/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=-1 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=0 load_dir=trained/retinanet_resnet50_v1_20k6_5/8320_on_inverted combine_sequences=1 input_size=416x416

<a id="score_thresholds_0___inverted_20k6_5_tf_api_eva_l_"></a>
#### score_thresholds_0       @ inverted/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0

<a id="acamp_all_coyote___20k6_5_tf_api_eval_"></a>
### acamp_all_coyote       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_coyote.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config sampling_ratio=0.05 inverted_sampling=1 random_sampling=0 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0

<a id="only_video___20k6_5_tf_api_eval_"></a>
### only_video       @ 20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=inv_only_vid load_det=1 load_dir=trained/retinanet_resnet50_v1_20k6_5/8320_on_inverted score_thresholds=0:1:0.001 

<a id="nms_thresh_0___only_video_20k6_5_tf_api_eva_l_"></a>
#### nms_thresh_0       @ only_video/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inv_only_vid load_det=1 load_dir=trained/retinanet_resnet50_v1_20k6_5/8320_on_nms_thresh_0_inv_only_vid score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0

<a id="bgr___nms_thresh_0_only_video_20k6_5_tf_api_eval_"></a>
##### bgr       @ nms_thresh_0/only_video/20k6_5/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20k6_5 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=24 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20k6_5.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inv_only_vid_bgr load_det=0 load_dir=trained/retinanet_resnet50_v1_20k6_5/8320_on_nms_thresh_0_inv_only_vid_bgr score_thresholds=0:1:0.001 inference_dir=inference_nms_thresh_0 bgr_to_rgb=0

<a id="20k6_60___tf_api_eva_l_"></a>
## 20k6_60       @ tf_api_eval

<a id="inverted___20k6_60_tf_api_eva_l_"></a>
### inverted       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20K6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20K6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="vis___20k6_60_tf_api_eva_l_"></a>
### vis       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20K6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20K6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 show_img=1 eval_every=-1

<a id="map_plotting___vis_20k6_60_tf_api_eva_l_"></a>
#### map_plotting       @ vis/20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20K6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20K6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=0 out_postfix=inverted load_det=1 eval_every=0 draw_plot=1 score_thresholds=0 load_dir=H:\UofA\PhD\Reports\190721_wacv_animal_detection_paper\results\retinanet\trained_retinanet_resnet50_v1_20K6_60_40839_on_inverted_z370_190608_151153 show_animation=1

<a id="combine_sequences_640x360___20k6_60_tf_api_eva_l_"></a>
### combine_sequences_640x360       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20K6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=1 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20K6_60.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp20k6_60_train inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 combine_sequences=1 input_size=640x360 show_animation=1 save_animation=1 show_img=1 eval_every=-1

<a id="acamp_1_per_class_6_class___20k6_60_tf_api_eva_l_"></a>
### acamp_1_per_class_6_class       @ 20k6_60/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_20K6_60 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=60 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_1_per_class_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_20K6_60.config sleep_time=10 write_summary=1 save_det=1

<a id="10k6_entire_seq___tf_api_eva_l_"></a>
## 10k6_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_10k6_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_10k6_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="15k6_entire_seq___tf_api_eva_l_"></a>
## 15k6_entire_seq       @ tf_api_eval

<a id="inverted___15k6_entire_seq_tf_api_eva_l_"></a>
### inverted       @ 15k6_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_15k6_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_15k6_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp15k6_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001

<a id="1k6_vid_entire_seq___tf_api_eva_l_"></a>
## 1k6_vid_entire_seq       @ tf_api_eval

<a id="inverted___1k6_vid_entire_seq_tf_api_eval_"></a>
### inverted       @ 1k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="10k6_vid_entire_seq___tf_api_eva_l_"></a>
## 10k6_vid_entire_seq       @ tf_api_eval

<a id="inverted___10k6_vid_entire_seq_tf_api_eva_l_"></a>
### inverted       @ 10k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="nms_thresh_0___10k6_vid_entire_seq_tf_api_eva_l_"></a>
### nms_thresh_0       @ 10k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_10k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_10k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=nms_thresh_0_inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1 inference_dir=inference_nms_thresh_0 n_threads=8

<a id="15k6_vid_entire_seq___tf_api_eva_l_"></a>
## 15k6_vid_entire_seq       @ tf_api_eval

<a id="inverted___15k6_vid_entire_seq_tf_api_eva_l_"></a>
### inverted       @ 15k6_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_15k6_vid_entire_seq labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_15k6_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001 allow_seq_skipping=1

<a id="1_per_seq_6_class_vid_67_b32___tf_api_eva_l_"></a>
## 1_per_seq_6_class_vid_67_b32       @ tf_api_eval

<a id="inverted___1_per_seq_6_class_vid_67_b32_tf_api_eval_"></a>
### inverted       @ 1_per_seq_6_class_vid_67_b32/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1_per_seq_6_class_vid_67_b32 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1_per_seq_6_class_vid_67_b32.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/1_per_seq_6_class_vid_67 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 

<a id="acamp_all_6_class_video_67___1_per_seq_6_class_vid_67_b32_tf_api_eval_"></a>
### acamp_all_6_class_video_67       @ 1_per_seq_6_class_vid_67_b32/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1_per_seq_6_class_vid_67_b32 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1_per_seq_6_class_vid_67_b32.config  sleep_time=10 write_summary=1 save_det=1 load_det=0 

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___1_per_seq_6_class_vid_67_b32_tf_api_eval_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 1_per_seq_6_class_vid_67_b32/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1_per_seq_6_class_vid_67_b32 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=56 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1_per_seq_6_class_vid_67_b32.config sleep_time=10 write_summary=1 save_det=1 load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted out_postfix=1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="2_per_seq_6_class_vid_67_b32___tf_api_eva_l_"></a>
## 2_per_seq_6_class_vid_67_b32       @ tf_api_eval

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___2_per_seq_6_class_vid_67_b32_tf_api_eval_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 2_per_seq_6_class_vid_67_b32/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_2_per_seq_6_class_vid_67_b32 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_2_per_seq_6_class_vid_67_b32.config sleep_time=10 write_summary=1 save_det=1 load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted out_postfix=1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="5_per_seq_6_class_vid_67_b32___tf_api_eva_l_"></a>
## 5_per_seq_6_class_vid_67_b32       @ tf_api_eval

<a id="inverted___5_per_seq_6_class_vid_67_b32_tf_api_eval_"></a>
### inverted       @ 5_per_seq_6_class_vid_67_b32/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_5_per_seq_6_class_vid_67_b32 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_5_per_seq_6_class_vid_67_b32.config  load_samples=1 load_samples_root=/data/acamp/acamp20k/5_per_seq_6_class_vid_67_b32 inverted_sampling=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 

<a id="acamp_all_6_class_video_67___5_per_seq_6_class_vid_67_b32_tf_api_eval_"></a>
### acamp_all_6_class_video_67       @ 5_per_seq_6_class_vid_67_b32/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_5_per_seq_6_class_vid_67_b32 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_5_per_seq_6_class_vid_67_b32.config  sleep_time=10 write_summary=1 save_det=1 load_det=0 

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___5_per_seq_6_class_vid_67_b32_tf_api_eval_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 5_per_seq_6_class_vid_67_b32/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_5_per_seq_6_class_vid_67_b32 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_5_per_seq_6_class_vid_67_b32.config sleep_time=10 write_summary=1 save_det=1 load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted out_postfix=1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="10_per_seq_6_class_vid_67_b32___tf_api_eva_l_"></a>
## 10_per_seq_6_class_vid_67_b32       @ tf_api_eval

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___10_per_seq_6_class_vid_67_b32_tf_api_eva_l_"></a>
### 1_2_5_10_per_seq_6_class_vid_67_inverted       @ 10_per_seq_6_class_vid_67_b32/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_10_per_seq_6_class_vid_67_b32 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=16 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_6_class_video_67.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_10_per_seq_6_class_vid_67_b32.config sleep_time=10 write_summary=1 save_det=1 load_det=0 load_samples=1 load_samples_root=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted out_postfix=1_2_5_10_per_seq_6_class_vid_67_inverted score_thresholds=0:1:0.001

<a id="acamp_switch_6_class___tf_api_eva_l_"></a>
## acamp_switch_6_class       @ tf_api_eval

<a id="100_per_class___acamp_switch_6_class_tf_api_eval_"></a>
### 100_per_class       @ acamp_switch_6_class/tf_api_eval

<a id="individual___100_per_class_acamp_switch_6_class_tf_api_eval_"></a>
#### individual       @ 100_per_class/acamp_switch_6_class/tf_api_eval

<a id="1___100_per_class_acamp_switch_6_class_tf_api_eval_"></a>
#### 1       @ 100_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_1_100 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_1.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_1_100.config load_samples=acamp_switch_6_class_1_100_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="2___100_per_class_acamp_switch_6_class_tf_api_eval_"></a>
#### 2       @ 100_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_2_100 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_2.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_2_100.config load_samples=acamp_switch_6_class_2_100_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="3___100_per_class_acamp_switch_6_class_tf_api_eval_"></a>
#### 3       @ 100_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_3_100 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_3.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_3_100.config load_samples=acamp_switch_6_class_3_100_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="4___100_per_class_acamp_switch_6_class_tf_api_eval_"></a>
#### 4       @ 100_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_4_100 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_4.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_4_100.config load_samples=acamp_switch_6_class_4_100_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="5___100_per_class_acamp_switch_6_class_tf_api_eval_"></a>
#### 5       @ 100_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_5_100 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_5.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_5_100.config load_samples=acamp_switch_6_class_5_100_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="1_to_5___100_per_class_acamp_switch_6_class_tf_api_eval_"></a>
#### 1_to_5       @ 100_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_combined_100 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_combined.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_combined_100.config load_samples=acamp_switch_6_class_1_100_test.txt,acamp_switch_6_class_2_100_test.txt,acamp_switch_6_class_3_100_test.txt,acamp_switch_6_class_4_100_test.txt,acamp_switch_6_class_5_100_test.txt load_samples_root=sampled_lists load_det=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted save_dir=trained/acamp_switch_6_class_1_to_5_100 out_prefix=acamp_switch_6_class_1_to_5_100


<a id="combined___100_per_class_acamp_switch_6_class_tf_api_eval_"></a>
#### combined       @ 100_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_combined_100 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_combined.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_combined_100.config load_samples=acamp_switch_6_class_1_100_test.txt,acamp_switch_6_class_2_100_test.txt,acamp_switch_6_class_3_100_test.txt,acamp_switch_6_class_4_100_test.txt,acamp_switch_6_class_5_100_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted


<a id="20_per_class___acamp_switch_6_class_tf_api_eval_"></a>
### 20_per_class       @ acamp_switch_6_class/tf_api_eval

<a id="individual___20_per_class_acamp_switch_6_class_tf_api_eva_l_"></a>
#### individual       @ 20_per_class/acamp_switch_6_class/tf_api_eval

<a id="1___20_per_class_acamp_switch_6_class_tf_api_eva_l_"></a>
#### 1       @ 20_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_1_20 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_1.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_1_20.config load_samples=acamp_switch_6_class_1_20_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="2___20_per_class_acamp_switch_6_class_tf_api_eva_l_"></a>
#### 2       @ 20_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_2_20 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_2.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_2_20.config load_samples=acamp_switch_6_class_2_20_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="3___20_per_class_acamp_switch_6_class_tf_api_eva_l_"></a>
#### 3       @ 20_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_3_20 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_3.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_3_20.config load_samples=acamp_switch_6_class_3_20_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="4___20_per_class_acamp_switch_6_class_tf_api_eva_l_"></a>
#### 4       @ 20_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_4_20 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_4.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_4_20.config load_samples=acamp_switch_6_class_4_20_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="5___20_per_class_acamp_switch_6_class_tf_api_eva_l_"></a>
#### 5       @ 20_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_5_20 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_5.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_5_20.config load_samples=acamp_switch_6_class_5_20_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="1_to_5___20_per_class_acamp_switch_6_class_tf_api_eva_l_"></a>
#### 1_to_5       @ 20_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_combined_20 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_combined.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_combined_20.config load_samples=acamp_switch_6_class_1_20_test.txt,acamp_switch_6_class_2_20_test.txt,acamp_switch_6_class_3_20_test.txt,acamp_switch_6_class_4_20_test.txt,acamp_switch_6_class_5_20_test.txt load_samples_root=sampled_lists load_det=1 sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted save_dir=trained/acamp_switch_6_class_1_to_5_20 out_prefix=acamp_switch_6_class_1_to_5_20


<a id="combined___20_per_class_acamp_switch_6_class_tf_api_eva_l_"></a>
#### combined       @ 20_per_class/acamp_switch_6_class/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_acamp_switch_6_class_combined_20 labels_path=data/wildlife_label_map_20k6.pbtxt n_frames=0 batch_size=32 show_img=0 n_classes=6  eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_switch_6_class_combined.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_acamp_switch_6_class_combined_20.config load_samples=acamp_switch_6_class_1_20_test.txt,acamp_switch_6_class_2_20_test.txt,acamp_switch_6_class_3_20_test.txt,acamp_switch_6_class_4_20_test.txt,acamp_switch_6_class_5_20_test.txt load_samples_root=sampled_lists sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted

<a id="4k8___tf_api_eva_l_"></a>
## 4k8       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_4k8 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_4k8.config samples_per_class=4000 even_sampling=-1 sleep_time=10 write_summary=1 save_det=1 out_postfix=4k8_inverted load_det=0

<a id="1k8_vid___tf_api_eva_l_"></a>
## 1k8_vid       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k8_vid labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k8_vid.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_inverted sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=0 score_thresholds=0:1:0.001


<a id="1k8_vid_entire_seq___tf_api_eva_l_"></a>
## 1k8_vid_entire_seq       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k8_vid_entire_seq labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1 n_threads=0

<a id="class_agnostic___1k8_vid_entire_seq_tf_api_eval_"></a>
### class_agnostic       @ 1k8_vid_entire_seq/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k8_vid_entire_seq labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=64 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k8_vid_entire_seq.config load_samples=1 load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1 n_threads=0 class_agnostic=1

<a id="1k8_vid_even_min_1___tf_api_eva_l_"></a>
## 1k8_vid_even_min_1       @ tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/retinanet_resnet50_v1_1k8_vid_even_min_1 labels_path=data/wildlife_label_map_4k8.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=8 eval_every=0 root_dir=/data/acamp/acamp20k seq_paths=acamp_all_8_class_video.txt pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k8_vid_even_min_1.config load_samples=1 load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1_inv sleep_time=10 write_summary=1 save_det=1 out_postfix=inverted load_det=1 score_thresholds=0:1:0.001 allow_seq_skipping=1



<a id="train_and_eva_l_"></a>
# train_and_eval

<a id="1k3a_static___train_and_eval_"></a>
## 1k3a_static       @ train_and_eval

CUDA_VISIBLE_DEVICES=2 python3 train_and_eval.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config checkpoint_dir=trained/retinanet_resnet50_v1_1k3a eval.eval_dir=trained/retinanet_resnet50_v1_1k3a/eval eval.run_once=1 steps_between_eval=1000 total_steps=500000

<a id="40k3a___train_and_eval_"></a>
## 40k3a       @ train_and_eval

CUDA_VISIBLE_DEVICES=1 python3 train_and_eval.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_40k3a.config checkpoint_dir=trained/retinanet_resnet50_v1_40k3a eval.eval_dir=trained/retinanet_resnet50_v1_40k3a/eval eval.run_once=1 steps_between_eval=1000 total_steps=500000

<a id="model_main___40k3a_train_and_eval_"></a>
### model_main       @ 40k3a/train_and_eval

CUDA_VISIBLE_DEVICES=0,1 python3 ~/models/research/object_detection/model_main.py logtostderr pipeline_config_path=configs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/retinanet_resnet50_v1_1k3a.config model_dir=trained/retinanet_resnet50_v1_1k3a num_train_steps=10


<a id="tes_t_"></a>
# test

<a id="pretrained___test_"></a>
## pretrained       @ test

<a id="camera___pretrained_tes_t_"></a>
### camera       @ pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=pre_trained_models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0

<a id="only_person___camera_pretrained_test_"></a>
#### only_person       @ camera/pretrained/test

CUDA_VISIBLE_DEVICES=0 python3 tf_api_test.py ckpt_path=pre_trained_models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=camera_0 n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person,