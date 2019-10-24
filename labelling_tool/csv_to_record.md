<!-- MarkdownTOC -->

- [5k       @ csv_to_record](#5k___csv_to_recor_d_)
    - [test       @ 5k](#test___5k_)
- [10k       @ csv_to_record](#10k___csv_to_recor_d_)
    - [test       @ 10k](#test___10_k_)
- [10k_ar       @ csv_to_record](#10k_ar___csv_to_recor_d_)
- [10kh       @ csv_to_record](#10kh___csv_to_recor_d_)
- [10ka       @ csv_to_record](#10ka___csv_to_recor_d_)
- [20k       @ csv_to_record](#20k___csv_to_recor_d_)
    - [human_static       @ 20k](#human_static___20_k_)
    - [coco       @ 20k](#coco___20_k_)
    - [test       @ 20k](#test___20_k_)
- [20k3       @ csv_to_record](#20k3___csv_to_recor_d_)
    - [test       @ 20k3](#test___20k3_)
    - [human_static       @ 20k3](#human_static___20k3_)
    - [coco       @ 20k3](#coco___20k3_)
- [20k4       @ csv_to_record](#20k4___csv_to_recor_d_)
- [10k_test       @ csv_to_record](#10k_test___csv_to_recor_d_)
- [20k7       @ csv_to_record](#20k7___csv_to_recor_d_)
- [20k5       @ csv_to_record](#20k5___csv_to_recor_d_)
- [25k7       @ csv_to_record](#25k7___csv_to_recor_d_)
- [25k5       @ csv_to_record](#25k5___csv_to_recor_d_)
- [25k3       @ csv_to_record](#25k3___csv_to_recor_d_)
- [25k4       @ csv_to_record](#25k4___csv_to_recor_d_)
- [25k6       @ csv_to_record](#25k6___csv_to_recor_d_)
- [40k3_coco       @ csv_to_record](#40k3_coco___csv_to_recor_d_)
- [40k3a       @ csv_to_record](#40k3a___csv_to_recor_d_)
- [1600_static3       @ csv_to_record](#1600_static3___csv_to_recor_d_)
- [1K_static3a       @ csv_to_record](#1k_static3a___csv_to_recor_d_)
    - [acamp1K_static3a_train_yolov3_pt       @ 1K_static3a](#acamp1k_static3a_train_yolov3_pt___1k_static3_a_)
    - [acamp1K_static3a_test_50_yolov3_pt       @ 1K_static3a](#acamp1k_static3a_test_50_yolov3_pt___1k_static3_a_)
- [all_non_static3a       @ csv_to_record](#all_non_static3a___csv_to_recor_d_)
- [sampling_based       @ csv_to_record](#sampling_based___csv_to_recor_d_)
    - [p1_a_h_bear_3_frg_per_bkg       @ sampling_based](#p1_a_h_bear_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_a_h_bear_3_frg_per_bkg/sampling_based](#inverted___p1_a_h_bear_3_frg_per_bkg_sampling_based_)
    - [p1_a_h_no_mask_bear_3_frg_per_bkg       @ sampling_based](#p1_a_h_no_mask_bear_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_a_h_no_mask_bear_3_frg_per_bkg/sampling_based](#inverted___p1_a_h_no_mask_bear_3_frg_per_bkg_sampling_based_)
    - [p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ sampling_based](#p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/sampling_based](#inverted___p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg_sampling_base_d_)
    - [20k_bear       @ sampling_based](#20k_bear___sampling_based_)
    - [20k_deer       @ sampling_based](#20k_deer___sampling_based_)
    - [20k_coyote       @ sampling_based](#20k_coyote___sampling_based_)
    - [20k_moose       @ sampling_based](#20k_moose___sampling_based_)
    - [20k_bison       @ sampling_based](#20k_bison___sampling_based_)
    - [20k_elk       @ sampling_based](#20k_elk___sampling_based_)
    - [no_static_bear       @ sampling_based](#no_static_bear___sampling_based_)
    - [no_static_deer       @ sampling_based](#no_static_deer___sampling_based_)
    - [no_static_coyote       @ sampling_based](#no_static_coyote___sampling_based_)
    - [p1_3_class_a_h_3_frg_per_bkg       @ sampling_based](#p1_3_class_a_h_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_3_class_a_h_3_frg_per_bkg/sampling_based](#inverted___p1_3_class_a_h_3_frg_per_bkg_sampling_base_d_)
    - [p1_3_class_a_h_no_mask_3_frg_per_bkg       @ sampling_based](#p1_3_class_a_h_no_mask_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_3_class_a_h_no_mask_3_frg_per_bkg/sampling_based](#inverted___p1_3_class_a_h_no_mask_3_frg_per_bkg_sampling_base_d_)
    - [p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ sampling_based](#p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/sampling_based](#inverted___p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_sampling_base_d_)
    - [p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ sampling_based](#p1_3_class_a_h_siam_mask_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_3_class_a_h_siam_mask_3_frg_per_bkg/sampling_based](#inverted___p1_3_class_a_h_siam_mask_3_frg_per_bkg_sampling_base_d_)
    - [p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg       @ sampling_based](#p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg/sampling_based](#inverted___p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg_sampling_base_d_)
    - [p1_a_h_3_class_30       @ sampling_based](#p1_a_h_3_class_30___sampling_based_)
        - [inverted       @ p1_a_h_3_class_30/sampling_based](#inverted___p1_a_h_3_class_30_sampling_based_)
    - [p1_a_h_3_class_70       @ sampling_based](#p1_a_h_3_class_70___sampling_based_)
    - [p1_a_h_3_class_100       @ sampling_based](#p1_a_h_3_class_100___sampling_based_)
    - [p1_a_h_no_mask_3_class_10       @ sampling_based](#p1_a_h_no_mask_3_class_10___sampling_based_)
    - [p1_a_h_no_mask_3_class_30       @ sampling_based](#p1_a_h_no_mask_3_class_30___sampling_based_)
        - [inverted       @ p1_a_h_no_mask_3_class_30/sampling_based](#inverted___p1_a_h_no_mask_3_class_30_sampling_based_)
    - [p1_a_h_no_mask_3_class_70       @ sampling_based](#p1_a_h_no_mask_3_class_70___sampling_based_)
    - [p1_a_h_no_mask_3_class_100       @ sampling_based](#p1_a_h_no_mask_3_class_100___sampling_based_)
    - [acamp10_static3a_sampled       @ sampling_based](#acamp10_static3a_sampled___sampling_based_)
    - [all_vid_3_class       @ sampling_based](#all_vid_3_class___sampling_based_)
    - [all_static_3_class       @ sampling_based](#all_static_3_class___sampling_based_)
    - [500_static3a       @ sampling_based](#500_static3a___sampling_based_)
        - [inverted       @ 500_static3a/sampling_based](#inverted___500_static3a_sampling_base_d_)
    - [1K_static3a_sampled       @ sampling_based](#1k_static3a_sampled___sampling_based_)
        - [inverted       @ 1K_static3a_sampled/sampling_based](#inverted___1k_static3a_sampled_sampling_based_)
    - [acamp1K_static3a_test_50_yolov3_pt       @ sampling_based](#acamp1k_static3a_test_50_yolov3_pt___sampling_based_)
    - [200_static3a       @ sampling_based](#200_static3a___sampling_based_)
    - [200_from_end_static3a       @ sampling_based](#200_from_end_static3a___sampling_based_)
        - [inverted       @ 200_from_end_static3a/sampling_based](#inverted___200_from_end_static3a_sampling_based_)
    - [20K_vid3a       @ sampling_based](#20k_vid3a___sampling_based_)
        - [inverted_200_per_class       @ 20K_vid3a/sampling_based](#inverted_200_per_class___20k_vid3a_sampling_based_)
    - [20k4       @ sampling_based](#20k4___sampling_based_)
    - [20K_vid4a       @ sampling_based](#20k_vid4a___sampling_based_)
        - [inverted_200_per_class       @ 20K_vid4a/sampling_based](#inverted_200_per_class___20k_vid4a_sampling_based_)
    - [prototype_1_vid_4_class_no_moving_bkg       @ sampling_based](#prototype_1_vid_4_class_no_moving_bkg___sampling_based_)
    - [p1_4_class_a_h       @ sampling_based](#p1_4_class_a_h___sampling_based_)
    - [p1_4_class_a_h_3_frg_per_bkg       @ sampling_based](#p1_4_class_a_h_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_4_class_a_h_3_frg_per_bkg/sampling_based](#inverted___p1_4_class_a_h_3_frg_per_bkg_sampling_base_d_)
    - [p1_4_class_a_h_no_mask_3_frg_per_bkg       @ sampling_based](#p1_4_class_a_h_no_mask_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_4_class_a_h_no_mask_3_frg_per_bkg/sampling_based](#inverted___p1_4_class_a_h_no_mask_3_frg_per_bkg_sampling_base_d_)
    - [p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ sampling_based](#p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/sampling_based](#inverted___p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_sampling_base_d_)
    - [p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ sampling_based](#p1_4_class_a_h_siam_mask_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_4_class_a_h_siam_mask_3_frg_per_bkg/sampling_based](#inverted___p1_4_class_a_h_siam_mask_3_frg_per_bkg_sampling_base_d_)
    - [p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg       @ sampling_based](#p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg___sampling_based_)
        - [inverted       @ p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg/sampling_based](#inverted___p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg_sampling_base_d_)
    - [p1_a_h_4_class_30       @ sampling_based](#p1_a_h_4_class_30___sampling_based_)
        - [inverted       @ p1_a_h_4_class_30/sampling_based](#inverted___p1_a_h_4_class_30_sampling_based_)
    - [p1_a_h_4_class_50       @ sampling_based](#p1_a_h_4_class_50___sampling_based_)
    - [p1_a_h_4_class_70       @ sampling_based](#p1_a_h_4_class_70___sampling_based_)
    - [p1_a_h_no_mask_4_class_10       @ sampling_based](#p1_a_h_no_mask_4_class_10___sampling_based_)
    - [p1_a_h_no_mask_4_class_30       @ sampling_based](#p1_a_h_no_mask_4_class_30___sampling_based_)
        - [inverted       @ p1_a_h_no_mask_4_class_30/sampling_based](#inverted___p1_a_h_no_mask_4_class_30_sampling_based_)
    - [p1_a_h_no_mask_4_class_50       @ sampling_based](#p1_a_h_no_mask_4_class_50___sampling_based_)
    - [p1_a_h_no_mask_4_class_70       @ sampling_based](#p1_a_h_no_mask_4_class_70___sampling_based_)
    - [20k6       @ sampling_based](#20k6___sampling_based_)
        - [test       @ 20k6/sampling_based](#test___20k6_sampling_base_d_)
    - [20k6_5       @ sampling_based](#20k6_5___sampling_based_)
        - [yolov3_pt_annotations       @ 20k6_5/sampling_based](#yolov3_pt_annotations___20k6_5_sampling_base_d_)
        - [inverted_annotations_only_vid       @ 20k6_5/sampling_based](#inverted_annotations_only_vid___20k6_5_sampling_base_d_)
        - [inverted_annotations       @ 20k6_5/sampling_based](#inverted_annotations___20k6_5_sampling_base_d_)
            - [yolov3_pt       @ inverted_annotations/20k6_5/sampling_based](#yolov3_pt___inverted_annotations_20k6_5_sampling_based_)
                - [video       @ yolov3_pt/inverted_annotations/20k6_5/sampling_based](#video___yolov3_pt_inverted_annotations_20k6_5_sampling_based_)
                - [bear_10_3       @ yolov3_pt/inverted_annotations/20k6_5/sampling_based](#bear_10_3___yolov3_pt_inverted_annotations_20k6_5_sampling_based_)
                - [bear_1_1       @ yolov3_pt/inverted_annotations/20k6_5/sampling_based](#bear_1_1___yolov3_pt_inverted_annotations_20k6_5_sampling_based_)
                - [coyote_jesse_1_1       @ yolov3_pt/inverted_annotations/20k6_5/sampling_based](#coyote_jesse_1_1___yolov3_pt_inverted_annotations_20k6_5_sampling_based_)
        - [5_from_end_annotations       @ 20k6_5/sampling_based](#5_from_end_annotations___20k6_5_sampling_base_d_)
        - [1_from_end_annotations       @ 20k6_5/sampling_based](#1_from_end_annotations___20k6_5_sampling_base_d_)
            - [yolov3_pt       @ 1_from_end_annotations/20k6_5/sampling_based](#yolov3_pt___1_from_end_annotations_20k6_5_sampling_based_)
            - [dummy_0.005       @ 1_from_end_annotations/20k6_5/sampling_based](#dummy_0_005___1_from_end_annotations_20k6_5_sampling_based_)
    - [20k6_60       @ sampling_based](#20k6_60___sampling_based_)
        - [yolov3_pt_annotations       @ 20k6_60/sampling_based](#yolov3_pt_annotations___20k6_60_sampling_based_)
        - [inverted_annotations       @ 20k6_60/sampling_based](#inverted_annotations___20k6_60_sampling_based_)
            - [yolov3_pt       @ inverted_annotations/20k6_60/sampling_based](#yolov3_pt___inverted_annotations_20k6_60_sampling_base_d_)
    - [1_per_seq_6_class_vid_67       @ sampling_based](#1_per_seq_6_class_vid_67___sampling_based_)
    - [2_per_seq_6_class_vid_67       @ sampling_based](#2_per_seq_6_class_vid_67___sampling_based_)
    - [5_per_seq_6_class_vid_67       @ sampling_based](#5_per_seq_6_class_vid_67___sampling_based_)
    - [10_per_seq_6_class_vid_67       @ sampling_based](#10_per_seq_6_class_vid_67___sampling_based_)
    - [1_5_10_per_seq_6_class_vid_67_inverted       @ sampling_based](#1_5_10_per_seq_6_class_vid_67_inverted___sampling_based_)
        - [bear_1_1       @ 1_5_10_per_seq_6_class_vid_67_inverted/sampling_based](#bear_1_1___1_5_10_per_seq_6_class_vid_67_inverted_sampling_base_d_)
    - [1_2_5_10_per_seq_6_class_vid_67_inverted       @ sampling_based](#1_2_5_10_per_seq_6_class_vid_67_inverted___sampling_based_)
    - [acamp_switch_6_class       @ sampling_based](#acamp_switch_6_class___sampling_based_)
        - [100_per_class       @ acamp_switch_6_class/sampling_based](#100_per_class___acamp_switch_6_class_sampling_base_d_)
            - [individual       @ 100_per_class/acamp_switch_6_class/sampling_based](#individual___100_per_class_acamp_switch_6_class_sampling_base_d_)
            - [combined       @ 100_per_class/acamp_switch_6_class/sampling_based](#combined___100_per_class_acamp_switch_6_class_sampling_base_d_)
            - [sampling_for_test       @ 100_per_class/acamp_switch_6_class/sampling_based](#sampling_for_test___100_per_class_acamp_switch_6_class_sampling_base_d_)
        - [20_per_class       @ acamp_switch_6_class/sampling_based](#20_per_class___acamp_switch_6_class_sampling_base_d_)
            - [individual       @ 20_per_class/acamp_switch_6_class/sampling_based](#individual___20_per_class_acamp_switch_6_class_sampling_based_)
            - [combined       @ 20_per_class/acamp_switch_6_class/sampling_based](#combined___20_per_class_acamp_switch_6_class_sampling_based_)
            - [sampling_for_test       @ 20_per_class/acamp_switch_6_class/sampling_based](#sampling_for_test___20_per_class_acamp_switch_6_class_sampling_based_)
    - [10k6       @ sampling_based](#10k6___sampling_based_)
        - [evenly_sampled       @ 10k6/sampling_based](#evenly_sampled___10k6_sampling_base_d_)
            - [test       @ evenly_sampled/10k6/sampling_based](#test___evenly_sampled_10k6_sampling_based_)
    - [10k6_entire_seq       @ sampling_based](#10k6_entire_seq___sampling_based_)
        - [inverted       @ 10k6_entire_seq/sampling_based](#inverted___10k6_entire_seq_sampling_based_)
            - [even_5       @ inverted/10k6_entire_seq/sampling_based](#even_5___inverted_10k6_entire_seq_sampling_base_d_)
    - [15k6_entire_seq       @ sampling_based](#15k6_entire_seq___sampling_based_)
        - [inverted       @ 15k6_entire_seq/sampling_based](#inverted___15k6_entire_seq_sampling_based_)
            - [even_5       @ inverted/15k6_entire_seq/sampling_based](#even_5___inverted_15k6_entire_seq_sampling_base_d_)
    - [1k6_vid_entire_seq       @ sampling_based](#1k6_vid_entire_seq___sampling_based_)
        - [load       @ 1k6_vid_entire_seq/sampling_based](#load___1k6_vid_entire_seq_sampling_base_d_)
        - [inverted       @ 1k6_vid_entire_seq/sampling_based](#inverted___1k6_vid_entire_seq_sampling_base_d_)
            - [even_5       @ inverted/1k6_vid_entire_seq/sampling_based](#even_5___inverted_1k6_vid_entire_seq_sampling_based_)
            - [1_per_seq       @ inverted/1k6_vid_entire_seq/sampling_based](#1_per_seq___inverted_1k6_vid_entire_seq_sampling_based_)
    - [10k6_vid_entire_seq       @ sampling_based](#10k6_vid_entire_seq___sampling_based_)
        - [load       @ 10k6_vid_entire_seq/sampling_based](#load___10k6_vid_entire_seq_sampling_based_)
        - [inverted       @ 10k6_vid_entire_seq/sampling_based](#inverted___10k6_vid_entire_seq_sampling_based_)
            - [even_5       @ inverted/10k6_vid_entire_seq/sampling_based](#even_5___inverted_10k6_vid_entire_seq_sampling_base_d_)
            - [1_per_seq       @ inverted/10k6_vid_entire_seq/sampling_based](#1_per_seq___inverted_10k6_vid_entire_seq_sampling_base_d_)
    - [15k6_vid_entire_seq       @ sampling_based](#15k6_vid_entire_seq___sampling_based_)
        - [inverted       @ 15k6_vid_entire_seq/sampling_based](#inverted___15k6_vid_entire_seq_sampling_based_)
            - [even_5       @ inverted/15k6_vid_entire_seq/sampling_based](#even_5___inverted_15k6_vid_entire_seq_sampling_base_d_)
            - [1_per_seq       @ inverted/15k6_vid_entire_seq/sampling_based](#1_per_seq___inverted_15k6_vid_entire_seq_sampling_base_d_)
    - [4k8       @ sampling_based](#4k8___sampling_based_)
    - [1k8_vid_even_min_1       @ sampling_based](#1k8_vid_even_min_1___sampling_based_)
        - [inverted       @ 1k8_vid_even_min_1/sampling_based](#inverted___1k8_vid_even_min_1_sampling_base_d_)
            - [bison       @ inverted/1k8_vid_even_min_1/sampling_based](#bison___inverted_1k8_vid_even_min_1_sampling_based_)
            - [100_per_class       @ inverted/1k8_vid_even_min_1/sampling_based](#100_per_class___inverted_1k8_vid_even_min_1_sampling_based_)
            - [1_per_seq       @ inverted/1k8_vid_even_min_1/sampling_based](#1_per_seq___inverted_1k8_vid_even_min_1_sampling_based_)
    - [1k8_vid_entire_seq       @ sampling_based](#1k8_vid_entire_seq___sampling_based_)
        - [load       @ 1k8_vid_entire_seq/sampling_based](#load___1k8_vid_entire_seq_sampling_base_d_)
        - [inverted       @ 1k8_vid_entire_seq/sampling_based](#inverted___1k8_vid_entire_seq_sampling_base_d_)
            - [even_5       @ inverted/1k8_vid_entire_seq/sampling_based](#even_5___inverted_1k8_vid_entire_seq_sampling_based_)
            - [1_per_seq       @ inverted/1k8_vid_entire_seq/sampling_based](#1_per_seq___inverted_1k8_vid_entire_seq_sampling_based_)

<!-- /MarkdownTOC -->

<a id="5k___csv_to_recor_d_"></a>
# 5k       @ csv_to_record

python2 csv_to_record.py --seq_paths=/data/acamp/acamp5k/train/images/acamp5k_train_images_1806162042.txt --output_path=/data/acamp/acamp5k/acamp5k_train.record

<a id="test___5k_"></a>
## test       @ 5k

python2 csv_to_record.py --seq_paths=/data/acamp/acamp5k/test/images/acamp_test_images_1806162058.txt --output_path=/data/acamp/acamp5k/acamp5k_test.record

python2 csv_to_record.py --seq_paths=/data/acamp/acamp5k/test/images/acamp_test_images_1806162058.txt --end_frame_id=1000 --output_path=/data/acamp/acamp5k/acamp5k_test_1000.record

python2 csv_to_record.py --seq_paths=/data/acamp/acamp5k/test/images/acamp_test_images_1806162058.txt --end_frame_id=500 --output_path=/data/acamp/acamp5k/acamp5k_test_500.record

<a id="10k___csv_to_recor_d_"></a>
# 10k       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp10k_train.txt --output_path=/data/acamp/acamp10k/acamp10k_train.record 

<a id="test___10_k_"></a>
## test       @ 10k

python2 csv_to_record.py --seq_paths=acamp10k_test.txt --output_path=/data/acamp/acamp10k/acamp10k_test.record 

<a id="10k_ar___csv_to_recor_d_"></a>
# 10k_ar       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp10k_ar_list.txt --output_path=/data/acamp/acamp10k/acamp10k_ar_train.record 

<a id="10kh___csv_to_recor_d_"></a>
# 10kh       @ csv_to_record

python2 csv_to_record.py --min_size=1 --seq_paths=acamp10kh_list.txt --output_path=/data/acamp/acamp10k/acamp10kh_train.record 

python2 csv_to_record.py --min_size=1 --seq_paths=acamp10kh_biwi_list.txt --output_path=/data/acamp/acamp10k/acamp10kh_biwi.record 

<a id="10ka___csv_to_recor_d_"></a>
# 10ka       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp10ka_list.txt --output_path=/data/acamp/acamp10k/acamp10ka_train.record

<a id="20k___csv_to_recor_d_"></a>
# 20k       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp20k_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k_train.record 

<a id="human_static___20_k_"></a>
## human_static       @ 20k

python2 csv_to_record.py --seq_paths=acamp20k_hs_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k_hs_train.record

<a id="coco___20_k_"></a>
## coco       @ 20k

python2 csv_to_record.py --seq_paths=acamp20k_coco_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k_coco_train.record

<a id="test___20_k_"></a>
## test       @ 20k

python2 csv_to_record.py --seq_paths=acamp20k_test.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k_test.record  

<a id="20k3___csv_to_recor_d_"></a>
# 20k3       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp20k3_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k3_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3.txt

<a id="test___20k3_"></a>
## test       @ 20k3

python2 csv_to_record.py --seq_paths=acamp20k3_test.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k3_test.record  

<a id="human_static___20k3_"></a>
## human_static       @ 20k3

python2 csv_to_record.py --seq_paths=acamp20k3_hs_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k3_hs_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3.txt

<a id="coco___20k3_"></a>
## coco       @ 20k3

python2 csv_to_record.py --seq_paths=acamp20k3_coco_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k3_coco_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3.txt

<a id="20k4___csv_to_recor_d_"></a>
# 20k4       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp20k4_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k4_train.record

<a id="10k_test___csv_to_recor_d_"></a>
# 10k_test       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp10k_test.txt --output_path=/data/acamp/acamp10k/acamp10k_test.record 

<a id="20k7___csv_to_recor_d_"></a>
# 20k7       @ csv_to_record

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k7.txt --seq_paths=acamp20k7_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k7_train.record


<a id="20k5___csv_to_recor_d_"></a>
# 20k5       @ csv_to_record

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k5.txt --seq_paths=acamp20k5_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k5_train.record

<a id="25k7___csv_to_recor_d_"></a>
# 25k7       @ csv_to_record

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k7.txt --seq_paths=acamp25k7_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp25k7_train.record

<a id="25k5___csv_to_recor_d_"></a>
# 25k5       @ csv_to_record

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k5.txt --seq_paths=acamp25k5_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp25k5_train.record

<a id="25k3___csv_to_recor_d_"></a>
# 25k3       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp25k3_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp25k3_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3.txt

<a id="25k4___csv_to_recor_d_"></a>
# 25k4       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp25k4_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp25k4_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k4.txt

<a id="25k6___csv_to_recor_d_"></a>
# 25k6       @ csv_to_record

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k7.txt --seq_paths=acamp25k7_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp25k7_train.record

<a id="40k3_coco___csv_to_recor_d_"></a>
# 40k3_coco       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp40k3_coco_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp40k3_coco_train.record --class_names_path=../labelling_tool/data/predefined_classes_20k3.txt

<a id="40k3a___csv_to_recor_d_"></a>
# 40k3a       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp40k3a_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp40k3a_train.record --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt

<a id="1600_static3___csv_to_recor_d_"></a>
# 1600_static3       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp1600_static3_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp1600_static3_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3.txt

<a id="1k_static3a___csv_to_recor_d_"></a>
# 1K_static3a       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp1K_static3a_train.txt --root_dir=/data/acamp/acamp20k/#static --output_path=/data/acamp/acamp20k/acamp1K_static3a_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt

<a id="acamp1k_static3a_train_yolov3_pt___1k_static3_a_"></a>
## acamp1K_static3a_train_yolov3_pt       @ 1K_static3a


python2 csv_to_record.py --seq_paths=acamp1K_static3a_train.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp1K_static3a_train.record --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --write_annotations_list=2 --write_tfrecord=0 --annotations_list_path=/data/acamp/acamp20k/acamp1K_static3a_train_yolov3_pt.txt

<a id="acamp1k_static3a_test_50_yolov3_pt___1k_static3_a_"></a>
## acamp1K_static3a_test_50_yolov3_pt       @ 1K_static3a

python2 csv_to_record.py --seq_paths=acamp1K_static3a_test.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp1K_static3a_test.record --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --sampling_ratio=0.5 --write_annotations_list=2 --write_tfrecord=0 --annotations_list_path=/data/acamp/acamp20k/acamp1K_static3a_test_50_yolov3_pt.txt

<a id="all_non_static3a___csv_to_recor_d_"></a>
# all_non_static3a       @ csv_to_record

python2 csv_to_record.py --seq_paths=acamp_no_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp_no_static_3_class.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt

<a id="sampling_based___csv_to_recor_d_"></a>
# sampling_based       @ csv_to_record

<a id="p1_a_h_bear_3_frg_per_bkg___sampling_based_"></a>
## p1_a_h_bear_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bear.txt --seq_paths=p1_a_h_bear.txt --root_dir=/data/acamp/acamp20k/prototype_1  --load_samples=p1_a_h_bear_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/prototype_1 --output_path=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_a_h_bear_3_frg_per_bkg_sampling_based_"></a>
### inverted       @ p1_a_h_bear_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bear.txt --seq_paths=p1_a_h_bear.txt --root_dir=/data/acamp/acamp20k/prototype_1  --load_samples=p1_a_h_bear_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/prototype_1 --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_a_h_bear_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0


<a id="p1_a_h_no_mask_bear_3_frg_per_bkg___sampling_based_"></a>
## p1_a_h_no_mask_bear_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bear.txt --seq_paths=p1_a_h_no_mask_bear.txt --root_dir=/data/acamp/acamp20k/p1_no_mask  --load_samples=p1_bear_a_h_no_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_no_mask --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_bear_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_a_h_no_mask_bear_3_frg_per_bkg_sampling_based_"></a>
### inverted       @ p1_a_h_no_mask_bear_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bear.txt --seq_paths=p1_a_h_no_mask_bear.txt --root_dir=/data/acamp/acamp20k/p1_no_mask  --load_samples=p1_bear_a_h_no_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_no_mask --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_bear_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg___sampling_based_"></a>
## p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bear.txt --seq_paths=p1_a_h_mask_rcnn_resnet101_bear.txt --root_dir=/data/acamp/acamp20k/p1_mask_rcnn_resnet101  --load_samples=p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_mask_rcnn_resnet101 --output_path=/data/acamp/acamp20k/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bear.txt --seq_paths=p1_a_h_mask_rcnn_resnet101_bear.txt --root_dir=/data/acamp/acamp20k/p1_mask_rcnn_resnet101  --load_samples=p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_mask_rcnn_resnet101 --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_bear_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg_inv.record --write_annotations_list=2  --write_tfrecord=0

<a id="20k_bear___sampling_based_"></a>
## 20k_bear       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bear.txt --seq_paths=acamp_all_bear.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20k_bear_train.record --sampling_ratio=0.8

<a id="20k_deer___sampling_based_"></a>
## 20k_deer       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_deer.txt --seq_paths=acamp_all_deer.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20k_deer_train.record --sampling_ratio=0.8

<a id="20k_coyote___sampling_based_"></a>
## 20k_coyote       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_coyote.txt --seq_paths=acamp_all_coyote.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20k_coyote_train.record --sampling_ratio=0.8


<a id="20k_moose___sampling_based_"></a>
## 20k_moose       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_moose.txt --seq_paths=acamp_all_moose.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20k_moose_train.record --sampling_ratio=0.8

<a id="20k_bison___sampling_based_"></a>
## 20k_bison       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bison.txt --seq_paths=acamp_all_bison.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20k_bison_train.record --sampling_ratio=0.8

<a id="20k_elk___sampling_based_"></a>
## 20k_elk       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_elk.txt --seq_paths=acamp_all_elk.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20k_elk_train.record --sampling_ratio=0.8

<a id="no_static_bear___sampling_based_"></a>
## no_static_bear       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_bear.txt --seq_paths=acamp_no_static_bear.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp_no_static_bear_train.record --sampling_ratio=1.0

<a id="no_static_deer___sampling_based_"></a>
## no_static_deer       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_deer.txt --seq_paths=acamp_no_static_deer.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp_no_static_deer_train.record --sampling_ratio=1.0

<a id="no_static_coyote___sampling_based_"></a>
## no_static_coyote       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_coyote.txt --seq_paths=acamp_no_static_coyote.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp_no_static_coyote_train.record --sampling_ratio=1.0


<a id="p1_3_class_a_h_3_frg_per_bkg___sampling_based_"></a>
## p1_3_class_a_h_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --load_samples=p1_3_class_a_h_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/prototype_1 --output_path=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_3_class_a_h_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_3_class_a_h_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --load_samples=p1_3_class_a_h_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/prototype_1 --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_3_class_a_h_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_3_class_a_h_no_mask_3_frg_per_bkg___sampling_based_"></a>
## p1_3_class_a_h_no_mask_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_no_mask_3_class.txt --root_dir=/data/acamp/acamp20k/p1_no_mask  --load_samples=p1_3_class_a_h_no_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_no_mask --output_path=/data/acamp/acamp20k/p1_3_class_a_h_no_mask_3_frg_per_bkg.record --write_annotations_list=2 --check_images=1

<a id="inverted___p1_3_class_a_h_no_mask_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_3_class_a_h_no_mask_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_no_mask_3_class.txt --root_dir=/data/acamp/acamp20k/p1_no_mask  --load_samples=p1_3_class_a_h_no_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_no_mask --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_3_class_a_h_no_mask_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___sampling_based_"></a>
## p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_mask_rcnn_resnet101_3_class.txt --root_dir=/data/acamp/acamp20k/p1_mask_rcnn_resnet101  --load_samples=p1_3_class_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_mask_rcnn_resnet101 --output_path=/data/acamp/acamp20k/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.record --write_annotations_list=2 --check_images=0

<a id="inverted___p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_mask_rcnn_resnet101_3_class.txt --root_dir=/data/acamp/acamp20k/p1_mask_rcnn_resnet101  --load_samples=p1_3_class_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_mask_rcnn_resnet101 --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_3_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_3_class_a_h_siam_mask_3_frg_per_bkg___sampling_based_"></a>
## p1_3_class_a_h_siam_mask_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_siam_mask_3_class.txt --root_dir=/data/acamp/acamp20k/p1_siam_mask  --load_samples=p1_3_class_a_h_siam_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_siam_mask --output_path=/data/acamp/acamp20k/p1_3_class_a_h_siam_mask_3_frg_per_bkg.record --write_annotations_list=2 --check_images=0

<a id="inverted___p1_3_class_a_h_siam_mask_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_3_class_a_h_siam_mask_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_siam_mask_3_class.txt --root_dir=/data/acamp/acamp20k/p1_siam_mask  --load_samples=p1_3_class_a_h_siam_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_siam_mask --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_3_class_a_h_siam_mask_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg___sampling_based_"></a>
## p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_siam_mask_davis_3_class.txt --root_dir=/data/acamp/acamp20k/p1_siam_mask_davis  --load_samples=p1_3_class_a_h_siam_mask_davis_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_siam_mask_davis --output_path=/data/acamp/acamp20k/p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg.record --write_annotations_list=2 --check_images=0

<a id="inverted___p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_siam_mask_davis_3_class.txt --root_dir=/data/acamp/acamp20k/p1_siam_mask_davis  --load_samples=p1_3_class_a_h_siam_mask_davis_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_siam_mask_davis --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_3_class_a_h_siam_mask_davis_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_a_h_3_class_30___sampling_based_"></a>
## p1_a_h_3_class_30       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --sampling_ratio=0.3 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_3_class_30.record --write_annotations_list=2 --write_tfrecord=0

<a id="inverted___p1_a_h_3_class_30_sampling_based_"></a>
### inverted       @ p1_a_h_3_class_30/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --sampling_ratio=0.3 --even_sampling=1 --inverted_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_3_class_30_inverted.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_a_h_3_class_70___sampling_based_"></a>
## p1_a_h_3_class_70       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --sampling_ratio=0.7 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_3_class_70.record

<a id="p1_a_h_3_class_100___sampling_based_"></a>
## p1_a_h_3_class_100       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --sampling_ratio=1.0 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_3_class_100.record

<a id="p1_a_h_no_mask_3_class_10___sampling_based_"></a>
## p1_a_h_no_mask_3_class_10       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_no_mask_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.1 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_3_class_10.record  --write_annotations_list=2

<a id="p1_a_h_no_mask_3_class_30___sampling_based_"></a>
## p1_a_h_no_mask_3_class_30       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_no_mask_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.3 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_3_class_30.record --write_annotations_list=2

<a id="inverted___p1_a_h_no_mask_3_class_30_sampling_based_"></a>
### inverted       @ p1_a_h_no_mask_3_class_30/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_no_mask_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.3 --even_sampling=1  --inverted_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_3_class_30_inverted.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_a_h_no_mask_3_class_70___sampling_based_"></a>
## p1_a_h_no_mask_3_class_70       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_no_mask_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.7 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_3_class_70.record --write_annotations_list=2

<a id="p1_a_h_no_mask_3_class_100___sampling_based_"></a>
## p1_a_h_no_mask_3_class_100       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --seq_paths=p1_a_h_no_mask_3_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=1.0 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_3_class_100.record --write_annotations_list=2

<a id="acamp10_static3a_sampled___sampling_based_"></a>
## acamp10_static3a_sampled       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp10_static3a_sampled_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --samples_per_class=10 --write_annotations_list=2 --write_tfrecord=1

<a id="all_vid_3_class___sampling_based_"></a>
## all_vid_3_class       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_video_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/all_vid_3_class.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --write_annotations_list=2 --write_tfrecord=0

<a id="all_static_3_class___sampling_based_"></a>
## all_static_3_class       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/all_static_3_class.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --write_annotations_list=2 --write_tfrecord=0

<a id="500_static3a___sampling_based_"></a>
## 500_static3a       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp500_static3a_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --samples_per_class=500 --write_annotations_list=2 --write_tfrecord=1

<a id="inverted___500_static3a_sampling_base_d_"></a>
### inverted       @ 500_static3a/sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp500_static3a_train_inverted.record --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp500_static3a_train --exclude_loaded_samples=1 --write_annotations_list=2 --write_tfrecord=0

<a id="1k_static3a_sampled___sampling_based_"></a>
## 1K_static3a_sampled       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp1K_static3a_sampled_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --samples_per_class=1000 --write_annotations_list=2 --write_tfrecord=1

<a id="inverted___1k_static3a_sampled_sampling_based_"></a>
### inverted       @ 1K_static3a_sampled/sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp1K_static3a_sampled_train_inverted.record --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1K_static3a_sampled_train --inverted_sampling=1 --write_annotations_list=2 --write_tfrecord=0

<a id="acamp1k_static3a_test_50_yolov3_pt___sampling_based_"></a>
## acamp1K_static3a_test_50_yolov3_pt       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp1K_static3a_test.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp1K_static3a_test.record --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --sampling_ratio=0.5 --write_annotations_list=2 --write_tfrecord=0 --annotations_list_path=/data/acamp/acamp20k/acamp1K_static3a_test_50_yolov3_pt.txt

<a id="200_static3a___sampling_based_"></a>
## 200_static3a       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp200_static3a_sampled_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --samples_per_class=200 --write_annotations_list=2 --write_tfrecord=1

<a id="200_from_end_static3a___sampling_based_"></a>
## 200_from_end_static3a       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp200_from_end_static3a_sampled_train.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --samples_per_class=-200 --write_annotations_list=2 --write_tfrecord=1

<a id="inverted___200_from_end_static3a_sampling_based_"></a>
### inverted       @ 200_from_end_static3a/sampling_based

python2 csv_to_record.py --seq_paths=acamp_static_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp200_static3a_sampled_train_inverted.record --class_names_path=../labelling_tool/data/predefined_classes_20k3a.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp200_static3a_sampled_train --inverted_sampling=1 --write_annotations_list=2 --write_tfrecord=0


<a id="20k_vid3a___sampling_based_"></a>
## 20K_vid3a       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_video_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20K_vid3a.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --samples_per_class=20000 --even_sampling=1 --write_annotations_list=2 --write_tfrecord=1

<a id="inverted_200_per_class___20k_vid3a_sampling_based_"></a>
### inverted_200_per_class       @ 20K_vid3a/sampling_based

python2 csv_to_record.py --seq_paths=acamp_video_3_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20K_vid3a_inverted_200_per_class.record --class_names_path=../labelling_tool/data//predefined_classes_20k3a.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp20K_vid3a --exclude_loaded_samples=1 --samples_per_class=200 --even_sampling=1 --write_annotations_list=2 --write_tfrecord=0


<a id="20k4___sampling_based_"></a>
## 20k4       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=acamp_all_4_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k4_train.record --sampling_ratio=0.8

<a id="20k_vid4a___sampling_based_"></a>
## 20K_vid4a       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_video_4_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20K_vid4a.record --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --samples_per_class=20000 --even_sampling=1 --write_annotations_list=2 --write_tfrecord=1

<a id="inverted_200_per_class___20k_vid4a_sampling_based_"></a>
### inverted_200_per_class       @ 20K_vid4a/sampling_based

python2 csv_to_record.py --seq_paths=acamp_video_4_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20K_vid4a_inverted_200_per_class.record --class_names_path=../labelling_tool/data//predefined_classes_20k4.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp20K_vid4a --exclude_loaded_samples=1 --samples_per_class=200 --even_sampling=1 --write_annotations_list=2 --write_tfrecord=0

<a id="prototype_1_vid_4_class_no_moving_bkg___sampling_based_"></a>
## prototype_1_vid_4_class_no_moving_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=prototype_1_vid_4_class_no_moving_bkg.txt --root_dir=/data/acamp/acamp20k/prototype_1_vid  --output_path=/data/acamp/acamp20k/prototype_1_vid_4_class_no_moving_bkg.record --write_annotations_list=2 --write_tfrecord=0 annotations_list_sep=\t

<a id="p1_4_class_a_h___sampling_based_"></a>
## p1_4_class_a_h       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --output_path=/data/acamp/acamp20k/p1_4_class_a_h.record --write_annotations_list=2

<a id="p1_4_class_a_h_3_frg_per_bkg___sampling_based_"></a>
## p1_4_class_a_h_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --load_samples=p1_4_class_a_h_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/prototype_1 --output_path=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_4_class_a_h_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_4_class_a_h_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --load_samples=p1_4_class_a_h_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/prototype_1 --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_4_class_a_h_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_4_class_a_h_no_mask_3_frg_per_bkg___sampling_based_"></a>
## p1_4_class_a_h_no_mask_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_no_mask_4_class.txt --root_dir=/data/acamp/acamp20k/p1_no_mask  --load_samples=p1_4_class_a_h_no_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_no_mask --output_path=/data/acamp/acamp20k/p1_4_class_a_h_no_mask_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_4_class_a_h_no_mask_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_4_class_a_h_no_mask_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_no_mask_4_class.txt --root_dir=/data/acamp/acamp20k/p1_no_mask  --load_samples=p1_4_class_a_h_no_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_no_mask --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_4_class_a_h_no_mask_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg___sampling_based_"></a>
## p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_mask_rcnn_resnet101_4_class.txt --root_dir=/data/acamp/acamp20k/p1_mask_rcnn_resnet101  --load_samples=p1_4_class_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_mask_rcnn_resnet101 --output_path=/data/acamp/acamp20k/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_mask_rcnn_resnet101_4_class.txt --root_dir=/data/acamp/acamp20k/p1_mask_rcnn_resnet101  --load_samples=p1_4_class_a_h_mask_rcnn_resnet101_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_mask_rcnn_resnet101 --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_4_class_a_h_mask_rcnn_resnet101_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_4_class_a_h_siam_mask_3_frg_per_bkg___sampling_based_"></a>
## p1_4_class_a_h_siam_mask_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_siam_mask_4_class.txt --root_dir=/data/acamp/acamp20k/p1_siam_mask  --load_samples=p1_4_class_a_h_siam_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_siam_mask --output_path=/data/acamp/acamp20k/p1_4_class_a_h_siam_mask_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_4_class_a_h_siam_mask_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_4_class_a_h_siam_mask_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_siam_mask_4_class.txt --root_dir=/data/acamp/acamp20k/p1_siam_mask  --load_samples=p1_4_class_a_h_siam_mask_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_siam_mask --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_4_class_a_h_siam_mask_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg___sampling_based_"></a>
## p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_siam_mask_davis_4_class.txt --root_dir=/data/acamp/acamp20k/p1_siam_mask_davis  --load_samples=p1_4_class_a_h_siam_mask_davis_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_siam_mask_davis --output_path=/data/acamp/acamp20k/p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg.record --write_annotations_list=2

<a id="inverted___p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg_sampling_base_d_"></a>
### inverted       @ p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_siam_mask_davis_4_class.txt --root_dir=/data/acamp/acamp20k/p1_siam_mask_davis  --load_samples=p1_4_class_a_h_siam_mask_davis_1_seq_to_samples_3_frg_per_bkg.txt --load_samples_root=/data/acamp/acamp20k/p1_siam_mask_davis --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/p1_4_class_a_h_siam_mask_davis_3_frg_per_bkg_inv.record --write_annotations_list=2 --write_tfrecord=0


<a id="p1_a_h_4_class_30___sampling_based_"></a>
## p1_a_h_4_class_30       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --sampling_ratio=0.3 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_4_class_30.record --write_annotations_list=2

<a id="inverted___p1_a_h_4_class_30_sampling_based_"></a>
### inverted       @ p1_a_h_4_class_30/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --sampling_ratio=0.3 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_4_class_30_inverted.record --write_annotations_list=2 --inverted_sampling=1 --write_tfrecord=0

<a id="p1_a_h_4_class_50___sampling_based_"></a>
## p1_a_h_4_class_50       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --sampling_ratio=0.5 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_4_class_50.record

<a id="p1_a_h_4_class_70___sampling_based_"></a>
## p1_a_h_4_class_70       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1  --sampling_ratio=0.7 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_4_class_70.record

<a id="p1_a_h_no_mask_4_class_10___sampling_based_"></a>
## p1_a_h_no_mask_4_class_10       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_no_mask_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.1 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_4_class_10.record --write_annotations_list=2

<a id="p1_a_h_no_mask_4_class_30___sampling_based_"></a>
## p1_a_h_no_mask_4_class_30       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_no_mask_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.3 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_4_class_30.record --write_annotations_list=2

<a id="inverted___p1_a_h_no_mask_4_class_30_sampling_based_"></a>
### inverted       @ p1_a_h_no_mask_4_class_30/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_no_mask_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.3 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_4_class_30_inverted.record --write_annotations_list=2 --inverted_sampling=1 --write_tfrecord=0

<a id="p1_a_h_no_mask_4_class_50___sampling_based_"></a>
## p1_a_h_no_mask_4_class_50       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_no_mask_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.5 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_4_class_50.record --write_annotations_list=2

<a id="p1_a_h_no_mask_4_class_70___sampling_based_"></a>
## p1_a_h_no_mask_4_class_70       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k4.txt --seq_paths=p1_a_h_no_mask_4_class.txt --root_dir=/data/acamp/acamp20k/prototype_1_no_mask  --sampling_ratio=0.7 --even_sampling=1 --output_path=/data/acamp/acamp20k/p1_a_h_no_mask_4_class_70.record --write_annotations_list=2

<a id="20k6___sampling_based_"></a>
## 20k6       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_train.record --sampling_ratio=0.8


<a id="test___20k6_sampling_base_d_"></a>
### test       @ 20k6/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_test.record --sampling_ratio=-0.2

<a id="20k6_5___sampling_based_"></a>
## 20k6_5       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_train.record --sampling_ratio=0.05 --write_annotations_list=1

<a id="yolov3_pt_annotations___20k6_5_sampling_base_d_"></a>
### yolov3_pt_annotations       @ 20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_train.record --sampling_ratio=0.05 --write_annotations_list=2 --write_tfrecord=0 --annotations_list_path=/data/acamp/acamp20k/acamp20k6_5_train_yolov3_pt.txt

<a id="inverted_annotations_only_vid___20k6_5_sampling_base_d_"></a>
### inverted_annotations_only_vid       @ 20k6_5/sampling_based

python2 csv_to_record.py --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp20k6_5_inv_only_vid.record --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp20k6_5_train --exclude_loaded_samples=1 --write_annotations_list=2 --write_tfrecord=0

<a id="inverted_annotations___20k6_5_sampling_base_d_"></a>
### inverted_annotations       @ 20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_inverted.record --sampling_ratio=0.05 --write_annotations_list=1 --write_tfrecord=0 --inverted_sampling=1

<a id="yolov3_pt___inverted_annotations_20k6_5_sampling_based_"></a>
#### yolov3_pt       @ inverted_annotations/20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_inverted.record --sampling_ratio=0.05 --write_annotations_list=2 --write_tfrecord=0 --inverted_sampling=1 --annotations_list_path=/data/acamp/acamp20k/acamp20k6_5_inverted_yolov3_pt.txt

<a id="video___yolov3_pt_inverted_annotations_20k6_5_sampling_based_"></a>
##### video       @ yolov3_pt/inverted_annotations/20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_video_inverted.record --sampling_ratio=0.05 --write_annotations_list=2 --write_tfrecord=0 --inverted_sampling=1 --annotations_list_path=/data/acamp/acamp20k/acamp20k6_5_video_inverted_yolov3_pt.txt

<a id="bear_10_3___yolov3_pt_inverted_annotations_20k6_5_sampling_based_"></a>
##### bear_10_3       @ yolov3_pt/inverted_annotations/20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=bear_10_3 --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_bear_10_3_inverted.record --sampling_ratio=0.05 --write_annotations_list=2 --write_tfrecord=0 --inverted_sampling=1 --annotations_list_path=/data/acamp/acamp20k/acamp20k6_5_bear_10_3_inverted_yolov3_pt.txt

<a id="bear_1_1___yolov3_pt_inverted_annotations_20k6_5_sampling_based_"></a>
##### bear_1_1       @ yolov3_pt/inverted_annotations/20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=bear_1_1 --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_bear_1_1_inverted.record --sampling_ratio=0.05 --write_annotations_list=2 --write_tfrecord=0 --inverted_sampling=1 --annotations_list_path=/data/acamp/acamp20k/acamp20k6_5_bear_1_1_inverted_yolov3_pt.txt

<a id="coyote_jesse_1_1___yolov3_pt_inverted_annotations_20k6_5_sampling_based_"></a>
##### coyote_jesse_1_1       @ yolov3_pt/inverted_annotations/20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=coyote_jesse_1_1 --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/coyote_jesse_1_1_5_inverted --sampling_ratio=0.05 --write_annotations_list=2 --write_tfrecord=0 --inverted_sampling=1 --annotations_list_path=/data/acamp/acamp20k/coyote_jesse_1_1_5_inverted_yolov3_pt.txt

<a id="5_from_end_annotations___20k6_5_sampling_base_d_"></a>
### 5_from_end_annotations       @ 20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_from_end.record --sampling_ratio=-0.05 --write_annotations_list=1 --write_tfrecord=0 --inverted_sampling=0

<a id="1_from_end_annotations___20k6_5_sampling_base_d_"></a>
### 1_from_end_annotations       @ 20k6_5/sampling_based

__pt__

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_1_from_end.record --sampling_ratio=-0.01 --write_annotations_list=2 --write_tfrecord=0 --inverted_sampling=0

<a id="yolov3_pt___1_from_end_annotations_20k6_5_sampling_based_"></a>
#### yolov3_pt       @ 1_from_end_annotations/20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_5_from_end.record --sampling_ratio=-0.05 --write_annotations_list=2 --write_tfrecord=0 --inverted_sampling=0 --annotations_list_path=/data/acamp/acamp20k/acamp20k6_5_from_end_yolov3_pt.txt

<a id="dummy_0_005___1_from_end_annotations_20k6_5_sampling_based_"></a>
#### dummy_0.005       @ 1_from_end_annotations/20k6_5/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_0p5.record --sampling_ratio=0.005 --write_annotations_list=1 --write_tfrecord=0

<a id="20k6_60___sampling_based_"></a>
## 20k6_60       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_60_train.record --sampling_ratio=0.6 --write_annotations_list=1

<a id="yolov3_pt_annotations___20k6_60_sampling_based_"></a>
### yolov3_pt_annotations       @ 20k6_60/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_60_train.record --sampling_ratio=0.6 --write_annotations_list=2 --write_tfrecord=0 --annotations_list_path=/data/acamp/acamp20k/acamp20k6_60_train_yolov3_pt.txt

<a id="inverted_annotations___20k6_60_sampling_based_"></a>
### inverted_annotations       @ 20k6_60/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_60_inverted.record --sampling_ratio=0.6 --write_annotations_list=1 --write_tfrecord=0 --inverted_sampling=1

<a id="yolov3_pt___inverted_annotations_20k6_60_sampling_base_d_"></a>
#### yolov3_pt       @ inverted_annotations/20k6_60/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp20k6_60_inverted.record --sampling_ratio=0.6 --write_annotations_list=2 --write_tfrecord=0 --inverted_sampling=1 --annotations_list_path=/data/acamp/acamp20k/acamp20k6_60_inverted_yolov3_pt.txt

<a id="1_per_seq_6_class_vid_67___sampling_based_"></a>
## 1_per_seq_6_class_vid_67       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video_67.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/1_per_seq_6_class_vid_67.record --samples_per_seq=1 --write_annotations_list=2 --even_sampling=1

<a id="2_per_seq_6_class_vid_67___sampling_based_"></a>
## 2_per_seq_6_class_vid_67       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video_67.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/2_per_seq_6_class_vid_67.record --samples_per_seq=2 --write_annotations_list=2 --even_sampling=1

<a id="5_per_seq_6_class_vid_67___sampling_based_"></a>
## 5_per_seq_6_class_vid_67       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video_67.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/5_per_seq_6_class_vid_67.record --samples_per_seq=5 --write_annotations_list=2 --even_sampling=1


<a id="10_per_seq_6_class_vid_67___sampling_based_"></a>
## 10_per_seq_6_class_vid_67       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video_67.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/10_per_seq_6_class_vid_67.record --samples_per_seq=10 --write_annotations_list=2 --even_sampling=1

<a id="1_5_10_per_seq_6_class_vid_67_inverted___sampling_based_"></a>
## 1_5_10_per_seq_6_class_vid_67_inverted       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video_67.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/1_5_10_per_seq_6_class_vid_67_inverted.record --load_samples=1_per_seq_6_class_vid_67,5_per_seq_6_class_vid_67,10_per_seq_6_class_vid_67 --load_samples_root=/data/acamp/acamp20k --exclude_loaded_samples=1 --write_annotations_list=2 --write_tfrecord=0

<a id="bear_1_1___1_5_10_per_seq_6_class_vid_67_inverted_sampling_base_d_"></a>
### bear_1_1       @ 1_5_10_per_seq_6_class_vid_67_inverted/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=bear_1_1 --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/1_5_10_per_seq_6_class_bear_1_1_inverted.record --load_samples=1_per_seq_6_class_vid_67,5_per_seq_6_class_vid_67,10_per_seq_6_class_vid_67 --load_samples_root=/data/acamp/acamp20k --exclude_loaded_samples=1 --write_annotations_list=2 --write_tfrecord=0

<a id="1_2_5_10_per_seq_6_class_vid_67_inverted___sampling_based_"></a>
## 1_2_5_10_per_seq_6_class_vid_67_inverted       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video_67.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/1_2_5_10_per_seq_6_class_vid_67_inverted.record --load_samples=1_per_seq_6_class_vid_67,2_per_seq_6_class_vid_67,5_per_seq_6_class_vid_67,10_per_seq_6_class_vid_67 --load_samples_root=/data/acamp/acamp20k --exclude_loaded_samples=1 --write_annotations_list=2 --write_tfrecord=0


<a id="acamp_switch_6_class___sampling_based_"></a>
## acamp_switch_6_class       @ sampling_based

<a id="100_per_class___acamp_switch_6_class_sampling_base_d_"></a>
### 100_per_class       @ acamp_switch_6_class/sampling_based

<a id="individual___100_per_class_acamp_switch_6_class_sampling_base_d_"></a>
#### individual       @ 100_per_class/acamp_switch_6_class/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_1.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_1_100.record --samples_per_class=100 --even_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_2.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_2_100.record --samples_per_class=100 --even_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_3.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_3_100.record --samples_per_class=100 --even_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_4.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_4_100.record --samples_per_class=100 --even_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_5.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_5_100.record --samples_per_class=100 --even_sampling=1

<a id="combined___100_per_class_acamp_switch_6_class_sampling_base_d_"></a>
#### combined       @ 100_per_class/acamp_switch_6_class/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_combined.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp_switch_6_class_combined_100.record --load_samples=acamp_switch_6_class_1_100.txt,acamp_switch_6_class_2_100.txt,acamp_switch_6_class_3_100.txt,acamp_switch_6_class_4_100.txt,acamp_switch_6_class_5_100.txt --load_samples_root=sampled_lists


<a id="sampling_for_test___100_per_class_acamp_switch_6_class_sampling_base_d_"></a>
#### sampling_for_test       @ 100_per_class/acamp_switch_6_class/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_1.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_1_100_test.record --samples_per_class=100 --even_sampling=1 --inverted_sampling=1 --only_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_2.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_2_100_test.record --samples_per_class=100 --even_sampling=1 --inverted_sampling=1 --only_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_3.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_3_100_test.record --samples_per_class=100 --even_sampling=1 --inverted_sampling=1 --only_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_4.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_4_100_test.record --samples_per_class=100 --even_sampling=1 --inverted_sampling=1 --only_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_5.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_5_100_test.record --samples_per_class=100 --even_sampling=1 --inverted_sampling=1 --only_sampling=1


<a id="20_per_class___acamp_switch_6_class_sampling_base_d_"></a>
### 20_per_class       @ acamp_switch_6_class/sampling_based

<a id="individual___20_per_class_acamp_switch_6_class_sampling_based_"></a>
#### individual       @ 20_per_class/acamp_switch_6_class/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_1.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_1_20.record --samples_per_class=20 --even_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_2.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_2_20.record --samples_per_class=20 --even_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_3.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_3_20.record --samples_per_class=20 --even_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_4.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_4_20.record --samples_per_class=20 --even_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_5.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_5_20.record --samples_per_class=20 --even_sampling=1

<a id="combined___20_per_class_acamp_switch_6_class_sampling_based_"></a>
#### combined       @ 20_per_class/acamp_switch_6_class/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_combined.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp_switch_6_class_combined_20.record --load_samples=acamp_switch_6_class_1_20.txt,acamp_switch_6_class_2_20.txt,acamp_switch_6_class_3_20.txt,acamp_switch_6_class_4_20.txt,acamp_switch_6_class_5_20.txt --load_samples_root=sampled_lists


<a id="sampling_for_test___20_per_class_acamp_switch_6_class_sampling_based_"></a>
#### sampling_for_test       @ 20_per_class/acamp_switch_6_class/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_1.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_1_to_5_20_test/1.record --samples_per_class=20 --even_sampling=1 --inverted_sampling=1 --only_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_2.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_1_to_5_20_test/2.record --samples_per_class=20 --even_sampling=1 --inverted_sampling=1 --only_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_3.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_1_to_5_20_test/3.record --samples_per_class=20 --even_sampling=1 --inverted_sampling=1 --only_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_4.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_1_to_5_20_test/4.record --samples_per_class=20 --even_sampling=1 --inverted_sampling=1 --only_sampling=1

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_switch_6_class_5.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp_switch_6_class_1_to_5_20_test/5.record --samples_per_class=20 --even_sampling=1 --inverted_sampling=1 --only_sampling=1


<a id="10k6___sampling_based_"></a>
## 10k6       @ sampling_based

<a id="evenly_sampled___10k6_sampling_base_d_"></a>
### evenly_sampled       @ 10k6/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp10k6_train.record --samples_per_class=-10000 --even_sampling=0.6

<a id="test___evenly_sampled_10k6_sampling_based_"></a>
#### test       @ evenly_sampled/10k6/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp10k6_test.record --samples_per_class=10000 --even_sampling=-0.6

<a id="10k6_entire_seq___sampling_based_"></a>
## 10k6_entire_seq       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp10k6_entire_seq.record --samples_per_class=10000 --sample_entire_seq=1 --write_annotations_list=2

<a id="inverted___10k6_entire_seq_sampling_based_"></a>
### inverted       @ 10k6_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp10k6_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp10k6_entire_seq_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="even_5___inverted_10k6_entire_seq_sampling_base_d_"></a>
#### even_5       @ inverted/10k6_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp10k6_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp10k6_entire_seq_inv_even_5.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --sampling_ratio=0.05


<a id="15k6_entire_seq___sampling_based_"></a>
## 15k6_entire_seq       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp15k6_entire_seq.record --samples_per_class=15000 --sample_entire_seq=1 --write_annotations_list=2

<a id="inverted___15k6_entire_seq_sampling_based_"></a>
### inverted       @ 15k6_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp15k6_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp15k6_entire_seq_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="even_5___inverted_15k6_entire_seq_sampling_base_d_"></a>
#### even_5       @ inverted/15k6_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp15k6_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp15k6_entire_seq_inv_even_5.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --sampling_ratio=0.05

<a id="1k6_vid_entire_seq___sampling_based_"></a>
## 1k6_vid_entire_seq       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq.record --samples_per_class=1000 --sample_entire_seq=1 --write_annotations_list=2

<a id="load___1k6_vid_entire_seq_sampling_base_d_"></a>
### load       @ 1k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq --output_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq.record --write_annotations_list=0 --write_tfrecord=1


<a id="inverted___1k6_vid_entire_seq_sampling_base_d_"></a>
### inverted       @ 1k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="even_5___inverted_1k6_vid_entire_seq_sampling_based_"></a>
#### even_5       @ inverted/1k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv_even_5.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --sampling_ratio=0.05

<a id="1_per_seq___inverted_1k6_vid_entire_seq_sampling_based_"></a>
#### 1_per_seq       @ inverted/1k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp1k6_vid_entire_seq_inv_1_per_seq.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --samples_per_seq=1

<a id="10k6_vid_entire_seq___sampling_based_"></a>
## 10k6_vid_entire_seq       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq.record --samples_per_class=10000 --sample_entire_seq=1 --write_annotations_list=2

<a id="load___10k6_vid_entire_seq_sampling_based_"></a>
### load       @ 10k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq --output_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq.record --write_annotations_list=0 --write_tfrecord=0 only_sampling=1

<a id="inverted___10k6_vid_entire_seq_sampling_based_"></a>
### inverted       @ 10k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="even_5___inverted_10k6_vid_entire_seq_sampling_base_d_"></a>
#### even_5       @ inverted/10k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv_even_5.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --sampling_ratio=0.05

<a id="1_per_seq___inverted_10k6_vid_entire_seq_sampling_base_d_"></a>
#### 1_per_seq       @ inverted/10k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp10k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp10k6_vid_entire_seq_inv_1_per_seq.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --samples_per_seq=1

<a id="15k6_vid_entire_seq___sampling_based_"></a>
## 15k6_vid_entire_seq       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp15k6_vid_entire_seq.record --samples_per_class=15000 --sample_entire_seq=1 --write_annotations_list=2

<a id="inverted___15k6_vid_entire_seq_sampling_based_"></a>
### inverted       @ 15k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp15k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="even_5___inverted_15k6_vid_entire_seq_sampling_base_d_"></a>
#### even_5       @ inverted/15k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp15k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv_even_5.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --sampling_ratio=0.05

<a id="1_per_seq___inverted_15k6_vid_entire_seq_sampling_base_d_"></a>
#### 1_per_seq       @ inverted/15k6_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_20k6.txt --seq_paths=acamp_all_6_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp15k6_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp15k6_vid_entire_seq_inv_1_per_seq.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --samples_per_seq=1


<a id="4k8___sampling_based_"></a>
## 4k8       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_all_8_class.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/acamp4k8_train.record --class_names_path=../labelling_tool/data//predefined_classes_4k8.txt --samples_per_class=4000 --even_sampling=1

<a id="1k8_vid_even_min_1___sampling_based_"></a>
## 1k8_vid_even_min_1       @ sampling_based

python2 csv_to_record.py --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/1k8_vid_even_min_1.record --class_names_path=../labelling_tool/data//predefined_classes_4k8.txt --samples_per_class=1000 --even_sampling=1  --write_annotations_list=2 --write_tfrecord=1 --allow_seq_skipping=0 --min_samples_per_seq=1

<a id="inverted___1k8_vid_even_min_1_sampling_base_d_"></a>
### inverted       @ 1k8_vid_even_min_1/sampling_based

python2 csv_to_record.py --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/1k8_vid_even_min_1_inv.record --class_names_path=../labelling_tool/data//predefined_classes_4k8.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1 --inverted_sampling=1  --write_annotations_list=2 --write_tfrecord=0

<a id="bison___inverted_1k8_vid_even_min_1_sampling_based_"></a>
#### bison       @ inverted/1k8_vid_even_min_1/sampling_based

python2 csv_to_record.py --seq_paths=acamp_bison.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/1k8_vid_even_min_1_inv_bison.record --class_names_path=../labelling_tool/data//predefined_classes_4k8.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1 --inverted_sampling=1  --write_annotations_list=0 --write_tfrecord=0

<a id="100_per_class___inverted_1k8_vid_even_min_1_sampling_based_"></a>
#### 100_per_class       @ inverted/1k8_vid_even_min_1/sampling_based

python2 csv_to_record.py --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k --output_path=/data/acamp/acamp20k/1k8_vid_even_min_1_inverted_100_per_class.record --class_names_path=../labelling_tool/data//predefined_classes_4k8.txt --load_samples=1 --load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1 --exclude_loaded_samples=1 --samples_per_class=100 --even_sampling=1  --write_annotations_list=2 --write_tfrecord=0

<a id="1_per_seq___inverted_1k8_vid_even_min_1_sampling_based_"></a>
#### 1_per_seq       @ inverted/1k8_vid_even_min_1/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_4k8.txt --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/1k8_vid_even_min_1 --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/1k8_vid_even_min_1_inv_1_per_seq.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --samples_per_seq=1


<a id="1k8_vid_entire_seq___sampling_based_"></a>
## 1k8_vid_entire_seq       @ sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_4k8.txt --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k/ --output_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq.record --samples_per_class=1000 --sample_entire_seq=1 --write_annotations_list=2

<a id="load___1k8_vid_entire_seq_sampling_base_d_"></a>
### load       @ 1k8_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_4k8.txt --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq --output_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq.record --write_annotations_list=0 --write_tfrecord=0

<a id="inverted___1k8_vid_entire_seq_sampling_base_d_"></a>
### inverted       @ 1k8_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_4k8.txt --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv.record --write_annotations_list=2 --write_tfrecord=0

<a id="even_5___inverted_1k8_vid_entire_seq_sampling_based_"></a>
#### even_5       @ inverted/1k8_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_4k8.txt --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv_even_5.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --sampling_ratio=0.05

<a id="1_per_seq___inverted_1k8_vid_entire_seq_sampling_based_"></a>
#### 1_per_seq       @ inverted/1k8_vid_entire_seq/sampling_based

python2 csv_to_record.py --class_names_path=../labelling_tool/data/predefined_classes_4k8.txt --seq_paths=acamp_all_8_class_video.txt --root_dir=/data/acamp/acamp20k  --load_samples=1 --load_samples_root=/data/acamp/acamp20k/acamp1k8_vid_entire_seq --exclude_loaded_samples=1 --output_path=/data/acamp/acamp20k/acamp1k8_vid_entire_seq_inv_1_per_seq.record --write_annotations_list=2 --write_tfrecord=0 --even_sampling=1 --samples_per_seq=1




