<!-- MarkdownTOC -->

- [train](#train)
    - [bear_1_1       @ train](#bear11__train)
        - [inception_resnet_v2       @ bear_1_1/train](#inception_resnet_v2__bear11train)
            - [size_400_max_1_res_66       @ inception_resnet_v2/bear_1_1/train](#size_400_max1res_66__inception_resnet_v2bear11train)
            - [half_size       @ inception_resnet_v2/bear_1_1/train](#half_size__inception_resnet_v2bear11train)
        - [resnet101       @ bear_1_1/train](#resnet101__bear11train)
            - [full_size       @ resnet101/bear_1_1/train](#full_size__resnet101bear11train)
            - [size_400_res_60       @ resnet101/bear_1_1/train](#size_400_res_60__resnet101bear11train)
            - [max_1_size_400_res_60       @ resnet101/bear_1_1/train](#max1size_400_res_60__resnet101bear11train)
                - [0x682_50       @ max_1_size_400_res_60/resnet101/bear_1_1/train](#0x682_50__max1size_400_res_60resnet101bear11train)
            - [half_size       @ resnet101/bear_1_1/train](#half_size__resnet101bear11train)
        - [inception_v2       @ bear_1_1/train](#inception_v2__bear11train)
            - [max_1_size_200_res_90       @ inception_v2/bear_1_1/train](#max1size_200_res_90__inception_v2bear11train)
        - [resnet50       @ bear_1_1/train](#resnet50__bear11train)
            - [max_1_size_400_res_60       @ resnet50/bear_1_1/train](#max1size_400_res_60__resnet50bear11train)
    - [bear_13_seq_103_frames       @ train](#bear_13_seq_103_frames__train)
        - [inception_resnet_v2       @ bear_13_seq_103_frames/train](#inception_resnet_v2__bear_13_seq_103_framestrain)
            - [size_400_max_1_res_66       @ inception_resnet_v2/bear_13_seq_103_frames/train](#size_400_max1res_66__inception_resnet_v2bear_13_seq_103_framestrain)
        - [resnet101       @ bear_13_seq_103_frames/train](#resnet101__bear_13_seq_103_framestrain)
            - [full_size       @ resnet101/bear_13_seq_103_frames/train](#full_size__resnet101bear_13_seq_103_framestrain)
            - [max_1_size_400_res_60       @ resnet101/bear_13_seq_103_frames/train](#max1size_400_res_60__resnet101bear_13_seq_103_framestrain)
            - [max_1_size_200_res_75       @ resnet101/bear_13_seq_103_frames/train](#max1size_200_res_75__resnet101bear_13_seq_103_framestrain)
        - [inception_v2       @ bear_13_seq_103_frames/train](#inception_v2__bear_13_seq_103_framestrain)
            - [max_1_size_200_res_90       @ inception_v2/bear_13_seq_103_frames/train](#max1size_200_res_90__inception_v2bear_13_seq_103_framestrain)
        - [resnet50       @ bear_13_seq_103_frames/train](#resnet50__bear_13_seq_103_framestrain)
            - [max_1_size_400_res_60       @ resnet50/bear_13_seq_103_frames/train](#max1size_400_res_60__resnet50bear_13_seq_103_framestrain)
    - [bear_13_seq_103_frames_ex1       @ train](#bear_13_seq_103_frames_ex1__train)
        - [resnet101       @ bear_13_seq_103_frames_ex1/train](#resnet101__bear_13_seq_103_frames_ex1train)
            - [max_1_size_400_res_60_0x682_1       @ resnet101/bear_13_seq_103_frames_ex1/train](#max1size_400_res_60_0x682_1__resnet101bear_13_seq_103_frames_ex1train)
    - [coyote_9_seq_54_frames       @ train](#coyote9seq_54_frames__train)
        - [resnet101       @ coyote_9_seq_54_frames/train](#resnet101__coyote9seq_54_framestrain)
            - [max_1_size_400_res_60_0x682_1       @ resnet101/coyote_9_seq_54_frames/train](#max1size_400_res_60_0x682_1__resnet101coyote9seq_54_framestrain)
                - [rt\(no_mp\)       @ max_1_size_400_res_60_0x682_1/resnet101/coyote_9_seq_54_frames/train](#rtno_mp__max1size_400_res_60_0x682_1resnet101coyote9seq_54_framestrain)
                - [mp       @ max_1_size_400_res_60_0x682_1/resnet101/coyote_9_seq_54_frames/train](#mp__max1size_400_res_60_0x682_1resnet101coyote9seq_54_framestrain)
    - [deer_11_seq_56_frames       @ train](#deer_11_seq_56_frames__train)
        - [resnet101       @ deer_11_seq_56_frames/train](#resnet101__deer_11_seq_56_framestrain)
            - [max_1_size_400_res_60_0x682_1       @ resnet101/deer_11_seq_56_frames/train](#max1size_400_res_60_0x682_1__resnet101deer_11_seq_56_framestrain)
            - [max_1_size_400_res_75_0x682_1       @ resnet101/deer_11_seq_56_frames/train](#max1size_400_res_75_0x682_1__resnet101deer_11_seq_56_framestrain)
    - [moose_10_seq_50_frames       @ train](#moose_10_seq_50_frames__train)
        - [resnet101       @ moose_10_seq_50_frames/train](#resnet101__moose_10_seq_50_framestrain)
            - [max_1_size_400_res_60_0x682_1       @ resnet101/moose_10_seq_50_frames/train](#max1size_400_res_60_0x682_1__resnet101moose_10_seq_50_framestrain)
- [tf_api_eval](#tf_api_eval)
    - [bear_1_1_100       @ tf_api_eval](#bear11_100__tf_api_eval)
        - [inception_resnet_v2       @ bear_1_1_100/tf_api_eval](#inception_resnet_v2__bear11_100tf_api_eval)
            - [full_size       @ inception_resnet_v2/bear_1_1_100/tf_api_eval](#full_size__inception_resnet_v2bear11_100tf_api_eval)
                - [bear_1_1_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval](#bear11_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval)
                - [bear_1_2_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval](#bear12_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval)
                - [bear_1_3_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval](#bear13_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval)
                - [bear_1_4_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval](#bear14_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval)
                - [bear_1_5_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval](#bear15_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval)
            - [size_400_max_1_res_66       @ inception_resnet_v2/bear_1_1_100/tf_api_eval](#size_400_max1res_66__inception_resnet_v2bear11_100tf_api_eval)
                - [bear_1_1_to_1_6_0x0_0_test       @ size_400_max_1_res_66/inception_resnet_v2/bear_1_1_100/tf_api_eval](#bear11_to16_0x00test__size_400_max1res_66inception_resnet_v2bear11_100tf_api_eval)
                - [bear_1_1_to_1_6_0x682_10_test       @ size_400_max_1_res_66/inception_resnet_v2/bear_1_1_100/tf_api_eval](#bear11_to16_0x682_10_test__size_400_max1res_66inception_resnet_v2bear11_100tf_api_eval)
        - [resnet101       @ bear_1_1_100/tf_api_eval](#resnet101__bear11_100tf_api_eval)
            - [full_size       @ resnet101/bear_1_1_100/tf_api_eval](#full_size__resnet101bear11_100tf_api_eval)
                - [masks_0x0_0_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval](#masks_0x00test__full_sizeresnet101bear11_100tf_api_eval)
                - [masks_0x1365_10       @ full_size/resnet101/bear_1_1_100/tf_api_eval](#masks_0x1365_10__full_sizeresnet101bear11_100tf_api_eval)
                - [masks_0x0_10       @ full_size/resnet101/bear_1_1_100/tf_api_eval](#masks_0x0_10__full_sizeresnet101bear11_100tf_api_eval)
                - [masks_0x0_0.6       @ full_size/resnet101/bear_1_1_100/tf_api_eval](#masks_0x0_06__full_sizeresnet101bear11_100tf_api_eval)
                - [masks_800x1365_0p586_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval](#masks_800x1365_0p586_test__full_sizeresnet101bear11_100tf_api_eval)
                - [bear_1_1_to_1_5_0x0_0_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval](#bear11_to15_0x00test__full_sizeresnet101bear11_100tf_api_eval)
                - [bear_1_6_0x0_0_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval](#bear16_0x00test__full_sizeresnet101bear11_100tf_api_eval)
                - [bear_1_1_to_1_6_0x682_10_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval](#bear11_to16_0x682_10_test__full_sizeresnet101bear11_100tf_api_eval)
            - [size_400_res_60       @ resnet101/bear_1_1_100/tf_api_eval](#size_400_res_60__resnet101bear11_100tf_api_eval)
                - [bear_1_1_to_1_6_0x0_0_test       @ size_400_res_60/resnet101/bear_1_1_100/tf_api_eval](#bear11_to16_0x00test__size_400_res_60resnet101bear11_100tf_api_eval)
                    - [combined       @ bear_1_1_to_1_6_0x0_0_test/size_400_res_60/resnet101/bear_1_1_100/tf_api_eval](#combined__bear11_to16_0x00testsize_400_res_60resnet101bear11_100tf_api_eval)
                - [bear_1_1_to_1_6_0x682_10_test       @ size_400_res_60/resnet101/bear_1_1_100/tf_api_eval](#bear11_to16_0x682_10_test__size_400_res_60resnet101bear11_100tf_api_eval)
            - [size_400_res_60_0x682_50       @ resnet101/bear_1_1_100/tf_api_eval](#size_400_res_60_0x682_50__resnet101bear11_100tf_api_eval)
                - [bear_1_1_to_1_6_0x682_50_test       @ size_400_res_60_0x682_50/resnet101/bear_1_1_100/tf_api_eval](#bear11_to16_0x682_50_test__size_400_res_60_0x682_50resnet101bear11_100tf_api_eval)
            - [half_size       @ resnet101/bear_1_1_100/tf_api_eval](#half_size__resnet101bear11_100tf_api_eval)
                - [bear_1_1_0x0_100_test       @ half_size/resnet101/bear_1_1_100/tf_api_eval](#bear11_0x0_100_test__half_sizeresnet101bear11_100tf_api_eval)
                - [masks_200x200_0_test       @ half_size/resnet101/bear_1_1_100/tf_api_eval](#masks_200x2000test__half_sizeresnet101bear11_100tf_api_eval)
                    - [image000149_0_bear.jpg       @ masks_200x200_0_test/half_size/resnet101/bear_1_1_100/tf_api_eval](#image0001490bearjpg__masks_200x2000testhalf_sizeresnet101bear11_100tf_api_eval)
        - [inception_v2       @ bear_1_1_100/tf_api_eval](#inception_v2__bear11_100tf_api_eval)
            - [max_1_size_200_res_90       @ inception_v2/bear_1_1_100/tf_api_eval](#max1size_200_res_90__inception_v2bear11_100tf_api_eval)
                - [bear_1_1_0x0_0_test       @ max_1_size_200_res_90/inception_v2/bear_1_1_100/tf_api_eval](#bear11_0x00test__max1size_200_res_90inception_v2bear11_100tf_api_eval)
                - [bear_1_1_to_1_6_0x0_0_test       @ max_1_size_200_res_90/inception_v2/bear_1_1_100/tf_api_eval](#bear11_to16_0x00test__max1size_200_res_90inception_v2bear11_100tf_api_eval)
                - [bear_1_1_to_1_6_0x682_10_test       @ max_1_size_200_res_90/inception_v2/bear_1_1_100/tf_api_eval](#bear11_to16_0x682_10_test__max1size_200_res_90inception_v2bear11_100tf_api_eval)
        - [resnet50       @ bear_1_1_100/tf_api_eval](#resnet50__bear11_100tf_api_eval)
            - [max_1_size_400_res_60       @ resnet50/bear_1_1_100/tf_api_eval](#max1size_400_res_60__resnet50bear11_100tf_api_eval)
                - [0x0_0       @ max_1_size_400_res_60/resnet50/bear_1_1_100/tf_api_eval](#0x0_0__max1size_400_res_60resnet50bear11_100tf_api_eval)
    - [bear_13_seq_103_frames_0x682_10       @ tf_api_eval](#bear_13_seq_103_frames_0x682_10__tf_api_eval)
        - [inception_resnet_v2       @ bear_13_seq_103_frames_0x682_10/tf_api_eval](#inception_resnet_v2__bear_13_seq_103_frames_0x682_10tf_api_eval)
            - [size_400_max_1_res_66       @ inception_resnet_v2/bear_13_seq_103_frames_0x682_10/tf_api_eval](#size_400_max1res_66__inception_resnet_v2bear_13_seq_103_frames_0x682_10tf_api_eval)
        - [resnet101       @ bear_13_seq_103_frames_0x682_10/tf_api_eval](#resnet101__bear_13_seq_103_frames_0x682_10tf_api_eval)
            - [full_size       @ resnet101/bear_13_seq_103_frames_0x682_10/tf_api_eval](#full_size__resnet101bear_13_seq_103_frames_0x682_10tf_api_eval)
            - [max_1_size_400_res_60       @ resnet101/bear_13_seq_103_frames_0x682_10/tf_api_eval](#max1size_400_res_60__resnet101bear_13_seq_103_frames_0x682_10tf_api_eval)
        - [inception_v2       @ bear_13_seq_103_frames_0x682_10/tf_api_eval](#inception_v2__bear_13_seq_103_frames_0x682_10tf_api_eval)
            - [max_1_size_200_res_90       @ inception_v2/bear_13_seq_103_frames_0x682_10/tf_api_eval](#max1size_200_res_90__inception_v2bear_13_seq_103_frames_0x682_10tf_api_eval)
        - [resnet50       @ bear_13_seq_103_frames_0x682_10/tf_api_eval](#resnet50__bear_13_seq_103_frames_0x682_10tf_api_eval)
            - [max_1_size_400_res_60       @ resnet50/bear_13_seq_103_frames_0x682_10/tf_api_eval](#max1size_400_res_60__resnet50bear_13_seq_103_frames_0x682_10tf_api_eval)
    - [bear_13_seq_103_frames_ex1       @ tf_api_eval](#bear_13_seq_103_frames_ex1__tf_api_eval)
        - [resnet101       @ bear_13_seq_103_frames_ex1/tf_api_eval](#resnet101__bear_13_seq_103_frames_ex1tf_api_eval)
            - [max_1_size_400_res_60_0x682_1       @ resnet101/bear_13_seq_103_frames_ex1/tf_api_eval](#max1size_400_res_60_0x682_1__resnet101bear_13_seq_103_frames_ex1tf_api_eval)
                - [p1_source_bear_0x682_1       @ max_1_size_400_res_60_0x682_1/resnet101/bear_13_seq_103_frames_ex1/tf_api_eval](#p1_source_bear_0x682_1__max1size_400_res_60_0x682_1resnet101bear_13_seq_103_frames_ex1tf_api_eval)
    - [coyote_9_seq_54_frames       @ tf_api_eval](#coyote9seq_54_frames__tf_api_eval)
        - [resnet101       @ coyote_9_seq_54_frames/tf_api_eval](#resnet101__coyote9seq_54_framestf_api_eval)
            - [p1_source_coyote_0x682_1       @ resnet101/coyote_9_seq_54_frames/tf_api_eval](#p1_source_coyote_0x682_1__resnet101coyote9seq_54_framestf_api_eval)
                - [mp       @ p1_source_coyote_0x682_1/resnet101/coyote_9_seq_54_frames/tf_api_eval](#mp__p1_source_coyote_0x682_1resnet101coyote9seq_54_framestf_api_eval)
    - [deer_11_seq_56_frames       @ tf_api_eval](#deer_11_seq_56_frames__tf_api_eval)
        - [resnet101       @ deer_11_seq_56_frames/tf_api_eval](#resnet101__deer_11_seq_56_framestf_api_eval)
            - [p1_source_deer_0x682_1       @ resnet101/deer_11_seq_56_frames/tf_api_eval](#p1_source_deer_0x682_1__resnet101deer_11_seq_56_framestf_api_eval)
    - [moose_10_seq_50_frames       @ tf_api_eval](#moose_10_seq_50_frames__tf_api_eval)
        - [resnet101       @ moose_10_seq_50_frames/tf_api_eval](#resnet101__moose_10_seq_50_framestf_api_eval)
            - [p1_source_moose_0x682_1       @ resnet101/moose_10_seq_50_frames/tf_api_eval](#p1_source_moose_0x682_1__resnet101moose_10_seq_50_framestf_api_eval)
- [visualize](#visualize)
    - [bear_1_1       @ visualize](#bear11__visualize)
        - [inception_resnet_v2_size_400_max_1_res_66       @ bear_1_1/visualize](#inception_resnet_v2_size_400_max1res_66__bear11visualize)
            - [0x682_10       @ inception_resnet_v2_size_400_max_1_res_66/bear_1_1/visualize](#0x682_10__inception_resnet_v2_size_400_max1res_66bear11visualize)
                - [only_blended       @ 0x682_10/inception_resnet_v2_size_400_max_1_res_66/bear_1_1/visualize](#only_blended__0x682_10inception_resnet_v2_size_400_max1res_66bear11visualize)
        - [resnet101       @ bear_1_1/visualize](#resnet101__bear11visualize)
            - [only_blended       @ resnet101/bear_1_1/visualize](#only_blended__resnet101bear11visualize)
                - [0x682_10       @ only_blended/resnet101/bear_1_1/visualize](#0x682_10__only_blendedresnet101bear11visualize)
        - [resnet101_max_1_size_400_res_60       @ bear_1_1/visualize](#resnet101_max1size_400_res_60__bear11visualize)
            - [only_blended       @ resnet101_max_1_size_400_res_60/bear_1_1/visualize](#only_blended__resnet101_max1size_400_res_60bear11visualize)
                - [0x682_10       @ only_blended/resnet101_max_1_size_400_res_60/bear_1_1/visualize](#0x682_10__only_blendedresnet101_max1size_400_res_60bear11visualize)
            - [bear_1_4       @ resnet101_max_1_size_400_res_60/bear_1_1/visualize](#bear14__resnet101_max1size_400_res_60bear11visualize)
            - [combine_seq       @ resnet101_max_1_size_400_res_60/bear_1_1/visualize](#combine_seq__resnet101_max1size_400_res_60bear11visualize)
        - [resnet101_max_1_size_400_res_60_0x682_50       @ bear_1_1/visualize](#resnet101_max1size_400_res_60_0x682_50__bear11visualize)
        - [inception_v2_max_1_size_200_res_90       @ bear_1_1/visualize](#inception_v2_max1size_200_res_90__bear11visualize)
            - [only_blended       @ inception_v2_max_1_size_200_res_90/bear_1_1/visualize](#only_blended__inception_v2_max1size_200_res_90bear11visualize)
                - [0x682_10       @ only_blended/inception_v2_max_1_size_200_res_90/bear_1_1/visualize](#0x682_10__only_blendedinception_v2_max1size_200_res_90bear11visualize)
    - [bear_13_seq_103_frames       @ visualize](#bear_13_seq_103_frames__visualize)
        - [inception_resnet_v2_size_400_max_1_res_66       @ bear_13_seq_103_frames/visualize](#inception_resnet_v2_size_400_max1res_66__bear_13_seq_103_framesvisualize)
            - [map_to_bbox       @ inception_resnet_v2_size_400_max_1_res_66/bear_13_seq_103_frames/visualize](#map_to_bbox__inception_resnet_v2_size_400_max1res_66bear_13_seq_103_framesvisualize)
        - [resnet101_full_size       @ bear_13_seq_103_frames/visualize](#resnet101_full_size__bear_13_seq_103_framesvisualize)
        - [resnet101_max_1_size_400_res_60       @ bear_13_seq_103_frames/visualize](#resnet101_max1size_400_res_60__bear_13_seq_103_framesvisualize)
            - [only_blended       @ resnet101_max_1_size_400_res_60/bear_13_seq_103_frames/visualize](#only_blended__resnet101_max1size_400_res_60bear_13_seq_103_framesvisualize)
            - [inception_v2_max_1_size_200_res_90       @ resnet101_max_1_size_400_res_60/bear_13_seq_103_frames/visualize](#inception_v2_max1size_200_res_90__resnet101_max1size_400_res_60bear_13_seq_103_framesvisualize)
- [augment](#augment)
    - [resnet101_bear_1_1_100       @ augment](#resnet101_bear11_100__augment)
        - [bear_1_1_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment](#bear11_0x00test_masks__resnet101_bear11_100augment)
        - [bear_1_2_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment](#bear12_0x00test_masks__resnet101_bear11_100augment)
        - [bear_1_3_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment](#bear13_0x00test_masks__resnet101_bear11_100augment)
        - [bear_1_4_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment](#bear14_0x00test_masks__resnet101_bear11_100augment)
        - [bear_1_5_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment](#bear15_0x00test_masks__resnet101_bear11_100augment)
    - [resnet101_bear_13_seq_103_frames_max_1_size_400_res_60       @ augment](#resnet101_bear_13_seq_103_frames_max1size_400_res_60__augment)
        - [acamp_office       @ resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#acamp_office__resnet101_bear_13_seq_103_frames_max1size_400_res_60augment)
            - [bear_8_1       @ acamp_office/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#bear81__acamp_officeresnet101_bear_13_seq_103_frames_max1size_400_res_60augment)
            - [bear_8_1       @ acamp_office/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#bear81__acamp_officeresnet101_bear_13_seq_103_frames_max1size_400_res_60augment-1)
        - [airport       @ resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#airport__resnet101_bear_13_seq_103_frames_max1size_400_res_60augment)
            - [bear_10_1       @ airport/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#bear_10_1__airportresnet101_bear_13_seq_103_frames_max1size_400_res_60augment)
            - [bear_16_3       @ airport/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#bear_16_3__airportresnet101_bear_13_seq_103_frames_max1size_400_res_60augment)
            - [bear_jesse_7_1_w       @ airport/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#bear_jesse71_w__airportresnet101_bear_13_seq_103_frames_max1size_400_res_60augment)
        - [highway       @ resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#highway__resnet101_bear_13_seq_103_frames_max1size_400_res_60augment)
            - [bear_jesse_23_1_w       @ highway/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment](#bear_jesse_231w__highwayresnet101_bear_13_seq_103_frames_max1size_400_res_60augment)

<!-- /MarkdownTOC -->

<a id="train"></a>
# train

<a id="bear11__train"></a>
## bear_1_1       @ train

<a id="inception_resnet_v2__bear11train"></a>
### inception_resnet_v2       @ bear_1_1/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100.config --train_dir=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100 --n_steps=1000000 --save_interval_secs=600

<a id="size_400_max1res_66__inception_resnet_v2bear11train"></a>
#### size_400_max_1_res_66       @ inception_resnet_v2/bear_1_1/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66.config --train_dir=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66 --n_steps=1000000 --save_interval_secs=600

<a id="half_size__inception_resnet_v2bear11train"></a>
#### half_size       @ inception_resnet_v2/bear_1_1/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100_half_size.config --train_dir=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100_half_size --n_steps=1000000 --save_interval_secs=600

<a id="resnet101__bear11train"></a>
### resnet101       @ bear_1_1/train

<a id="full_size__resnet101bear11train"></a>
#### full_size       @ resnet101/bear_1_1/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config --train_dir=trained/mask_rcnn_resnet101_bear_1_1_100 --n_steps=1000000 --save_interval_secs=600

<a id="size_400_res_60__resnet101bear11train"></a>
#### size_400_res_60       @ resnet101/bear_1_1/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_size_400_res_60.config --train_dir=trained/mask_rcnn_resnet101_bear_1_1_100_size_400_res_60 --n_steps=1000000 --save_interval_secs=600

<a id="max1size_400_res_60__resnet101bear11train"></a>
#### max_1_size_400_res_60       @ resnet101/bear_1_1/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60.config --train_dir=trained/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60 --n_steps=1000000 --save_interval_secs=600

<a id="0x682_50__max1size_400_res_60resnet101bear11train"></a>
##### 0x682_50       @ max_1_size_400_res_60/resnet101/bear_1_1/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60_0x682_50.config --train_dir=trained/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60_0x682_50 --n_steps=1000000 --save_interval_secs=600

<a id="half_size__resnet101bear11train"></a>
#### half_size       @ resnet101/bear_1_1/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_half_size.config --train_dir=trained/mask_rcnn_resnet101_bear_1_1_100_half_size --n_steps=1000000 --save_interval_secs=600

<a id="inception_v2__bear11train"></a>
### inception_v2       @ bear_1_1/train

<a id="max1size_200_res_90__inception_v2bear11train"></a>
#### max_1_size_200_res_90       @ inception_v2/bear_1_1/train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_bear_1_1_100_max_1_size_200_res_90.config --train_dir=trained/mask_rcnn_inception_v2_bear_1_1_100_max_1_size_200_res_90 --n_steps=1000000 --save_interval_secs=600

<a id="resnet50__bear11train"></a>
### resnet50       @ bear_1_1/train

<a id="max1size_400_res_60__resnet50bear11train"></a>
#### max_1_size_400_res_60       @ resnet50/bear_1_1/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet50_atrous_coco_2018_01_28/mask_rcnn_resnet50_bear_1_1_100_max_1_size_400_res_60.config --train_dir=trained/mask_rcnn_resnet50_bear_1_1_100_max_1_size_400_res_60 --n_steps=1000000 --save_interval_secs=600

<a id="bear_13_seq_103_frames__train"></a>
## bear_13_seq_103_frames       @ train
<a id="mask_rcnn_resnet101_bear_13_seq_103_frames_max1size_400_res_60__bear_13_seq_103_framestrain"></a>

<a id="inception_resnet_v2__bear_13_seq_103_framestrain"></a>
### inception_resnet_v2       @ bear_13_seq_103_frames/train

<a id="size_400_max1res_66__inception_resnet_v2bear_13_seq_103_framestrain"></a>
#### size_400_max_1_res_66       @ inception_resnet_v2/bear_13_seq_103_frames/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_13_seq_103_frames_size_400_max_1_res_66.config --train_dir=trained/mask_rcnn_inception_resnet_v2_bear_13_seq_103_frames_size_400_max_1_res_66 --n_steps=1000000 --save_interval_secs=600

<a id="resnet101__bear_13_seq_103_framestrain"></a>
### resnet101       @ bear_13_seq_103_frames/train

<a id="full_size__resnet101bear_13_seq_103_framestrain"></a>
#### full_size       @ resnet101/bear_13_seq_103_frames/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_13_seq_103_frames.config --train_dir=trained/mask_rcnn_resnet101_bear_13_seq_103_frames --n_steps=1000000 --save_interval_secs=600

<a id="max1size_400_res_60__resnet101bear_13_seq_103_framestrain"></a>
#### max_1_size_400_res_60       @ resnet101/bear_13_seq_103_frames/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_13_seq_103_frames_max_1_size_400_res_60.config --train_dir=trained/mask_rcnn_resnet101_bear_13_seq_103_frames_max_1_size_400_res_60 --n_steps=1000000 --save_interval_secs=600

<a id="max1size_200_res_75__resnet101bear_13_seq_103_framestrain"></a>
#### max_1_size_200_res_75       @ resnet101/bear_13_seq_103_frames/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_13_seq_103_frames_max_1_size_200_res_75.config --train_dir=trained/mask_rcnn_resnet101_bear_13_seq_103_frames_max_1_size_200_res_75 --n_steps=1000000 --save_interval_secs=600

<a id="inception_v2__bear_13_seq_103_framestrain"></a>
### inception_v2       @ bear_13_seq_103_frames/train

<a id="max1size_200_res_90__inception_v2bear_13_seq_103_framestrain"></a>
#### max_1_size_200_res_90       @ inception_v2/bear_13_seq_103_frames/train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_bear_13_seq_103_frames_max_1_size_200_res_90.config --train_dir=trained/mask_rcnn_inception_v2_bear_13_seq_103_frames_max_1_size_200_res_90 --n_steps=1000000 --save_interval_secs=600

<a id="resnet50__bear_13_seq_103_framestrain"></a>
### resnet50       @ bear_13_seq_103_frames/train

<a id="max1size_400_res_60__resnet50bear_13_seq_103_framestrain"></a>
#### max_1_size_400_res_60       @ resnet50/bear_13_seq_103_frames/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet50_atrous_coco_2018_01_28/mask_rcnn_resnet50_bear_13_seq_103_frames_max_1_size_400_res_60.config --train_dir=trained/mask_rcnn_resnet50_bear_13_seq_103_frames_max_1_size_400_res_60 --n_steps=1000000 --save_interval_secs=600

<a id="bear_13_seq_103_frames_ex1__train"></a>
## bear_13_seq_103_frames_ex1       @ train

<a id="resnet101__bear_13_seq_103_frames_ex1train"></a>
### resnet101       @ bear_13_seq_103_frames_ex1/train

<a id="max1size_400_res_60_0x682_1__resnet101bear_13_seq_103_frames_ex1train"></a>
#### max_1_size_400_res_60_0x682_1       @ resnet101/bear_13_seq_103_frames_ex1/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_13_seq_103_frames_ex1_max_1_size_400_res_60_0x682_1.config --train_dir=trained/mask_rcnn_resnet101_bear_13_seq_103_frames_ex1_max_1_size_400_res_60_0x682_1 --n_steps=1000000 --save_interval_secs=600

<a id="coyote9seq_54_frames__train"></a>
## coyote_9_seq_54_frames       @ train

<a id="resnet101__coyote9seq_54_framestrain"></a>
### resnet101       @ coyote_9_seq_54_frames/train

<a id="max1size_400_res_60_0x682_1__resnet101coyote9seq_54_framestrain"></a>
#### max_1_size_400_res_60_0x682_1       @ resnet101/coyote_9_seq_54_frames/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1.config --train_dir=trained/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1 --n_steps=1000000 --save_interval_secs=600

<a id="rtno_mp__max1size_400_res_60_0x682_1resnet101coyote9seq_54_framestrain"></a>
##### rt(no_mp)       @ max_1_size_400_res_60_0x682_1/resnet101/coyote_9_seq_54_frames/train

CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1.config --train_dir=trained/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1_rt --n_steps=1000000 --save_interval_secs=600 --enable_mixed_precision=0


<a id="mp__max1size_400_res_60_0x682_1resnet101coyote9seq_54_framestrain"></a>
##### mp       @ max_1_size_400_res_60_0x682_1/resnet101/coyote_9_seq_54_frames/train

CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1.config --train_dir=trained/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1_mp --n_steps=1000000 --save_interval_secs=600 --enable_mixed_precision=1


<a id="deer_11_seq_56_frames__train"></a>
## deer_11_seq_56_frames       @ train

<a id="resnet101__deer_11_seq_56_framestrain"></a>
### resnet101       @ deer_11_seq_56_frames/train

<a id="max1size_400_res_60_0x682_1__resnet101deer_11_seq_56_framestrain"></a>
#### max_1_size_400_res_60_0x682_1       @ resnet101/deer_11_seq_56_frames/train

TF_ENABLE_AUTO_MIXED_PRECISION=1
TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=1
CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_deer_11_seq_56_frames_max_1_size_400_res_60_0x682_1.config --train_dir=trained/mask_rcnn_resnet101_deer_11_seq_56_frames_max_1_size_400_res_60_0x682_1 --n_steps=1000000 --save_interval_secs=600

<a id="max1size_400_res_75_0x682_1__resnet101deer_11_seq_56_framestrain"></a>
#### max_1_size_400_res_75_0x682_1       @ resnet101/deer_11_seq_56_frames/train

TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=1 CUDA_VISIBLE_DEVICES=1 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_deer_11_seq_56_frames_max_1_size_400_res_75_0x682_1.config --train_dir=trained/mask_rcnn_resnet101_deer_11_seq_56_frames_max_1_size_400_res_75_0x682_1 --n_steps=1000000 --save_interval_secs=600


<a id="moose_10_seq_50_frames__train"></a>
## moose_10_seq_50_frames       @ train

<a id="resnet101__moose_10_seq_50_framestrain"></a>
### resnet101       @ moose_10_seq_50_frames/train

<a id="max1size_400_res_60_0x682_1__resnet101moose_10_seq_50_framestrain"></a>
#### max_1_size_400_res_60_0x682_1       @ resnet101/moose_10_seq_50_frames/train

CUDA_VISIBLE_DEVICES=2 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_moose_10_seq_50_frames_max_1_size_400_res_60_0x682_1.config --train_dir=trained/mask_rcnn_resnet101_moose_10_seq_50_frames_max_1_size_400_res_60_0x682_1 --n_steps=1000000 --save_interval_secs=600

<a id="tf_api_eval"></a>
# tf_api_eval

<a id="bear11_100__tf_api_eval"></a>
## bear_1_1_100       @ tf_api_eval

<a id="inception_resnet_v2__bear11_100tf_api_eval"></a>
### inception_resnet_v2       @ bear_1_1_100/tf_api_eval

<a id="full_size__inception_resnet_v2bear11_100tf_api_eval"></a>
#### full_size       @ inception_resnet_v2/bear_1_1_100/tf_api_eval

<a id="bear11_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval"></a>
##### bear_1_1_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0_test pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100/inference_140101/frozen_inference_graph

<a id="bear12_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval"></a>
##### bear_1_2_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_2_0x0_0_test pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100/inference_323020/frozen_inference_graph.pb

<a id="bear13_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval"></a>
##### bear_1_3_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_3_0x0_0_test pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100/inference_323020/frozen_inference_graph.pb

<a id="bear14_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval"></a>
##### bear_1_4_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_4_0x0_0_test pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100/inference_323020/frozen_inference_graph.pb

<a id="bear15_0x00test__full_sizeinception_resnet_v2bear11_100tf_api_eval"></a>
##### bear_1_5_0x0_0_test       @ full_size/inception_resnet_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_5_0x0_0_test pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100/inference_323020/frozen_inference_graph.pb

<a id="size_400_max1res_66__inception_resnet_v2bear11_100tf_api_eval"></a>
#### size_400_max_1_res_66       @ inception_resnet_v2/bear_1_1_100/tf_api_eval

<a id="bear11_to16_0x00test__size_400_max1res_66inception_resnet_v2bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_6_0x0_0_test       @ size_400_max_1_res_66/inception_resnet_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_1_1_0x0_0_test,bear_1_2_0x0_0_test,bear_1_3_0x0_0_test,bear_1_4_0x0_0_test,bear_1_5_0x0_0_test,bear_1_6_0x0_0_test pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="bear11_to16_0x682_10_test__size_400_max1res_66inception_resnet_v2bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_6_0x682_10_test       @ size_400_max_1_res_66/inception_resnet_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_10_test pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="resnet101__bear11_100tf_api_eval"></a>
### resnet101       @ bear_1_1_100/tf_api_eval

<a id="full_size__resnet101bear11_100tf_api_eval"></a>
#### full_size       @ resnet101/bear_1_1_100/tf_api_eval

<a id="masks_0x00test__full_sizeresnet101bear11_100tf_api_eval"></a>
##### masks_0x0_0_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 

<a id="masks_0x1365_10__full_sizeresnet101bear11_100tf_api_eval"></a>
##### masks_0x1365_10       @ full_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1/masks_0x1365_10/test_images pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 

<a id="masks_0x0_10__full_sizeresnet101bear11_100tf_api_eval"></a>
##### masks_0x0_10       @ full_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_10_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 

<a id="masks_0x0_06__full_sizeresnet101bear11_100tf_api_eval"></a>
##### masks_0x0_0.6       @ full_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1/masks_0x0_0.6/test_images pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 

<a id="masks_800x1365_0p586_test__full_sizeresnet101bear11_100tf_api_eval"></a>
##### masks_800x1365_0p586_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_800x1365_0p586_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet101_bear_1_1_100/inference_323020/frozen_inference_graph

<a id="bear11_to15_0x00test__full_sizeresnet101bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_5_0x0_0_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_1_1_0x0_0_test,bear_1_2_0x0_0_test,bear_1_3_0x0_0_test,bear_1_4_0x0_0_test,bear_1_5_0x0_0_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet101_bear_1_1_100/inference_323020/frozen_inference_graph.pb

<a id="bear16_0x00test__full_sizeresnet101bear11_100tf_api_eval"></a>
##### bear_1_6_0x0_0_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_1_6_0x0_0_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="bear11_to16_0x682_10_test__full_sizeresnet101bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_6_0x682_10_test       @ full_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_10_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 out_postfix=0x682_10

<a id="size_400_res_60__resnet101bear11_100tf_api_eval"></a>
#### size_400_res_60       @ resnet101/bear_1_1_100/tf_api_eval

<a id="bear11_to16_0x00test__size_400_res_60resnet101bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_6_0x0_0_test       @ size_400_res_60/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_1_1_0x0_0_test,bear_1_2_0x0_0_test,bear_1_3_0x0_0_test,bear_1_4_0x0_0_test,bear_1_5_0x0_0_test,bear_1_6_0x0_0_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60/inference_323020/frozen_inference_graph.p

<a id="combined__bear11_to16_0x00testsize_400_res_60resnet101bear11_100tf_api_eval"></a>
###### combined       @ bear_1_1_to_1_6_0x0_0_test/size_400_res_60/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_1_2_to_1_6_0x0_0_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="bear11_to16_0x682_10_test__size_400_res_60resnet101bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_6_0x682_10_test       @ size_400_res_60/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_10_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 out_postfix=0x682_10

<a id="size_400_res_60_0x682_50__resnet101bear11_100tf_api_eval"></a>
#### size_400_res_60_0x682_50       @ resnet101/bear_1_1_100/tf_api_eval

<a id="bear11_to16_0x682_50_test__size_400_res_60_0x682_50resnet101bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_6_0x682_50_test       @ size_400_res_60_0x682_50/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60_0x682_50 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_50_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60_0x682_50.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 out_postfix=0x682_50


<a id="half_size__resnet101bear11_100tf_api_eval"></a>
#### half_size       @ resnet101/bear_1_1_100/tf_api_eval

<a id="bear11_0x0_100_test__half_sizeresnet101bear11_100tf_api_eval"></a>
##### bear_1_1_0x0_100_test       @ half_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100_half_size labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_100_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_half_size.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet101_bear_1_1_100_half_size/inference_157255/frozen_inference_graph.pb

<a id="masks_200x2000test__half_sizeresnet101bear11_100tf_api_eval"></a>
##### masks_200x200_0_test       @ half_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100_half_size labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_200x200_0_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_half_size.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet101_bear_1_1_100_half_size/inference_157255/frozen_inference_graph.pb

<a id="image0001490bearjpg__masks_200x2000testhalf_sizeresnet101bear11_100tf_api_eval"></a>
###### image000149_0_bear.jpg       @ masks_200x200_0_test/half_size/resnet101/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100_half_size labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_200x200_0_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_half_size.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet101_bear_1_1_100_half_size/inference_157255/frozen_inference_graph eval_on_single_image=image000149_0_bear.jpg

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_1_1_100_half_size labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_200x200_0_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_half_size.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet101_bear_1_1_100_half_size/inference_157255/frozen_inference_graph start_frame_id=30

<a id="inception_v2__bear11_100tf_api_eval"></a>
### inception_v2       @ bear_1_1_100/tf_api_eval

<a id="max1size_200_res_90__inception_v2bear11_100tf_api_eval"></a>
#### max_1_size_200_res_90       @ inception_v2/bear_1_1_100/tf_api_eval

<a id="bear11_0x00test__max1size_200_res_90inception_v2bear11_100tf_api_eval"></a>
##### bear_1_1_0x0_0_test       @ max_1_size_200_res_90/inception_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_v2_bear_1_1_100_max_1_size_200_res_90 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0_test pipeline_config_path=configs/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_bear_1_1_100_max_1_size_200_res_90.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="bear11_to16_0x00test__max1size_200_res_90inception_v2bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_6_0x0_0_test       @ max_1_size_200_res_90/inception_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_v2_bear_1_1_100_max_1_size_200_res_90 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_1_1_0x0_0_test,bear_1_2_0x0_0_test,bear_1_3_0x0_0_test,bear_1_4_0x0_0_test,bear_1_5_0x0_0_test,bear_1_6_0x0_0_test pipeline_config_path=configs/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_bear_1_1_100_max_1_size_200_res_90.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1


<a id="bear11_to16_0x682_10_test__max1size_200_res_90inception_v2bear11_100tf_api_eval"></a>
##### bear_1_1_to_1_6_0x682_10_test       @ max_1_size_200_res_90/inception_v2/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_v2_bear_1_1_100_max_1_size_200_res_90 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_10_test pipeline_config_path=configs/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_bear_1_1_100_max_1_size_200_res_90.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 out_postfix=0x682_10

<a id="resnet50__bear11_100tf_api_eval"></a>
### resnet50       @ bear_1_1_100/tf_api_eval

<a id="max1size_400_res_60__resnet50bear11_100tf_api_eval"></a>
#### max_1_size_400_res_60       @ resnet50/bear_1_1_100/tf_api_eval

<a id="0x0_0__max1size_400_res_60resnet50bear11_100tf_api_eval"></a>
##### 0x0_0       @ max_1_size_400_res_60/resnet50/bear_1_1_100/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet50_bear_1_1_100_max_1_size_400_res_60 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 seq_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0_test pipeline_config_path=configs/mask_rcnn_resnet50_atrous_coco_2018_01_28/mask_rcnn_resnet50_bear_1_1_100_max_1_size_400_res_60.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="bear_13_seq_103_frames_0x682_10__tf_api_eval"></a>
## bear_13_seq_103_frames_0x682_10       @ tf_api_eval

<a id="inception_resnet_v2__bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
### inception_resnet_v2       @ bear_13_seq_103_frames_0x682_10/tf_api_eval

<a id="size_400_max1res_66__inception_resnet_v2bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
#### size_400_max_1_res_66       @ inception_resnet_v2/bear_13_seq_103_frames_0x682_10/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_resnet_v2_bear_13_seq_103_frames_size_400_max_1_res_66 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_13_seq_103_frames_0x682_10_test.txt pipeline_config_path=configs/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/mask_rcnn_inception_resnet_v2_bear_13_seq_103_frames_size_400_max_1_res_66.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_inception_resnet_v2_bear_13_seq_103_frames_size_400_max_1_res_66/inference_157255/frozen_inference_graph.pb

<a id="resnet101__bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
### resnet101       @ bear_13_seq_103_frames_0x682_10/tf_api_eval

<a id="full_size__resnet101bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
#### full_size       @ resnet101/bear_13_seq_103_frames_0x682_10/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_13_seq_103_frames labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_0x682_10_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_13_seq_103_frames.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="max1size_400_res_60__resnet101bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
#### max_1_size_400_res_60       @ resnet101/bear_13_seq_103_frames_0x682_10/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_13_seq_103_frames_max_1_size_400_res_60 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_13_seq_103_frames_0x682_10_test.txt pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_13_seq_103_frames_max_1_size_400_res_60.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/inference_157255/frozen_inference_graph.pb

<a id="inception_v2__bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
### inception_v2       @ bear_13_seq_103_frames_0x682_10/tf_api_eval

<a id="max1size_200_res_90__inception_v2bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
#### max_1_size_200_res_90       @ inception_v2/bear_13_seq_103_frames_0x682_10/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_inception_v2_bear_13_seq_103_frames_max_1_size_200_res_90 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_13_seq_103_frames_0x682_10_test.txt pipeline_config_path=configs/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_bear_13_seq_103_frames_max_1_size_200_res_90.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_inception_v2_bear_13_seq_103_frames_max_1_size_200_res_90/inference_157255/frozen_inference_graph.pb

<a id="resnet50__bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
### resnet50       @ bear_13_seq_103_frames_0x682_10/tf_api_eval

<a id="max1size_400_res_60__resnet50bear_13_seq_103_frames_0x682_10tf_api_eval"></a>
#### max_1_size_400_res_60       @ resnet50/bear_13_seq_103_frames_0x682_10/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet50_bear_13_seq_103_frames_max_1_size_400_res_60 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/masks seq_paths=bear_13_seq_103_frames_0x682_10_test.txt pipeline_config_path=configs/mask_rcnn_resnet50_atrous_coco_2018_01_28/mask_rcnn_resnet50_bear_13_seq_103_frames_max_1_size_400_res_60.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 froazen_graph_path=trained/mask_rcnn_resnet50_bear_13_seq_103_frames_max_1_size_400_res_60/inference_157255/frozen_inference_graph.pb

<a id="bear_13_seq_103_frames_ex1__tf_api_eval"></a>
## bear_13_seq_103_frames_ex1       @ tf_api_eval

<a id="resnet101__bear_13_seq_103_frames_ex1tf_api_eval"></a>
### resnet101       @ bear_13_seq_103_frames_ex1/tf_api_eval

<a id="max1size_400_res_60_0x682_1__resnet101bear_13_seq_103_frames_ex1tf_api_eval"></a>
#### max_1_size_400_res_60_0x682_1       @ resnet101/bear_13_seq_103_frames_ex1/tf_api_eval

<a id="p1_source_bear_0x682_1__max1size_400_res_60_0x682_1resnet101bear_13_seq_103_frames_ex1tf_api_eval"></a>
##### p1_source_bear_0x682_1       @ max_1_size_400_res_60_0x682_1/resnet101/bear_13_seq_103_frames_ex1/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_13_seq_103_frames_ex1_max_1_size_400_res_60_0x682_1 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/p1_source_bear_0x682_1 pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_13_seq_103_frames_ex1_max_1_size_400_res_60_0x682_1.config sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="coyote9seq_54_frames__tf_api_eval"></a>
## coyote_9_seq_54_frames       @ tf_api_eval

<a id="resnet101__coyote9seq_54_framestf_api_eval"></a>
### resnet101       @ coyote_9_seq_54_frames/tf_api_eval

<a id="p1_source_coyote_0x682_1__resnet101coyote9seq_54_framestf_api_eval"></a>
#### p1_source_coyote_0x682_1       @ resnet101/coyote_9_seq_54_frames/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/p1_source_coyote_0x682_1 pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1.config sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="mp__p1_source_coyote_0x682_1resnet101coyote9seq_54_framestf_api_eval"></a>
##### mp       @ p1_source_coyote_0x682_1/resnet101/coyote_9_seq_54_frames/tf_api_eval

CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1_mp labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/p1_source_coyote_0x682_1 pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1.config sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="deer_11_seq_56_frames__tf_api_eval"></a>
## deer_11_seq_56_frames       @ tf_api_eval

<a id="resnet101__deer_11_seq_56_framestf_api_eval"></a>
### resnet101       @ deer_11_seq_56_frames/tf_api_eval

<a id="p1_source_deer_0x682_1__resnet101deer_11_seq_56_framestf_api_eval"></a>
#### p1_source_deer_0x682_1       @ resnet101/deer_11_seq_56_frames/tf_api_eval

CUDA_VISIBLE_DEVICES=1 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_deer_11_seq_56_frames_max_1_size_400_res_60_0x682_1 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/p1_source_deer_0x682_1 pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_deer_11_seq_56_frames_max_1_size_400_res_60_0x682_1.config sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="moose_10_seq_50_frames__tf_api_eval"></a>
## moose_10_seq_50_frames       @ tf_api_eval

<a id="resnet101__moose_10_seq_50_framestf_api_eval"></a>
### resnet101       @ moose_10_seq_50_frames/tf_api_eval

<a id="p1_source_moose_0x682_1__resnet101moose_10_seq_50_framestf_api_eval"></a>
#### p1_source_moose_0x682_1       @ resnet101/moose_10_seq_50_frames/tf_api_eval

CUDA_VISIBLE_DEVICES=2 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_moose_10_seq_50_frames_max_1_size_400_res_60_0x682_1 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/p1_source_moose_0x682_1 pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_moose_10_seq_50_frames_max_1_size_400_res_60_0x682_1.config sleep_time=10 save_video=0 save_det=0 enable_masks=1

<a id="visualize"></a>
# visualize

<a id="bear11__visualize"></a>
## bear_1_1       @ visualize

<a id="inception_resnet_v2_size_400_max1res_66__bear11visualize"></a>
### inception_resnet_v2_size_400_max_1_res_66       @ bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66_201625_grs_190404_193211 map_to_bbox=1 combine=1

<a id="0x682_10__inception_resnet_v2_size_400_max1res_66bear11visualize"></a>
#### 0x682_10       @ inception_resnet_v2_size_400_max_1_res_66/bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66_201625_0x682_10_grs_190404_221420 map_to_bbox=1 combine=1 border=10

<a id="only_blended__0x682_10inception_resnet_v2_size_400_max1res_66bear11visualize"></a>
##### only_blended       @ 0x682_10/inception_resnet_v2_size_400_max_1_res_66/bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/inception_resnet_v2_bear_1_1_100_size_400_max_1_res_66_201625_0x682_10_grs_190404_221420 map_to_bbox=1 combine=1 border=10 include_orig=0 include_binary=0 out_size=0x0 write_text=0

<a id="resnet101__bear11visualize"></a>
### resnet101       @ bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/resnet101_bear_1_1_100_323020_grs_190404_152641 map_to_bbox=1 combine=1 include_orig=0 include_binary=0

<a id="only_blended__resnet101bear11visualize"></a>
#### only_blended       @ resnet101/bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/resnet101_bear_1_1_100_323020_grs_190404_152641 map_to_bbox=1 combine=1 include_orig=0 include_binary=0 out_size=0x0 write_text=0

<a id="0x682_10__only_blendedresnet101bear11visualize"></a>
##### 0x682_10       @ only_blended/resnet101/bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/resnet101_bear_1_1_100_323020_on_0x682_10_grs_190405_071408 map_to_bbox=1 combine=1 include_orig=0 include_binary=0 out_size=0x0 border=10 write_text=0

<a id="resnet101_max1size_400_res_60__bear11visualize"></a>
### resnet101_max_1_size_400_res_60       @ bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results\mask_rcnn\resnet101_bear_1_1_100_max_1_size_400_res_60_tf_api_eval_ckpt_226767__z370_190403_090952 map_to_bbox=1

<a id="only_blended__resnet101_max1size_400_res_60bear11visualize"></a>
#### only_blended       @ resnet101_max_1_size_400_res_60/bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results\mask_rcnn\resnet101_bear_1_1_100_max_1_size_400_res_60_tf_api_eval_ckpt_226767__z370_190403_090952 map_to_bbox=1 combine=1 include_orig=0 include_binary=0 out_size=0x0 write_text=0

<a id="0x682_10__only_blendedresnet101_max1size_400_res_60bear11visualize"></a>
##### 0x682_10       @ only_blended/resnet101_max_1_size_400_res_60/bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results\mask_rcnn\resnet101_bear_1_1_100_max_1_size_400_res_60_226767_on_0x682_10_z370_190405_111328 map_to_bbox=1 combine=1 include_orig=0 include_binary=0 out_size=0x0 write_text=0 border=10

<a id="bear14__resnet101_max1size_400_res_60bear11visualize"></a>
#### bear_1_4       @ resnet101_max_1_size_400_res_60/bear_1_1/visualize

python3 visualize_masks.py img_paths=bear_1_4 img_root_dir=/data/acamp/acamp20k/bear mask_paths=bear_1_4_masks_0x0_0_test_masks mask_root_dir=/results\mask_rcnn\resnet101_bear_1_1_100_max_1_size_400_res_60_tf_api_eval_ckpt_226767__z370_190403_090952 map_to_bbox=1

<a id="combine_seq__resnet101_max1size_400_res_60bear11visualize"></a>
#### combine_seq       @ resnet101_max_1_size_400_res_60/bear_1_1/visualize

python3 visualize_masks.py img_paths=bear_1_2_to_1_6_mapped img_root_dir=/data/acamp/acamp20k/masks mask_paths=resnet101_bear_1_1_100_max_1_size_400_res_60_226767_bear_1_2_to_1_6_0x0_0_test_masks_z370_190403_101700 mask_root_dir=/results/mask_rcnn map_to_bbox=1

<a id="resnet101_max1size_400_res_60_0x682_50__bear11visualize"></a>
### resnet101_max_1_size_400_res_60_0x682_50       @ bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results\mask_rcnn\resnet101_bear_1_1_100_max_1_size_400_res_60_0x682_50_304861_on_0x682_10_z370_190405_112240 map_to_bbox=1 combine=1 include_orig=0 include_binary=0 out_size=0x0 write_text=0 border=50

<a id="inception_v2_max1size_200_res_90__bear11visualize"></a>
### inception_v2_max_1_size_200_res_90       @ bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/inception_v2_bear_1_1_100_max_1_size_200_res_90_697997_grs_190404_155405 map_to_bbox=1 combine=1

<a id="only_blended__inception_v2_max1size_200_res_90bear11visualize"></a>
#### only_blended       @ inception_v2_max_1_size_200_res_90/bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/inception_v2_bear_1_1_100_max_1_size_200_res_90_697997_grs_190404_155405 map_to_bbox=1 combine=1 include_orig=0 include_binary=0 out_size=0x0 write_text=0

<a id="0x682_10__only_blendedinception_v2_max1size_200_res_90bear11visualize"></a>
##### 0x682_10       @ only_blended/inception_v2_max_1_size_200_res_90/bear_1_1/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_1_1_to_bear_1_6.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/inception_v2_bear_1_1_100_max_1_size_200_res_90_697997_on_0x682_10_grs_190405_114152 map_to_bbox=1 combine=1 include_orig=0 include_binary=0 out_size=0x0 write_text=0 border=10

<a id="bear_13_seq_103_frames__visualize"></a>
## bear_13_seq_103_frames       @ visualize

<a id="inception_resnet_v2_size_400_max1res_66__bear_13_seq_103_framesvisualize"></a>
### inception_resnet_v2_size_400_max_1_res_66       @ bear_13_seq_103_frames/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames_0x682_10_test.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results\mask_rcnn\inception_resnet_v2_bear_13_seq_103_frames_size_400_max_1_res_66_ckpt_233911_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_095713 combine=1

<a id="map_to_bbox__inception_resnet_v2_size_400_max1res_66bear_13_seq_103_framesvisualize"></a>
#### map_to_bbox       @ inception_resnet_v2_size_400_max_1_res_66/bear_13_seq_103_frames/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results\mask_rcnn\inception_resnet_v2_bear_13_seq_103_frames_size_400_max_1_res_66_ckpt_233911_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_095713 map_to_bbox=1 border=10 include_orig=1 include_binary=1 combine=1

<a id="resnet101_full_size__bear_13_seq_103_framesvisualize"></a>
### resnet101_full_size       @ bear_13_seq_103_frames/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/resnet101_bear_13_seq_103_frames_271399_z370_190408_225632 map_to_bbox=1 border=10 combine=1 include_orig=0 include_binary=0 out_size=0x0 write_text=0 

<a id="resnet101_max1size_400_res_60__bear_13_seq_103_framesvisualize"></a>
### resnet101_max_1_size_400_res_60       @ bear_13_seq_103_frames/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60_236449_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_060204 map_to_bbox=1 border=10 combine=1

<a id="only_blended__resnet101_max1size_400_res_60bear_13_seq_103_framesvisualize"></a>
#### only_blended       @ resnet101_max_1_size_400_res_60/bear_13_seq_103_frames/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results/mask_rcnn/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60_236449_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_060204 map_to_bbox=1 combine=1 include_orig=0 include_binary=0 out_size=0x0 write_text=0 border=10

<a id="inception_v2_max1size_200_res_90__resnet101_max1size_400_res_60bear_13_seq_103_framesvisualize"></a>
#### inception_v2_max_1_size_200_res_90       @ resnet101_max_1_size_400_res_60/bear_13_seq_103_frames/visualize

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/results\mask_rcnn\inception_v2_bear_13_seq_103_frames_max_1_size_200_res_90_220913_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_065417 combine=1 map_to_bbox=1 border=10

<a id="augment"></a>
# augment

<a id="resnet101_bear11_100__augment"></a>
## resnet101_bear_1_1_100       @ augment

<a id="bear11_0x00test_masks__resnet101_bear11_100augment"></a>
### bear_1_1_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_1_1 bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=1 start_frame_id=100 mask_paths=/results/mask_rcnn\resnet101_bear_1_1_100_323020_bear_1_1_0x0_0_test_grs_190305_094959

<a id="bear12_0x00test_masks__resnet101_bear11_100augment"></a>
### bear_1_2_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear_1_2 bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=1 start_frame_id=0 mask_root_dir=H:\UofA\Acamp\code\results\mask_rcnn\trained\mask_rcnn_resnet101_bear_1_1_100\tf_api_eval\ckpt_323020 mask_paths=bear_1_2_0x0_0_test_masks

<a id="bear13_0x00test_masks__resnet101_bear11_100augment"></a>
### bear_1_3_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear_1_3 bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=1 start_frame_id=0 mask_root_dir=H:\UofA\Acamp\code\results\mask_rcnn\trained\mask_rcnn_resnet101_bear_1_1_100\tf_api_eval\ckpt_323020 mask_paths=bear_1_3_0x0_0_test_masks

<a id="bear14_0x00test_masks__resnet101_bear11_100augment"></a>
### bear_1_4_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear_1_4 bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=1 start_frame_id=0 mask_root_dir=H:\UofA\Acamp\code\results\mask_rcnn\trained\mask_rcnn_resnet101_bear_1_1_100\tf_api_eval\ckpt_323020 mask_paths=bear_1_4_0x0_0_test_masks

<a id="bear15_0x00test_masks__resnet101_bear11_100augment"></a>
### bear_1_5_0x0_0_test_masks       @ resnet101_bear_1_1_100/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear_1_5 bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=1 start_frame_id=0 mask_root_dir=H:\UofA\Acamp\code\results\mask_rcnn\trained\mask_rcnn_resnet101_bear_1_1_100\tf_api_eval\ckpt_323020 mask_paths=bear_1_5_0x0_0_test_masks

<a id="resnet101_bear_13_seq_103_frames_max1size_400_res_60__augment"></a>
## resnet101_bear_13_seq_103_frames_max_1_size_400_res_60       @ augment

<a id="acamp_office__resnet101_bear_13_seq_103_frames_max1size_400_res_60augment"></a>
### acamp_office       @ resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

<a id="bear81__acamp_officeresnet101_bear_13_seq_103_frames_max1size_400_res_60augment"></a>
#### bear_8_1       @ acamp_office/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_8_1 bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=3 mask_paths=/results/mask_rcnn\resnet101_bear_13_seq_103_frames_max_1_size_400_res_60_236449_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_060204/bear_8_1_masks_0x682_10_test_masks map_to_bbox=1 border=10
qqq
<a id="bear81__acamp_officeresnet101_bear_13_seq_103_frames_max1size_400_res_60augment-1"></a>
#### bear_8_1       @ acamp_office/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_8_1 bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=2 mask_paths=/results/mask_rcnn\resnet101_bear_13_seq_103_frames_max_1_size_400_res_60_236449_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_060204/bear_8_1_masks_0x682_10_test_masks map_to_bbox=1 border=10

<a id="airport__resnet101_bear_13_seq_103_frames_max1size_400_res_60augment"></a>
### airport       @ resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

<a id="bear_10_1__airportresnet101_bear_13_seq_103_frames_max1size_400_res_60augment"></a>
#### bear_10_1       @ airport/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_10_1 bkg_paths=/data/acamp/acamp20k/backgrounds/airport  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=10 mask_paths=/results/mask_rcnn\resnet101_bear_13_seq_103_frames_max_1_size_400_res_60_236449_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_060204/bear_10_1_masks_0x682_10_test_masks map_to_bbox=1 border=10

<a id="bear_16_3__airportresnet101_bear_13_seq_103_frames_max1size_400_res_60augment"></a>
#### bear_16_3       @ airport/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_16_3 bkg_paths=/data/acamp/acamp20k/backgrounds/airport  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=10 mask_paths=/results/mask_rcnn\resnet101_bear_13_seq_103_frames_max_1_size_400_res_60_236449_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_060204/bear_16_3_masks_0x682_10_test_masks map_to_bbox=1 border=10


<a id="bear_jesse71_w__airportresnet101_bear_13_seq_103_frames_max1size_400_res_60augment"></a>
#### bear_jesse_7_1_w       @ airport/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_jesse_7_1_w bkg_paths=/data/acamp/acamp20k/backgrounds/airport  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=10 mask_paths=/results/mask_rcnn\resnet101_bear_13_seq_103_frames_max_1_size_400_res_60_236449_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_060204/bear_jesse_7_1_w_masks_0x682_10_test_masks map_to_bbox=1 border=10

<a id="highway__resnet101_bear_13_seq_103_frames_max1size_400_res_60augment"></a>
### highway       @ resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

<a id="bear_jesse_231w__highwayresnet101_bear_13_seq_103_frames_max1size_400_res_60augment"></a>
#### bear_jesse_23_1_w       @ highway/resnet101_bear_13_seq_103_frames_max_1_size_400_res_60/augment

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_jesse_23_1_w bkg_paths=/data/acamp/acamp20k/backgrounds/highway  only_one_src_obj=1 aug_seq_size=1000 visualize=1  bkg_size=1280x720 static_bkg=10 mask_paths=/results/mask_rcnn\resnet101_bear_13_seq_103_frames_max_1_size_400_res_60_236449_on_bear_13_seq_103_frames_masks_0x682_10_test_grs_190315_060204/bear_jesse_23_1_w_masks_0x682_10_test_masks map_to_bbox=1 border=10

