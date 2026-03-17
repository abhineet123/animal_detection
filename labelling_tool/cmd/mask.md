<!-- MarkdownTOC -->

- [to_mask_seq](#to_mask_seq_)
    - [bear_1_1       @ to_mask_seq](#bear_1_1___to_mask_se_q_)
        - [0x0_0       @ bear_1_1/to_mask_seq](#0x0_0___bear_1_1_to_mask_seq_)
        - [0x1365_10       @ bear_1_1/to_mask_seq](#0x1365_10___bear_1_1_to_mask_seq_)
        - [0x682_50       @ bear_1_1/to_mask_seq](#0x682_50___bear_1_1_to_mask_seq_)
        - [224x224_0       @ bear_1_1/to_mask_seq](#224x224_0___bear_1_1_to_mask_seq_)
        - [224x224_5       @ bear_1_1/to_mask_seq](#224x224_5___bear_1_1_to_mask_seq_)
        - [224x224_10       @ bear_1_1/to_mask_seq](#224x224_10___bear_1_1_to_mask_seq_)
        - [448_448_0       @ bear_1_1/to_mask_seq](#448_448_0___bear_1_1_to_mask_seq_)
        - [448_448_10       @ bear_1_1/to_mask_seq](#448_448_10___bear_1_1_to_mask_seq_)
            - [448_448_25       @ 448_448_10/bear_1_1/to_mask_seq](#448_448_25___448_448_10_bear_1_1_to_mask_se_q_)
            - [fixed_ar       @ 448_448_10/bear_1_1/to_mask_seq](#fixed_ar___448_448_10_bear_1_1_to_mask_se_q_)
        - [500x500_10       @ bear_1_1/to_mask_seq](#500x500_10___bear_1_1_to_mask_seq_)
    - [bear_1_1_to_bear_1_6       @ to_mask_seq](#bear_1_1_to_bear_1_6___to_mask_se_q_)
        - [0x0_0       @ bear_1_1_to_bear_1_6/to_mask_seq](#0x0_0___bear_1_1_to_bear_1_6_to_mask_seq_)
            - [combine_seq       @ 0x0_0/bear_1_1_to_bear_1_6/to_mask_seq](#combine_seq___0x0_0_bear_1_1_to_bear_1_6_to_mask_seq_)
                - [map_to_bbox       @ combine_seq/0x0_0/bear_1_1_to_bear_1_6/to_mask_seq](#map_to_bbox___combine_seq_0x0_0_bear_1_1_to_bear_1_6_to_mask_seq_)
        - [224x224_0_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#224x224_0_test___bear_1_1_to_bear_1_6_to_mask_seq_)
        - [0x682_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#0x682_10_test___bear_1_1_to_bear_1_6_to_mask_seq_)
        - [0x682_50_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#0x682_50_test___bear_1_1_to_bear_1_6_to_mask_seq_)
        - [448x448_ar_1p0_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#448x448_ar_1p0_test___bear_1_1_to_bear_1_6_to_mask_seq_)
        - [448x448_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#448x448_10_test___bear_1_1_to_bear_1_6_to_mask_seq_)
            - [map_to_bbox       @ 448x448_10_test/bear_1_1_to_bear_1_6/to_mask_seq](#map_to_bbox___448x448_10_test_bear_1_1_to_bear_1_6_to_mask_seq_)
        - [500x500_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#500x500_10_test___bear_1_1_to_bear_1_6_to_mask_seq_)
    - [bear_1_2_to_1_6       @ to_mask_seq](#bear_1_2_to_1_6___to_mask_se_q_)
        - [448x448_25_test       @ bear_1_2_to_1_6/to_mask_seq](#448x448_25_test___bear_1_2_to_1_6_to_mask_se_q_)
        - [combine_seq       @ bear_1_2_to_1_6/to_mask_seq](#combine_seq___bear_1_2_to_1_6_to_mask_se_q_)
        - [448x448_0_test       @ bear_1_2_to_1_6/to_mask_seq](#448x448_0_test___bear_1_2_to_1_6_to_mask_se_q_)
            - [fixed_ar       @ 448x448_0_test/bear_1_2_to_1_6/to_mask_seq](#fixed_ar___448x448_0_test_bear_1_2_to_1_6_to_mask_seq_)
    - [bear_8_2       @ to_mask_seq](#bear_8_2___to_mask_se_q_)
        - [400x0_0       @ bear_8_2/to_mask_seq](#400x0_0___bear_8_2_to_mask_seq_)
    - [bear_10_1       @ to_mask_seq](#bear_10_1___to_mask_se_q_)
    - [bear_13_seq_103_frames       @ to_mask_seq](#bear_13_seq_103_frames___to_mask_se_q_)
        - [map_to_bbox       @ bear_13_seq_103_frames/to_mask_seq](#map_to_bbox___bear_13_seq_103_frames_to_mask_seq_)
        - [800x800_25       @ bear_13_seq_103_frames/to_mask_seq](#800x800_25___bear_13_seq_103_frames_to_mask_seq_)
        - [0x682_0       @ bear_13_seq_103_frames/to_mask_seq](#0x682_0___bear_13_seq_103_frames_to_mask_seq_)
        - [0x682_1       @ bear_13_seq_103_frames/to_mask_seq](#0x682_1___bear_13_seq_103_frames_to_mask_seq_)
        - [0x682_10       @ bear_13_seq_103_frames/to_mask_seq](#0x682_10___bear_13_seq_103_frames_to_mask_seq_)
        - [0x682_2       @ bear_13_seq_103_frames/to_mask_seq](#0x682_2___bear_13_seq_103_frames_to_mask_seq_)
            - [test       @ 0x682_2/bear_13_seq_103_frames/to_mask_seq](#test___0x682_2_bear_13_seq_103_frames_to_mask_seq_)
                - [bear_1_1       @ test/0x682_2/bear_13_seq_103_frames/to_mask_seq](#bear_1_1___test_0x682_2_bear_13_seq_103_frames_to_mask_se_q_)
            - [bear_13_seq_103_frames_0x682_10_test       @ 0x682_2/bear_13_seq_103_frames/to_mask_seq](#bear_13_seq_103_frames_0x682_10_test___0x682_2_bear_13_seq_103_frames_to_mask_seq_)
        - [224x224_5       @ bear_13_seq_103_frames/to_mask_seq](#224x224_5___bear_13_seq_103_frames_to_mask_seq_)
            - [combine_seq       @ 224x224_5/bear_13_seq_103_frames/to_mask_seq](#combine_seq___224x224_5_bear_13_seq_103_frames_to_mask_seq_)
        - [224x224_5_test       @ bear_13_seq_103_frames/to_mask_seq](#224x224_5_test___bear_13_seq_103_frames_to_mask_seq_)
            - [combine_seq       @ 224x224_5_test/bear_13_seq_103_frames/to_mask_seq](#combine_seq___224x224_5_test_bear_13_seq_103_frames_to_mask_se_q_)
            - [bear_1_1_even_10       @ 224x224_5_test/bear_13_seq_103_frames/to_mask_seq](#bear_1_1_even_10___224x224_5_test_bear_13_seq_103_frames_to_mask_se_q_)
        - [bear_13_seq_103_frames_448x448_10_test       @ bear_13_seq_103_frames/to_mask_seq](#bear_13_seq_103_frames_448x448_10_test___bear_13_seq_103_frames_to_mask_seq_)
        - [bear_13_seq_103_frames_800x800_25_test       @ bear_13_seq_103_frames/to_mask_seq](#bear_13_seq_103_frames_800x800_25_test___bear_13_seq_103_frames_to_mask_seq_)
            - [map_to_bbox       @ bear_13_seq_103_frames_800x800_25_test/bear_13_seq_103_frames/to_mask_seq](#map_to_bbox___bear_13_seq_103_frames_800x800_25_test_bear_13_seq_103_frames_to_mask_se_q_)
    - [bear_13_seq_103_frames_ex1       @ to_mask_seq](#bear_13_seq_103_frames_ex1___to_mask_se_q_)
    - [coyote_9_seq_54_frames       @ to_mask_seq](#coyote_9_seq_54_frames___to_mask_se_q_)
        - [map_to_bbox       @ coyote_9_seq_54_frames/to_mask_seq](#map_to_bbox___coyote_9_seq_54_frames_to_mask_seq_)
        - [0x682_1       @ coyote_9_seq_54_frames/to_mask_seq](#0x682_1___coyote_9_seq_54_frames_to_mask_seq_)
    - [deer_11_seq_56_frames       @ to_mask_seq](#deer_11_seq_56_frames___to_mask_se_q_)
        - [map_to_bbox       @ deer_11_seq_56_frames/to_mask_seq](#map_to_bbox___deer_11_seq_56_frames_to_mask_se_q_)
            - [deer_12_1       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq](#deer_12_1___map_to_bbox_deer_11_seq_56_frames_to_mask_se_q_)
            - [deer_jesse_2_1       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq](#deer_jesse_2_1___map_to_bbox_deer_11_seq_56_frames_to_mask_se_q_)
            - [deer_jesse_2_2       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq](#deer_jesse_2_2___map_to_bbox_deer_11_seq_56_frames_to_mask_se_q_)
        - [0x682_1       @ deer_11_seq_56_frames/to_mask_seq](#0x682_1___deer_11_seq_56_frames_to_mask_se_q_)
    - [moose_10_seq_50_frames       @ to_mask_seq](#moose_10_seq_50_frames___to_mask_se_q_)
        - [map_to_bbox       @ moose_10_seq_50_frames/to_mask_seq](#map_to_bbox___moose_10_seq_50_frames_to_mask_seq_)
        - [0x682_1       @ moose_10_seq_50_frames/to_mask_seq](#0x682_1___moose_10_seq_50_frames_to_mask_seq_)
    - [p1_deer       @ to_mask_seq](#p1_deer___to_mask_se_q_)
        - [0x0_0       @ p1_deer/to_mask_seq](#0x0_0___p1_deer_to_mask_se_q_)
            - [acamp_all_deer       @ 0x0_0/p1_deer/to_mask_seq](#acamp_all_deer___0x0_0_p1_deer_to_mask_se_q_)
        - [map_to_bbox       @ p1_deer/to_mask_seq](#map_to_bbox___p1_deer_to_mask_se_q_)
    - [p1_coyote       @ to_mask_seq](#p1_coyote___to_mask_se_q_)
        - [0x0_0       @ p1_coyote/to_mask_seq](#0x0_0___p1_coyote_to_mask_se_q_)
            - [coyote_9_1       @ 0x0_0/p1_coyote/to_mask_seq](#coyote_9_1___0x0_0_p1_coyote_to_mask_se_q_)
            - [coyote_10_5       @ 0x0_0/p1_coyote/to_mask_seq](#coyote_10_5___0x0_0_p1_coyote_to_mask_se_q_)
            - [coyote_b       @ 0x0_0/p1_coyote/to_mask_seq](#coyote_b___0x0_0_p1_coyote_to_mask_se_q_)
        - [map_to_bbox       @ p1_coyote/to_mask_seq](#map_to_bbox___p1_coyote_to_mask_se_q_)
    - [p1_moose       @ to_mask_seq](#p1_moose___to_mask_se_q_)
        - [0x0_0       @ p1_moose/to_mask_seq](#0x0_0___p1_moose_to_mask_seq_)
            - [moose_5_2       @ 0x0_0/p1_moose/to_mask_seq](#moose_5_2___0x0_0_p1_moose_to_mask_seq_)
        - [map_to_bbox       @ p1_moose/to_mask_seq](#map_to_bbox___p1_moose_to_mask_seq_)
    - [p1_bear       @ to_mask_seq](#p1_bear___to_mask_se_q_)
        - [0x0_0       @ p1_bear/to_mask_seq](#0x0_0___p1_bear_to_mask_se_q_)
            - [bear_1_1       @ 0x0_0/p1_bear/to_mask_seq](#bear_1_1___0x0_0_p1_bear_to_mask_se_q_)
        - [map_to_bbox       @ p1_bear/to_mask_seq](#map_to_bbox___p1_bear_to_mask_se_q_)
    - [p1_source_0x682_1       @ to_mask_seq](#p1_source_0x682_1___to_mask_se_q_)
- [mask_to_record](#mask_to_recor_d_)
    - [bear_1_1       @ mask_to_record](#bear_1_1___mask_to_record_)
        - [0x682_50       @ bear_1_1/mask_to_record](#0x682_50___bear_1_1_mask_to_recor_d_)
    - [bear_13_seq_103_frames       @ mask_to_record](#bear_13_seq_103_frames___mask_to_record_)
        - [0x682_0       @ bear_13_seq_103_frames/mask_to_record](#0x682_0___bear_13_seq_103_frames_mask_to_recor_d_)
        - [0x682_10       @ bear_13_seq_103_frames/mask_to_record](#0x682_10___bear_13_seq_103_frames_mask_to_recor_d_)
    - [bear_13_seq_103_frames_ex1       @ mask_to_record](#bear_13_seq_103_frames_ex1___mask_to_record_)
        - [0x682_1       @ bear_13_seq_103_frames_ex1/mask_to_record](#0x682_1___bear_13_seq_103_frames_ex1_mask_to_recor_d_)
    - [coyote_9_seq_54_frames       @ mask_to_record](#coyote_9_seq_54_frames___mask_to_record_)
        - [0x682_1       @ coyote_9_seq_54_frames/mask_to_record](#0x682_1___coyote_9_seq_54_frames_mask_to_recor_d_)
    - [deer_11_seq_56_frames       @ mask_to_record](#deer_11_seq_56_frames___mask_to_record_)
        - [0x682_1       @ deer_11_seq_56_frames/mask_to_record](#0x682_1___deer_11_seq_56_frames_mask_to_record_)
    - [moose_10_seq_50_frames       @ mask_to_record](#moose_10_seq_50_frames___mask_to_record_)
        - [0x682_1       @ moose_10_seq_50_frames/mask_to_record](#0x682_1___moose_10_seq_50_frames_mask_to_recor_d_)
- [augment_mask](#augment_mas_k_)
    - [beer_1_1       @ augment_mask](#beer_1_1___augment_mask_)
        - [airport       @ beer_1_1/augment_mask](#airport___beer_1_1_augment_mas_k_)
        - [acamp_office       @ beer_1_1/augment_mask](#acamp_office___beer_1_1_augment_mas_k_)
            - [0x0_0       @ acamp_office/beer_1_1/augment_mask](#0x0_0___acamp_office_beer_1_1_augment_mask_)
            - [0x0_50       @ acamp_office/beer_1_1/augment_mask](#0x0_50___acamp_office_beer_1_1_augment_mask_)
        - [general       @ beer_1_1/augment_mask](#general___beer_1_1_augment_mas_k_)
            - [5_boxes_per_bkg       @ general/beer_1_1/augment_mask](#5_boxes_per_bkg___general_beer_1_1_augment_mas_k_)
    - [bear_jesse_7_1_w       @ augment_mask](#bear_jesse_7_1_w___augment_mask_)
        - [airport       @ bear_jesse_7_1_w/augment_mask](#airport___bear_jesse_7_1_w_augment_mas_k_)
    - [p1_deer       @ augment_mask](#p1_deer___augment_mask_)
        - [highway       @ p1_deer/augment_mask](#highway___p1_deer_augment_mask_)
        - [airport       @ p1_deer/augment_mask](#airport___p1_deer_augment_mask_)
    - [p1_coyote       @ augment_mask](#p1_coyote___augment_mask_)
        - [highway       @ p1_coyote/augment_mask](#highway___p1_coyote_augment_mask_)
        - [airport       @ p1_coyote/augment_mask](#airport___p1_coyote_augment_mask_)
    - [p1_moose       @ augment_mask](#p1_moose___augment_mask_)
        - [highway       @ p1_moose/augment_mask](#highway___p1_moose_augment_mas_k_)
        - [airport       @ p1_moose/augment_mask](#airport___p1_moose_augment_mas_k_)
    - [p1_bear       @ augment_mask](#p1_bear___augment_mask_)
        - [highway       @ p1_bear/augment_mask](#highway___p1_bear_augment_mask_)
        - [airport       @ p1_bear/augment_mask](#airport___p1_bear_augment_mask_)
- [visualize_masks](#visualize_masks_)
    - [animal       @ visualize_masks](#animal___visualize_mask_s_)
        - [bear_1_1       @ animal/visualize_masks](#bear_1_1___animal_visualize_masks_)
        - [bear_13_seq_103_frames       @ animal/visualize_masks](#bear_13_seq_103_frames___animal_visualize_masks_)
        - [bear_13_seq_103_frames_ex1       @ animal/visualize_masks](#bear_13_seq_103_frames_ex1___animal_visualize_masks_)
            - [0x682_10       @ bear_13_seq_103_frames_ex1/animal/visualize_masks](#0x682_10___bear_13_seq_103_frames_ex1_animal_visualize_mask_s_)
            - [0x682_0       @ bear_13_seq_103_frames_ex1/animal/visualize_masks](#0x682_0___bear_13_seq_103_frames_ex1_animal_visualize_mask_s_)
    - [reorg_roi       @ visualize_masks](#reorg_roi___visualize_mask_s_)

<!-- /MarkdownTOC -->

<a id="to_mask_seq_"></a>
# to_mask_seq

<a id="bear_1_1___to_mask_se_q_"></a>
## bear_1_1       @ to_mask_seq-->mask

<a id="0x0_0___bear_1_1_to_mask_seq_"></a>
### 0x0_0       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x0 out_border=0 show_img=0

<a id="0x1365_10___bear_1_1_to_mask_seq_"></a>
### 0x1365_10       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x1365 out_border=10 show_img=0
<a id="0x682_50___bear_1_1_to_mask_seq_"></a>
### 0x682_50       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=50 show_img=0

<a id="224x224_0___bear_1_1_to_mask_seq_"></a>
### 224x224_0       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=224x224 out_border=5 show_img=0 save_test=0

<a id="224x224_5___bear_1_1_to_mask_seq_"></a>
### 224x224_5       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=224x224 out_border=5 show_img=0

<a id="224x224_10___bear_1_1_to_mask_seq_"></a>
### 224x224_10       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=224x224 out_border=10 show_img=0

<a id="448_448_0___bear_1_1_to_mask_seq_"></a>
### 448_448_0       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=448x448 out_border=0 show_img=0

<a id="448_448_10___bear_1_1_to_mask_seq_"></a>
### 448_448_10       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=10 show_img=0 save_raw_mask=1

<a id="448_448_25___448_448_10_bear_1_1_to_mask_se_q_"></a>
#### 448_448_25       @ 448_448_10/bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=448x448 out_border=25 show_img=0

<a id="fixed_ar___448_448_10_bear_1_1_to_mask_se_q_"></a>
#### fixed_ar       @ 448_448_10/bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=0 show_img=0 fixed_ar=1 save_raw_mask=1

<a id="500x500_10___bear_1_1_to_mask_seq_"></a>
### 500x500_10       @ bear_1_1/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=500x500 out_border=10 show_img=0 save_raw_mask=1

<a id="bear_1_1_to_bear_1_6___to_mask_se_q_"></a>
## bear_1_1_to_bear_1_6       @ to_mask_seq-->mask

<a id="0x0_0___bear_1_1_to_bear_1_6_to_mask_seq_"></a>
### 0x0_0       @ bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x0 out_border=0 show_img=0

<a id="combine_seq___0x0_0_bear_1_1_to_bear_1_6_to_mask_seq_"></a>
#### combine_seq       @ 0x0_0/bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k save_train=0 out_mask_size=0x0 out_border=0 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_0x0_0_test combine_seq=1

<a id="map_to_bbox___combine_seq_0x0_0_bear_1_1_to_bear_1_6_to_mask_seq_"></a>
##### map_to_bbox       @ combine_seq/0x0_0/bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear save_train=0 out_mask_size=0x0 out_border=0 show_img=1 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_mapped combine_seq=1 map_to_bbox=1

<a id="224x224_0_test___bear_1_1_to_bear_1_6_to_mask_seq_"></a>
### 224x224_0_test       @ bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=224x224 out_border=0 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_224x224_0_test save_train=0

<a id="0x682_10_test___bear_1_1_to_bear_1_6_to_mask_seq_"></a>
### 0x682_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_10_test save_train=0

<a id="0x682_50_test___bear_1_1_to_bear_1_6_to_mask_seq_"></a>
### 0x682_50_test       @ bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=50 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_50_test save_train=0

<a id="448x448_ar_1p0_test___bear_1_1_to_bear_1_6_to_mask_seq_"></a>
### 448x448_ar_1p0_test       @ bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k save_train=0 out_mask_size=448x448 out_border=0 fixed_ar=1 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_448x448_ar_1p0_test combine_seq=1

<a id="448x448_10_test___bear_1_1_to_bear_1_6_to_mask_seq_"></a>
### 448x448_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k save_train=0 out_mask_size=448x448 out_border=10 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_448x448_10_test combine_seq=1

<a id="map_to_bbox___448x448_10_test_bear_1_1_to_bear_1_6_to_mask_seq_"></a>
#### map_to_bbox       @ 448x448_10_test/bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear save_train=0 out_mask_size=448x448 out_border=10 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_448x448_10_mapped_test combine_seq=1 map_to_bbox=1

<a id="500x500_10_test___bear_1_1_to_bear_1_6_to_mask_seq_"></a>
### 500x500_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k save_train=0 out_mask_size=500x500 out_border=10 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_500x500_10_test/images combine_seq=1

<a id="bear_1_2_to_1_6___to_mask_se_q_"></a>
## bear_1_2_to_1_6       @ to_mask_seq-->mask

<a id="448x448_25_test___bear_1_2_to_1_6_to_mask_se_q_"></a>
### 448x448_25_test       @ bear_1_2_to_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=25 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_2_to_1_6_448x448_25_test save_train=0 save_test=1

<a id="combine_seq___bear_1_2_to_1_6_to_mask_se_q_"></a>
### combine_seq       @ bear_1_2_to_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=448x448 out_border=25 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_2_to_1_6_448x448_25_test save_train=0 save_test=1 combine_seq=1

<a id="448x448_0_test___bear_1_2_to_1_6_to_mask_se_q_"></a>
### 448x448_0_test       @ bear_1_2_to_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=0 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_2_to_1_6_448x448_0_test save_train=0 save_test=1

<a id="fixed_ar___448x448_0_test_bear_1_2_to_1_6_to_mask_seq_"></a>
#### fixed_ar       @ 448x448_0_test/bear_1_2_to_1_6/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=0 fixed_ar=1 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_2_to_1_6_448x448_ar_1p0_test save_train=0 save_test=1 combine_seq=1 

<a id="bear_8_2___to_mask_se_q_"></a>
## bear_8_2       @ to_mask_seq-->mask

<a id="400x0_0___bear_8_2_to_mask_seq_"></a>
### 400x0_0       @ bear_8_2/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_8_2 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=400x0 out_border=0 show_img=0 save_test=0 save_raw_mask=0

<a id="bear_10_1___to_mask_se_q_"></a>
## bear_10_1       @ to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_10_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=400x0 out_border=0 show_img=0 save_test=0 save_raw_mask=0

<a id="bear_13_seq_103_frames___to_mask_se_q_"></a>
## bear_13_seq_103_frames       @ to_mask_seq-->mask

<a id="map_to_bbox___bear_13_seq_103_frames_to_mask_seq_"></a>
### map_to_bbox       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=800x800 out_border=25 show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1


<a id="800x800_25___bear_13_seq_103_frames_to_mask_seq_"></a>
### 800x800_25       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=800x800 out_border=25 show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_800x800_25 combine_seq=1

<a id="0x682_0___bear_13_seq_103_frames_to_mask_seq_"></a>
### 0x682_0       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=0x682 out_border=0 show_img=0 save_test=0 save_raw_mask=0

<a id="0x682_1___bear_13_seq_103_frames_to_mask_seq_"></a>
### 0x682_1       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=1 show_img=0 save_test=0 save_raw_mask=0 allow_skipping_images=1

<a id="0x682_10___bear_13_seq_103_frames_to_mask_seq_"></a>
### 0x682_10       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 save_test=0 save_raw_mask=0

<a id="0x682_2___bear_13_seq_103_frames_to_mask_seq_"></a>
### 0x682_2       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=2 show_img=0 save_test=0 save_raw_mask=0

<a id="test___0x682_2_bear_13_seq_103_frames_to_mask_seq_"></a>
#### test       @ 0x682_2/bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 save_test=1 save_raw_mask=0

<a id="bear_1_1___test_0x682_2_bear_13_seq_103_frames_to_mask_se_q_"></a>
##### bear_1_1       @ test/0x682_2/bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=0x682 out_border=10 show_img=0 save_test=1 save_raw_mask=0

<a id="bear_13_seq_103_frames_0x682_10_test___0x682_2_bear_13_seq_103_frames_to_mask_seq_"></a>
#### bear_13_seq_103_frames_0x682_10_test       @ 0x682_2/bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 save_test=1  save_train=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_0x682_10_test combine_seq=0


<a id="224x224_5___bear_13_seq_103_frames_to_mask_seq_"></a>
### 224x224_5       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=0 save_raw_mask=0

<a id="combine_seq___224x224_5_bear_13_seq_103_frames_to_mask_seq_"></a>
#### combine_seq       @ 224x224_5/bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5 combine_seq=1



<a id="224x224_5_test___bear_13_seq_103_frames_to_mask_seq_"></a>
### 224x224_5_test       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=1 save_train=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5_test

<a id="combine_seq___224x224_5_test_bear_13_seq_103_frames_to_mask_se_q_"></a>
#### combine_seq       @ 224x224_5_test/bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=1 save_train=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5_test combine_seq=1

<a id="bear_1_1_even_10___224x224_5_test_bear_13_seq_103_frames_to_mask_se_q_"></a>
#### bear_1_1_even_10       @ 224x224_5_test/bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1_even_10 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=1 save_train=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5_test

<a id="bear_13_seq_103_frames_448x448_10_test___bear_13_seq_103_frames_to_mask_seq_"></a>
### bear_13_seq_103_frames_448x448_10_test       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=10 show_img=0 save_test=1 save_train=0 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_448x448_10_test combine_seq=1
<a id="bear_13_seq_103_frames_800x800_25_test___bear_13_seq_103_frames_to_mask_seq_"></a>
### bear_13_seq_103_frames_800x800_25_test       @ bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=800x800 out_border=25 show_img=0 save_test=1 save_train=0 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_800x800_25_test combine_seq=1

<a id="map_to_bbox___bear_13_seq_103_frames_800x800_25_test_bear_13_seq_103_frames_to_mask_se_q_"></a>
#### map_to_bbox       @ bear_13_seq_103_frames_800x800_25_test/bear_13_seq_103_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=800x800 out_border=25 show_img=0 save_test=1 save_train=0 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_800x800_25_mapped_test combine_seq=1 map_to_bbox=1

<a id="bear_13_seq_103_frames_ex1___to_mask_se_q_"></a>
## bear_13_seq_103_frames_ex1       @ to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames_ex1.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear_13_seq_103_frames_ex1 out_mask_size=0x682 out_border=10 show_img=0 save_test=0 save_raw_mask=0

<a id="coyote_9_seq_54_frames___to_mask_se_q_"></a>
## coyote_9_seq_54_frames       @ to_mask_seq-->mask

<a id="map_to_bbox___coyote_9_seq_54_frames_to_mask_seq_"></a>
### map_to_bbox       @ coyote_9_seq_54_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/coyote_9_seq_54_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/coyote_9_seq_54_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1

<a id="0x682_1___coyote_9_seq_54_frames_to_mask_seq_"></a>
### 0x682_1       @ coyote_9_seq_54_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/coyote_9_seq_54_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote_9_seq_54_frames out_mask_size=0x682 out_border=1 show_img=0 save_test=0 save_raw_mask=0 allow_skipping_images=0

<a id="deer_11_seq_56_frames___to_mask_se_q_"></a>
## deer_11_seq_56_frames       @ to_mask_seq-->mask

<a id="map_to_bbox___deer_11_seq_56_frames_to_mask_se_q_"></a>
### map_to_bbox       @ deer_11_seq_56_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/deer_11_seq_56_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/deer_11_seq_56_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1

<a id="deer_12_1___map_to_bbox_deer_11_seq_56_frames_to_mask_se_q_"></a>
#### deer_12_1       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=deer_12_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/deer_11_seq_56_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1

<a id="deer_jesse_2_1___map_to_bbox_deer_11_seq_56_frames_to_mask_se_q_"></a>
#### deer_jesse_2_1       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=deer_jesse_2_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/deer_11_seq_56_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1

<a id="deer_jesse_2_2___map_to_bbox_deer_11_seq_56_frames_to_mask_se_q_"></a>
#### deer_jesse_2_2       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=deer_jesse_2_2 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/deer_11_seq_56_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1


<a id="0x682_1___deer_11_seq_56_frames_to_mask_se_q_"></a>
### 0x682_1       @ deer_11_seq_56_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/deer_11_seq_56_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/deer_11_seq_56_frames out_mask_size=0x682 out_border=1 show_img=0 save_test=0 save_raw_mask=0 allow_skipping_images=0

<a id="moose_10_seq_50_frames___to_mask_se_q_"></a>
## moose_10_seq_50_frames       @ to_mask_seq-->mask

<a id="map_to_bbox___moose_10_seq_50_frames_to_mask_seq_"></a>
### map_to_bbox       @ moose_10_seq_50_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/moose_10_seq_50_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/moose_10_seq_50_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1


<a id="0x682_1___moose_10_seq_50_frames_to_mask_seq_"></a>
### 0x682_1       @ moose_10_seq_50_frames/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=../tf_api/moose_10_seq_50_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/moose_10_seq_50_frames out_mask_size=0x682 out_border=1 show_img=0 save_test=0 save_raw_mask=0 allow_skipping_images=0


<a id="p1_deer___to_mask_se_q_"></a>
## p1_deer       @ to_mask_seq-->mask

<a id="0x0_0___p1_deer_to_mask_se_q_"></a>
### 0x0_0       @ p1_deer/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_deer.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/deer out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0

<a id="acamp_all_deer___0x0_0_p1_deer_to_mask_se_q_"></a>
#### acamp_all_deer       @ 0x0_0/p1_deer/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_deer.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/deer out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0

<a id="map_to_bbox___p1_deer_to_mask_se_q_"></a>
### map_to_bbox       @ p1_deer/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_deer.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1

<a id="p1_coyote___to_mask_se_q_"></a>
## p1_coyote       @ to_mask_seq-->mask

<a id="0x0_0___p1_coyote_to_mask_se_q_"></a>
### 0x0_0       @ p1_coyote/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_coyote.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="coyote_9_1___0x0_0_p1_coyote_to_mask_se_q_"></a>
#### coyote_9_1       @ 0x0_0/p1_coyote/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=coyote_9_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="coyote_10_5___0x0_0_p1_coyote_to_mask_se_q_"></a>
#### coyote_10_5       @ 0x0_0/p1_coyote/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=coyote_10_5 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="coyote_b___0x0_0_p1_coyote_to_mask_se_q_"></a>
#### coyote_b       @ 0x0_0/p1_coyote/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=coyote_b class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="map_to_bbox___p1_coyote_to_mask_se_q_"></a>
### map_to_bbox       @ p1_coyote/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_coyote.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1

<a id="p1_moose___to_mask_se_q_"></a>
## p1_moose       @ to_mask_seq-->mask

<a id="0x0_0___p1_moose_to_mask_seq_"></a>
### 0x0_0       @ p1_moose/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_moose.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/moose out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="moose_5_2___0x0_0_p1_moose_to_mask_seq_"></a>
#### moose_5_2       @ 0x0_0/p1_moose/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=moose_5_2 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/acamp20k_annotations out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="map_to_bbox___p1_moose_to_mask_seq_"></a>
### map_to_bbox       @ p1_moose/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_moose.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1

<a id="p1_bear___to_mask_se_q_"></a>
## p1_bear       @ to_mask_seq-->mask

<a id="0x0_0___p1_bear_to_mask_se_q_"></a>
### 0x0_0       @ p1_bear/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_bear.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 masks_per_seq=1 

<a id="bear_1_1___0x0_0_p1_bear_to_mask_se_q_"></a>
#### bear_1_1       @ 0x0_0/p1_bear/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 masks_per_seq=1 

<a id="map_to_bbox___p1_bear_to_mask_se_q_"></a>
### map_to_bbox       @ p1_bear/to_mask_seq-->mask

python3 to_mask_seq.py seq_paths=p1_bear.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1

<a id="p1_source_0x682_1___to_mask_se_q_"></a>
## p1_source_0x682_1       @ to_mask_seq-->mask

python3 to_mask_seq.py class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/prototype_1_source out_mask_size=0x682 out_border=1 show_img=1 save_test=0 save_raw_mask=0 map_to_bbox=0 out_img_dir=__n__



<a id="mask_to_recor_d_"></a>
# mask_to_record

<a id="bear_1_1___mask_to_record_"></a>
## bear_1_1       @ mask_to_record-->mask

python2 mask_to_record.py --root_dir=/data/acamp/acamp20k --seq_paths=bear_1_1_masks_0x1365_10 --output_path=/data/acamp/acamp20k/bear_1_1_masks_0x1365_10.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="0x682_50___bear_1_1_mask_to_recor_d_"></a>
### 0x682_50       @ bear_1_1/mask_to_record-->mask

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/masks --seq_paths=bear_1_1_0x682_50 --output_path=/data/acamp/acamp20k/masks/bear_1_1_0x682_50.record --label_map_path=data/wildlife_label_map_mask.pbtxt


<a id="bear_13_seq_103_frames___mask_to_record_"></a>
## bear_13_seq_103_frames       @ mask_to_record-->mask

<a id="0x682_0___bear_13_seq_103_frames_mask_to_recor_d_"></a>
### 0x682_0       @ bear_13_seq_103_frames/mask_to_record-->mask

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k --seq_paths=bear_13_seq_103_frames.txt --seq_postfix=masks_0x682_0 --output_path=/data/acamp/acamp20k/bear_13_seq_103_frames_0x682_0.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="0x682_10___bear_13_seq_103_frames_mask_to_recor_d_"></a>
### 0x682_10       @ bear_13_seq_103_frames/mask_to_record-->mask

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k --seq_paths=bear_13_seq_103_frames.txt --seq_postfix=masks_0x682_10 --output_path=/data/acamp/acamp20k/bear_13_seq_103_frames_0x682_10.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="bear_13_seq_103_frames_ex1___mask_to_record_"></a>
## bear_13_seq_103_frames_ex1       @ mask_to_record-->mask

<a id="0x682_1___bear_13_seq_103_frames_ex1_mask_to_recor_d_"></a>
### 0x682_1       @ bear_13_seq_103_frames_ex1/mask_to_record-->mask

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/bear_13_seq_103_frames_ex1_0x682_1 --seq_paths=bear_13_seq_103_frames_ex1.txt --seq_postfix=0x682_1 --output_path=/data/acamp/acamp20k/bear_13_seq_103_frames_ex1_0x682_1.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="coyote_9_seq_54_frames___mask_to_record_"></a>
## coyote_9_seq_54_frames       @ mask_to_record-->mask

<a id="0x682_1___coyote_9_seq_54_frames_mask_to_recor_d_"></a>
### 0x682_1       @ coyote_9_seq_54_frames/mask_to_record-->mask

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/coyote_9_seq_54_frames/masks --seq_paths=coyote_9_seq_54_frames.txt --seq_postfix=0x682_1 --output_path=/data/acamp/acamp20k/coyote_9_seq_54_frames_0x682_1.record --label_map_path=data/wildlife_label_map_mask.pbtxt


<a id="deer_11_seq_56_frames___mask_to_record_"></a>
## deer_11_seq_56_frames       @ mask_to_record-->mask

<a id="0x682_1___deer_11_seq_56_frames_mask_to_record_"></a>
### 0x682_1       @ deer_11_seq_56_frames/mask_to_record-->mask

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/deer_11_seq_56_frames/masks --seq_paths=deer_11_seq_56_frames.txt --seq_postfix=0x682_1 --output_path=/data/acamp/acamp20k/deer_11_seq_56_frames_0x682_1.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="moose_10_seq_50_frames___mask_to_record_"></a>
## moose_10_seq_50_frames       @ mask_to_record-->mask

<a id="0x682_1___moose_10_seq_50_frames_mask_to_recor_d_"></a>
### 0x682_1       @ moose_10_seq_50_frames/mask_to_record-->mask

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/moose_10_seq_50_frames/masks --seq_paths=moose_10_seq_50_frames.txt --seq_postfix=0x682_1 --output_path=/data/acamp/acamp20k/moose_10_seq_50_frames_0x682_1.record --label_map_path=data/wildlife_label_map_mask.pbtxt


<a id="augment_mas_k_"></a>
# augment_mask

<a id="beer_1_1___augment_mask_"></a>
## beer_1_1       @ augment_mask-->mask

<a id="airport___beer_1_1_augment_mas_k_"></a>
### airport       @ beer_1_1/augment_mask-->mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0/labels bkg_paths=/data/acamp/acamp20k/backgrounds/airport  aug_seq_prefix=bear_1_1_augmented_mask_airport only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1  bkg_size=1280x720 static_bkg=5 map_to_bbox=1

<a id="acamp_office___beer_1_1_augment_mas_k_"></a>
### acamp_office       @ beer_1_1/augment_mask-->mask

<a id="0x0_0___acamp_office_beer_1_1_augment_mask_"></a>
#### 0x0_0       @ acamp_office/beer_1_1/augment_mask-->mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear/bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0/labels bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1 bkg_size=1280x720 static_bkg=1 map_to_bbox=1

<a id="0x0_50___acamp_office_beer_1_1_augment_mask_"></a>
#### 0x0_50       @ acamp_office/beer_1_1/augment_mask-->mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear/bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_50/labels bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1 bkg_size=1280x720 static_bkg=1 map_to_bbox=1 border=50

<a id="general___beer_1_1_augment_mas_k_"></a>
### general       @ beer_1_1/augment_mask-->mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_1_1 mask_paths=/data/acamp/acamp20k/bear/bear_1_1_masks_0x682_10\labels bkg_paths=/data/acamp/acamp20k/backgrounds/general border=10 aug_seq_prefix=bear_1_1_augmented_mask_backgrounds only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1 static_bkg=3 bkg_size=1280x720

<a id="5_boxes_per_bkg___general_beer_1_1_augment_mas_k_"></a>
#### 5_boxes_per_bkg       @ general/beer_1_1/augment_mask-->mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_1_1 mask_paths=/data/acamp/acamp20k/bear\bear_1_1_masks_0x682_10\labels bkg_paths=/data/acamp/acamp20k/backgrounds/general border=10 aug_seq_prefix=bear_1_1_augmented_mask_backgrounds only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1 static_bkg=3 bkg_size=1280x720 boxes_per_bkg=5

<a id="bear_jesse_7_1_w___augment_mask_"></a>
## bear_jesse_7_1_w       @ augment_mask-->mask

<a id="airport___bear_jesse_7_1_w_augment_mas_k_"></a>
### airport       @ bear_jesse_7_1_w/augment_mask-->mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_jesse_7_1_w mask_paths=/data/acamp/acamp20k/masks/bear_jesse_7_1_w_0x0_0/labels bkg_paths=/data/acamp/acamp20k/backgrounds/airport  aug_seq_prefix=bear_1_1_augmented_mask_airport only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1  bkg_size=1280x720 static_bkg=13 map_to_bbox=1 show_bbox=1 show_blended=0

<a id="p1_deer___augment_mask_"></a>
## p1_deer       @ augment_mask-->mask

<a id="highway___p1_deer_augment_mask_"></a>
### highway       @ p1_deer/augment_mask-->mask

python3 augment_mask.py src_paths=p1_deer.txt src_root_dir=/data/acamp/acamp20k/deer bkg_paths=/data/acamp/acamp20k/backgrounds/highway only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels aug_seq_prefix=p1_deer src_id=0

<a id="airport___p1_deer_augment_mask_"></a>
### airport       @ p1_deer/augment_mask-->mask

python3 augment_mask.py src_paths=p1_deer.txt src_root_dir=/data/acamp/acamp20k/deer bkg_paths=/data/acamp/acamp20k/backgrounds/airport only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels save_path=/data/acamp/acamp20k/prototype_1 aug_seq_prefix=p1_deer_airport src_id=0

<a id="p1_coyote___augment_mask_"></a>
## p1_coyote       @ augment_mask-->mask

<a id="highway___p1_coyote_augment_mask_"></a>
### highway       @ p1_coyote/augment_mask-->mask

python3 augment_mask.py src_paths=p1_coyote.txt src_root_dir=/data/acamp/acamp20k/coyote mask_root_dir=/data/acamp/acamp20k/coyote/masks bkg_paths=/data/acamp/acamp20k/backgrounds/highway only_one_src_obj=1 aug_seq_size=1000 visualize=2 bkg_size=1280x720 static_bkg=-1 hist_match_type=0 mask_dir=labels boxes_per_bkg=1 aug_seq_prefix=p1_coyote src_id=0

<a id="airport___p1_coyote_augment_mask_"></a>
### airport       @ p1_coyote/augment_mask-->mask

python3 augment_mask.py src_paths=p1_coyote.txt src_root_dir=/data/acamp/acamp20k/coyote bkg_paths=/data/acamp/acamp20k/backgrounds/airport only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels save_path=/data/acamp/acamp20k/prototype_1 aug_seq_prefix=p1_coyote_airport src_id=0

<a id="p1_moose___augment_mask_"></a>
## p1_moose       @ augment_mask-->mask

<a id="highway___p1_moose_augment_mas_k_"></a>
### highway       @ p1_moose/augment_mask-->mask

python3 augment_mask.py src_paths=p1_moose.txt src_root_dir=/data/acamp/acamp20k/moose bkg_paths=/data/acamp/acamp20k/backgrounds/highway only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels aug_seq_prefix=p1_moose src_id=0

<a id="airport___p1_moose_augment_mas_k_"></a>
### airport       @ p1_moose/augment_mask-->mask

python3 augment_mask.py src_paths=p1_moose.txt src_root_dir=/data/acamp/acamp20k/moose bkg_paths=/data/acamp/acamp20k/backgrounds/airport only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels save_path=/data/acamp/acamp20k/prototype_1 aug_seq_prefix=p1_moose_airport src_id=0

<a id="p1_bear___augment_mask_"></a>
## p1_bear       @ augment_mask-->mask

<a id="highway___p1_bear_augment_mask_"></a>
### highway       @ p1_bear/augment_mask-->mask

python3 augment_mask.py src_paths=p1_bear.txt src_root_dir=/data/acamp/acamp20k bkg_paths=/data/acamp/acamp20k/backgrounds/highway only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=0x0_0 mask_dir=labels aug_seq_prefix=p1_bear_highway src_id=0

<a id="airport___p1_bear_augment_mask_"></a>
### airport       @ p1_bear/augment_mask-->mask

python3 augment_mask.py src_paths=p1_bear.txt src_root_dir=/data/acamp/acamp20k/bear bkg_paths=/data/acamp/acamp20k/backgrounds/airport only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels save_path=/data/acamp/acamp20k/prototype_1 aug_seq_prefix=p1_bear_airport src_id=0


<a id="visualize_masks_"></a>
# visualize_masks

<a id="animal___visualize_mask_s_"></a>
## animal       @ visualize_masks-->mask

<a id="bear_1_1___animal_visualize_masks_"></a>
### bear_1_1       @ animal/visualize_masks-->mask

<a id="inception_resnet_v2_size_400_max1res_66_233911__bear11visualize_masks"></a>

python3 visualize_masks.py img_paths=bear_1_1 img_root_dir=/data/acamp/acamp20k/bear mask_paths=bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0 mask_subdir=labels map_to_bbox=1

<a id="bear_13_seq_103_frames___animal_visualize_masks_"></a>
### bear_13_seq_103_frames       @ animal/visualize_masks-->mask

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_190313_1022 mask_subdir=labels map_to_bbox=1 border=0

<a id="bear_13_seq_103_frames_ex1___animal_visualize_masks_"></a>
### bear_13_seq_103_frames_ex1       @ animal/visualize_masks-->mask

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames_ex1.txt img_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_ex1 mask_subdir=labels

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames_ex1.txt img_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_ex1 mask_subdir=labels

<a id="0x682_10___bear_13_seq_103_frames_ex1_animal_visualize_mask_s_"></a>
#### 0x682_10       @ bear_13_seq_103_frames_ex1/animal/visualize_masks-->mask

python3 visualize_masks.py img_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_ex1_0x682_10  img_subdir=images mask_subdir=labels border=10

<a id="0x682_0___bear_13_seq_103_frames_ex1_animal_visualize_mask_s_"></a>
#### 0x682_0       @ bear_13_seq_103_frames_ex1/animal/visualize_masks-->mask

python3 visualize_masks.py img_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_ex1_0x682_0  img_subdir=images mask_subdir=labels border=0

<a id="reorg_roi___visualize_mask_s_"></a>
## reorg_roi       @ visualize_masks-->mask

python3 visualize_masks.py img_paths=reorg_roi.txt img_root_dir=/data/ipsc/well_3/reorg_roi mask_paths=bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0 mask_subdir=labels map_to_bbox=1

