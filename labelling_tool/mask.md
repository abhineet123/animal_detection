<!-- MarkdownTOC -->

- [to_mask_seq](#to_mask_seq)
    - [bear_1_1       @ to_mask_seq](#bear11__to_mask_seq)
        - [0x0_0       @ bear_1_1/to_mask_seq](#0x0_0__bear11to_mask_seq)
        - [0x1365_10       @ bear_1_1/to_mask_seq](#0x1365_10__bear11to_mask_seq)
        - [0x682_50       @ bear_1_1/to_mask_seq](#0x682_50__bear11to_mask_seq)
        - [224x224_0       @ bear_1_1/to_mask_seq](#224x224_0__bear11to_mask_seq)
        - [224x224_5       @ bear_1_1/to_mask_seq](#224x224_5__bear11to_mask_seq)
        - [224x224_10       @ bear_1_1/to_mask_seq](#224x224_10__bear11to_mask_seq)
        - [448_448_0       @ bear_1_1/to_mask_seq](#448_448_0__bear11to_mask_seq)
        - [448_448_10       @ bear_1_1/to_mask_seq](#448_448_10__bear11to_mask_seq)
            - [448_448_25       @ 448_448_10/bear_1_1/to_mask_seq](#448_448_25__448_448_10bear11to_mask_seq)
            - [fixed_ar       @ 448_448_10/bear_1_1/to_mask_seq](#fixed_ar__448_448_10bear11to_mask_seq)
        - [500x500_10       @ bear_1_1/to_mask_seq](#500x500_10__bear11to_mask_seq)
    - [bear_1_1_to_bear_1_6       @ to_mask_seq](#bear11_to_bear16__to_mask_seq)
        - [0x0_0       @ bear_1_1_to_bear_1_6/to_mask_seq](#0x0_0__bear11_to_bear16to_mask_seq)
            - [combine_seq       @ 0x0_0/bear_1_1_to_bear_1_6/to_mask_seq](#combine_seq__0x0_0bear11_to_bear16to_mask_seq)
                - [map_to_bbox       @ combine_seq/0x0_0/bear_1_1_to_bear_1_6/to_mask_seq](#map_to_bbox__combine_seq0x0_0bear11_to_bear16to_mask_seq)
        - [224x224_0_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#224x2240test__bear11_to_bear16to_mask_seq)
        - [0x682_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#0x682_10_test__bear11_to_bear16to_mask_seq)
        - [0x682_50_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#0x682_50_test__bear11_to_bear16to_mask_seq)
        - [448x448_ar_1p0_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#448x448_ar_1p0_test__bear11_to_bear16to_mask_seq)
        - [448x448_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#448x448_10_test__bear11_to_bear16to_mask_seq)
            - [map_to_bbox       @ 448x448_10_test/bear_1_1_to_bear_1_6/to_mask_seq](#map_to_bbox__448x448_10_testbear11_to_bear16to_mask_seq)
        - [500x500_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq](#500x500_10_test__bear11_to_bear16to_mask_seq)
    - [bear_1_2_to_1_6       @ to_mask_seq](#bear12_to16__to_mask_seq)
        - [448x448_25_test       @ bear_1_2_to_1_6/to_mask_seq](#448x448_25_test__bear12_to16to_mask_seq)
        - [combine_seq       @ bear_1_2_to_1_6/to_mask_seq](#combine_seq__bear12_to16to_mask_seq)
        - [448x448_0_test       @ bear_1_2_to_1_6/to_mask_seq](#448x4480test__bear12_to16to_mask_seq)
            - [fixed_ar       @ 448x448_0_test/bear_1_2_to_1_6/to_mask_seq](#fixed_ar__448x4480testbear12_to16to_mask_seq)
    - [bear_8_2       @ to_mask_seq](#bear82__to_mask_seq)
        - [400x0_0       @ bear_8_2/to_mask_seq](#400x0_0__bear82to_mask_seq)
    - [bear_10_1       @ to_mask_seq](#bear_10_1__to_mask_seq)
    - [bear_13_seq_103_frames       @ to_mask_seq](#bear_13_seq_103_frames__to_mask_seq)
        - [map_to_bbox       @ bear_13_seq_103_frames/to_mask_seq](#map_to_bbox__bear_13_seq_103_framesto_mask_seq)
        - [800x800_25       @ bear_13_seq_103_frames/to_mask_seq](#800x800_25__bear_13_seq_103_framesto_mask_seq)
        - [0x682_0       @ bear_13_seq_103_frames/to_mask_seq](#0x682_0__bear_13_seq_103_framesto_mask_seq)
        - [0x682_1       @ bear_13_seq_103_frames/to_mask_seq](#0x682_1__bear_13_seq_103_framesto_mask_seq)
        - [0x682_10       @ bear_13_seq_103_frames/to_mask_seq](#0x682_10__bear_13_seq_103_framesto_mask_seq)
        - [0x682_2       @ bear_13_seq_103_frames/to_mask_seq](#0x682_2__bear_13_seq_103_framesto_mask_seq)
            - [test       @ 0x682_2/bear_13_seq_103_frames/to_mask_seq](#test__0x682_2bear_13_seq_103_framesto_mask_seq)
                - [bear_1_1       @ test/0x682_2/bear_13_seq_103_frames/to_mask_seq](#bear11__test0x682_2bear_13_seq_103_framesto_mask_seq)
            - [bear_13_seq_103_frames_0x682_10_test       @ 0x682_2/bear_13_seq_103_frames/to_mask_seq](#bear_13_seq_103_frames_0x682_10_test__0x682_2bear_13_seq_103_framesto_mask_seq)
        - [224x224_5       @ bear_13_seq_103_frames/to_mask_seq](#224x224_5__bear_13_seq_103_framesto_mask_seq)
            - [combine_seq       @ 224x224_5/bear_13_seq_103_frames/to_mask_seq](#combine_seq__224x224_5bear_13_seq_103_framesto_mask_seq)
        - [224x224_5_test       @ bear_13_seq_103_frames/to_mask_seq](#224x2245test__bear_13_seq_103_framesto_mask_seq)
            - [combine_seq       @ 224x224_5_test/bear_13_seq_103_frames/to_mask_seq](#combine_seq__224x2245testbear_13_seq_103_framesto_mask_seq)
            - [bear_1_1_even_10       @ 224x224_5_test/bear_13_seq_103_frames/to_mask_seq](#bear11_even_10__224x2245testbear_13_seq_103_framesto_mask_seq)
        - [bear_13_seq_103_frames_448x448_10_test       @ bear_13_seq_103_frames/to_mask_seq](#bear_13_seq_103_frames_448x448_10_test__bear_13_seq_103_framesto_mask_seq)
        - [bear_13_seq_103_frames_800x800_25_test       @ bear_13_seq_103_frames/to_mask_seq](#bear_13_seq_103_frames_800x800_25_test__bear_13_seq_103_framesto_mask_seq)
            - [map_to_bbox       @ bear_13_seq_103_frames_800x800_25_test/bear_13_seq_103_frames/to_mask_seq](#map_to_bbox__bear_13_seq_103_frames_800x800_25_testbear_13_seq_103_framesto_mask_seq)
    - [bear_13_seq_103_frames_ex1       @ to_mask_seq](#bear_13_seq_103_frames_ex1__to_mask_seq)
    - [coyote_9_seq_54_frames       @ to_mask_seq](#coyote9seq_54_frames__to_mask_seq)
        - [map_to_bbox       @ coyote_9_seq_54_frames/to_mask_seq](#map_to_bbox__coyote9seq_54_framesto_mask_seq)
        - [0x682_1       @ coyote_9_seq_54_frames/to_mask_seq](#0x682_1__coyote9seq_54_framesto_mask_seq)
    - [deer_11_seq_56_frames       @ to_mask_seq](#deer_11_seq_56_frames__to_mask_seq)
        - [map_to_bbox       @ deer_11_seq_56_frames/to_mask_seq](#map_to_bbox__deer_11_seq_56_framesto_mask_seq)
            - [deer_12_1       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq](#deer_12_1__map_to_bboxdeer_11_seq_56_framesto_mask_seq)
            - [deer_jesse_2_1       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq](#deer_jesse21__map_to_bboxdeer_11_seq_56_framesto_mask_seq)
            - [deer_jesse_2_2       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq](#deer_jesse22__map_to_bboxdeer_11_seq_56_framesto_mask_seq)
        - [0x682_1       @ deer_11_seq_56_frames/to_mask_seq](#0x682_1__deer_11_seq_56_framesto_mask_seq)
    - [moose_10_seq_50_frames       @ to_mask_seq](#moose_10_seq_50_frames__to_mask_seq)
        - [map_to_bbox       @ moose_10_seq_50_frames/to_mask_seq](#map_to_bbox__moose_10_seq_50_framesto_mask_seq)
        - [0x682_1       @ moose_10_seq_50_frames/to_mask_seq](#0x682_1__moose_10_seq_50_framesto_mask_seq)
    - [p1_deer       @ to_mask_seq](#p1_deer__to_mask_seq)
        - [0x0_0       @ p1_deer/to_mask_seq](#0x0_0__p1_deerto_mask_seq)
            - [acamp_all_deer       @ 0x0_0/p1_deer/to_mask_seq](#acamp_all_deer__0x0_0p1_deerto_mask_seq)
        - [map_to_bbox       @ p1_deer/to_mask_seq](#map_to_bbox__p1_deerto_mask_seq)
    - [p1_coyote       @ to_mask_seq](#p1_coyote__to_mask_seq)
        - [0x0_0       @ p1_coyote/to_mask_seq](#0x0_0__p1_coyoteto_mask_seq)
            - [coyote_9_1       @ 0x0_0/p1_coyote/to_mask_seq](#coyote91__0x0_0p1_coyoteto_mask_seq)
            - [coyote_10_5       @ 0x0_0/p1_coyote/to_mask_seq](#coyote_10_5__0x0_0p1_coyoteto_mask_seq)
            - [coyote_b       @ 0x0_0/p1_coyote/to_mask_seq](#coyote_b__0x0_0p1_coyoteto_mask_seq)
        - [map_to_bbox       @ p1_coyote/to_mask_seq](#map_to_bbox__p1_coyoteto_mask_seq)
    - [p1_moose       @ to_mask_seq](#p1_moose__to_mask_seq)
        - [0x0_0       @ p1_moose/to_mask_seq](#0x0_0__p1_mooseto_mask_seq)
            - [moose_5_2       @ 0x0_0/p1_moose/to_mask_seq](#moose52__0x0_0p1_mooseto_mask_seq)
        - [map_to_bbox       @ p1_moose/to_mask_seq](#map_to_bbox__p1_mooseto_mask_seq)
    - [p1_bear       @ to_mask_seq](#p1_bear__to_mask_seq)
        - [0x0_0       @ p1_bear/to_mask_seq](#0x0_0__p1_bearto_mask_seq)
            - [bear_1_1       @ 0x0_0/p1_bear/to_mask_seq](#bear11__0x0_0p1_bearto_mask_seq)
        - [map_to_bbox       @ p1_bear/to_mask_seq](#map_to_bbox__p1_bearto_mask_seq)
    - [p1_source_0x682_1       @ to_mask_seq](#p1_source_0x682_1__to_mask_seq)
- [mask_to_record](#mask_to_record)
    - [bear_1_1       @ mask_to_record](#bear11__mask_to_record)
        - [0x682_50       @ bear_1_1/mask_to_record](#0x682_50__bear11mask_to_record)
    - [bear_13_seq_103_frames       @ mask_to_record](#bear_13_seq_103_frames__mask_to_record)
        - [0x682_0       @ bear_13_seq_103_frames/mask_to_record](#0x682_0__bear_13_seq_103_framesmask_to_record)
        - [0x682_10       @ bear_13_seq_103_frames/mask_to_record](#0x682_10__bear_13_seq_103_framesmask_to_record)
    - [bear_13_seq_103_frames_ex1       @ mask_to_record](#bear_13_seq_103_frames_ex1__mask_to_record)
        - [0x682_1       @ bear_13_seq_103_frames_ex1/mask_to_record](#0x682_1__bear_13_seq_103_frames_ex1mask_to_record)
    - [coyote_9_seq_54_frames       @ mask_to_record](#coyote9seq_54_frames__mask_to_record)
        - [0x682_1       @ coyote_9_seq_54_frames/mask_to_record](#0x682_1__coyote9seq_54_framesmask_to_record)
    - [deer_11_seq_56_frames       @ mask_to_record](#deer_11_seq_56_frames__mask_to_record)
        - [0x682_1       @ deer_11_seq_56_frames/mask_to_record](#0x682_1__deer_11_seq_56_framesmask_to_record)
    - [moose_10_seq_50_frames       @ mask_to_record](#moose_10_seq_50_frames__mask_to_record)
        - [0x682_1       @ moose_10_seq_50_frames/mask_to_record](#0x682_1__moose_10_seq_50_framesmask_to_record)
- [augment_mask](#augment_mask)
    - [beer_1_1       @ augment_mask](#beer11__augment_mask)
        - [airport       @ beer_1_1/augment_mask](#airport__beer11augment_mask)
        - [acamp_office       @ beer_1_1/augment_mask](#acamp_office__beer11augment_mask)
            - [0x0_0       @ acamp_office/beer_1_1/augment_mask](#0x0_0__acamp_officebeer11augment_mask)
            - [0x0_50       @ acamp_office/beer_1_1/augment_mask](#0x0_50__acamp_officebeer11augment_mask)
        - [general       @ beer_1_1/augment_mask](#general__beer11augment_mask)
            - [5_boxes_per_bkg       @ general/beer_1_1/augment_mask](#5_boxes_per_bkg__generalbeer11augment_mask)
    - [bear_jesse_7_1_w       @ augment_mask](#bear_jesse71_w__augment_mask)
        - [airport       @ bear_jesse_7_1_w/augment_mask](#airport__bear_jesse71_waugment_mask)
    - [p1_deer       @ augment_mask](#p1_deer__augment_mask)
        - [highway       @ p1_deer/augment_mask](#highway__p1_deeraugment_mask)
        - [airport       @ p1_deer/augment_mask](#airport__p1_deeraugment_mask)
    - [p1_coyote       @ augment_mask](#p1_coyote__augment_mask)
        - [highway       @ p1_coyote/augment_mask](#highway__p1_coyoteaugment_mask)
        - [airport       @ p1_coyote/augment_mask](#airport__p1_coyoteaugment_mask)
    - [p1_moose       @ augment_mask](#p1_moose__augment_mask)
        - [highway       @ p1_moose/augment_mask](#highway__p1_mooseaugment_mask)
        - [airport       @ p1_moose/augment_mask](#airport__p1_mooseaugment_mask)
    - [p1_bear       @ augment_mask](#p1_bear__augment_mask)
        - [highway       @ p1_bear/augment_mask](#highway__p1_bearaugment_mask)
        - [airport       @ p1_bear/augment_mask](#airport__p1_bearaugment_mask)
- [visualize_masks](#visualize_masks)
        - [bear_1_1       @ visualize_masks/](#bear11__visualize_masks)
        - [bear_13_seq_103_frames       @ visualize_masks/](#bear_13_seq_103_frames__visualize_masks)
        - [bear_13_seq_103_frames_ex1       @ visualize_masks/](#bear_13_seq_103_frames_ex1__visualize_masks)
            - [0x682_10       @ bear_13_seq_103_frames_ex1/visualize_masks/](#0x682_10__bear_13_seq_103_frames_ex1visualize_masks)
            - [0x682_0       @ bear_13_seq_103_frames_ex1/visualize_masks/](#0x682_0__bear_13_seq_103_frames_ex1visualize_masks)

<!-- /MarkdownTOC -->

<a id="to_mask_seq"></a>
# to_mask_seq

<a id="bear11__to_mask_seq"></a>
## bear_1_1       @ to_mask_seq

<a id="0x0_0__bear11to_mask_seq"></a>
### 0x0_0       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x0 out_border=0 show_img=0

<a id="0x1365_10__bear11to_mask_seq"></a>
### 0x1365_10       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x1365 out_border=10 show_img=0
<a id="0x682_50__bear11to_mask_seq"></a>
### 0x682_50       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=50 show_img=0

<a id="224x224_0__bear11to_mask_seq"></a>
### 224x224_0       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=224x224 out_border=5 show_img=0 save_test=0

<a id="224x224_5__bear11to_mask_seq"></a>
### 224x224_5       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=224x224 out_border=5 show_img=0

<a id="224x224_10__bear11to_mask_seq"></a>
### 224x224_10       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=224x224 out_border=10 show_img=0

<a id="448_448_0__bear11to_mask_seq"></a>
### 448_448_0       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=448x448 out_border=0 show_img=0

<a id="448_448_10__bear11to_mask_seq"></a>
### 448_448_10       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=10 show_img=0 save_raw_mask=1

<a id="448_448_25__448_448_10bear11to_mask_seq"></a>
#### 448_448_25       @ 448_448_10/bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=448x448 out_border=25 show_img=0

<a id="fixed_ar__448_448_10bear11to_mask_seq"></a>
#### fixed_ar       @ 448_448_10/bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=0 show_img=0 fixed_ar=1 save_raw_mask=1

<a id="500x500_10__bear11to_mask_seq"></a>
### 500x500_10       @ bear_1_1/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=500x500 out_border=10 show_img=0 save_raw_mask=1

<a id="bear11_to_bear16__to_mask_seq"></a>
## bear_1_1_to_bear_1_6       @ to_mask_seq

<a id="0x0_0__bear11_to_bear16to_mask_seq"></a>
### 0x0_0       @ bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x0 out_border=0 show_img=0

<a id="combine_seq__0x0_0bear11_to_bear16to_mask_seq"></a>
#### combine_seq       @ 0x0_0/bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k save_train=0 out_mask_size=0x0 out_border=0 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_0x0_0_test combine_seq=1

<a id="map_to_bbox__combine_seq0x0_0bear11_to_bear16to_mask_seq"></a>
##### map_to_bbox       @ combine_seq/0x0_0/bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear save_train=0 out_mask_size=0x0 out_border=0 show_img=1 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_mapped combine_seq=1 map_to_bbox=1

<a id="224x2240test__bear11_to_bear16to_mask_seq"></a>
### 224x224_0_test       @ bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=224x224 out_border=0 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_224x224_0_test save_train=0

<a id="0x682_10_test__bear11_to_bear16to_mask_seq"></a>
### 0x682_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_10_test save_train=0

<a id="0x682_50_test__bear11_to_bear16to_mask_seq"></a>
### 0x682_50_test       @ bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=50 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_bear_1_6_0x682_50_test save_train=0

<a id="448x448_ar_1p0_test__bear11_to_bear16to_mask_seq"></a>
### 448x448_ar_1p0_test       @ bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k save_train=0 out_mask_size=448x448 out_border=0 fixed_ar=1 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_448x448_ar_1p0_test combine_seq=1

<a id="448x448_10_test__bear11_to_bear16to_mask_seq"></a>
### 448x448_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k save_train=0 out_mask_size=448x448 out_border=10 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_448x448_10_test combine_seq=1

<a id="map_to_bbox__448x448_10_testbear11_to_bear16to_mask_seq"></a>
#### map_to_bbox       @ 448x448_10_test/bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear save_train=0 out_mask_size=448x448 out_border=10 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_448x448_10_mapped_test combine_seq=1 map_to_bbox=1

<a id="500x500_10_test__bear11_to_bear16to_mask_seq"></a>
### 500x500_10_test       @ bear_1_1_to_bear_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1,bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k save_train=0 out_mask_size=500x500 out_border=10 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_1_to_1_6_500x500_10_test/images combine_seq=1

<a id="bear12_to16__to_mask_seq"></a>
## bear_1_2_to_1_6       @ to_mask_seq

<a id="448x448_25_test__bear12_to16to_mask_seq"></a>
### 448x448_25_test       @ bear_1_2_to_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=25 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_2_to_1_6_448x448_25_test save_train=0 save_test=1

<a id="combine_seq__bear12_to16to_mask_seq"></a>
### combine_seq       @ bear_1_2_to_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=448x448 out_border=25 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_2_to_1_6_448x448_25_test save_train=0 save_test=1 combine_seq=1

<a id="448x4480test__bear12_to16to_mask_seq"></a>
### 448x448_0_test       @ bear_1_2_to_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=0 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_2_to_1_6_448x448_0_test save_train=0 save_test=1

<a id="fixed_ar__448x4480testbear12_to16to_mask_seq"></a>
#### fixed_ar       @ 448x448_0_test/bear_1_2_to_1_6/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_2,bear_1_3,bear_1_4,bear_1_5,bear_1_6 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=0 fixed_ar=1 show_img=0 out_root_path=/data/acamp/acamp20k/masks/bear_1_2_to_1_6_448x448_ar_1p0_test save_train=0 save_test=1 combine_seq=1 

<a id="bear82__to_mask_seq"></a>
## bear_8_2       @ to_mask_seq

<a id="400x0_0__bear82to_mask_seq"></a>
### 400x0_0       @ bear_8_2/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_8_2 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=400x0 out_border=0 show_img=0 save_test=0 save_raw_mask=0

<a id="bear_10_1__to_mask_seq"></a>
## bear_10_1       @ to_mask_seq

python3 to_mask_seq.py seq_paths=bear_10_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=400x0 out_border=0 show_img=0 save_test=0 save_raw_mask=0

<a id="bear_13_seq_103_frames__to_mask_seq"></a>
## bear_13_seq_103_frames       @ to_mask_seq

<a id="map_to_bbox__bear_13_seq_103_framesto_mask_seq"></a>
### map_to_bbox       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=800x800 out_border=25 show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1


<a id="800x800_25__bear_13_seq_103_framesto_mask_seq"></a>
### 800x800_25       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=800x800 out_border=25 show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_800x800_25 combine_seq=1

<a id="0x682_0__bear_13_seq_103_framesto_mask_seq"></a>
### 0x682_0       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=0x682 out_border=0 show_img=0 save_test=0 save_raw_mask=0

<a id="0x682_1__bear_13_seq_103_framesto_mask_seq"></a>
### 0x682_1       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=1 show_img=0 save_test=0 save_raw_mask=0 allow_skipping_images=1

<a id="0x682_10__bear_13_seq_103_framesto_mask_seq"></a>
### 0x682_10       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 save_test=0 save_raw_mask=0

<a id="0x682_2__bear_13_seq_103_framesto_mask_seq"></a>
### 0x682_2       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=2 show_img=0 save_test=0 save_raw_mask=0

<a id="test__0x682_2bear_13_seq_103_framesto_mask_seq"></a>
#### test       @ 0x682_2/bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 save_test=1 save_raw_mask=0

<a id="bear11__test0x682_2bear_13_seq_103_framesto_mask_seq"></a>
##### bear_1_1       @ test/0x682_2/bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=0x682 out_border=10 show_img=0 save_test=1 save_raw_mask=0

<a id="bear_13_seq_103_frames_0x682_10_test__0x682_2bear_13_seq_103_framesto_mask_seq"></a>
#### bear_13_seq_103_frames_0x682_10_test       @ 0x682_2/bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 save_test=1  save_train=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_0x682_10_test combine_seq=0


<a id="224x224_5__bear_13_seq_103_framesto_mask_seq"></a>
### 224x224_5       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=0 save_raw_mask=0

<a id="combine_seq__224x224_5bear_13_seq_103_framesto_mask_seq"></a>
#### combine_seq       @ 224x224_5/bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5 combine_seq=1



<a id="224x2245test__bear_13_seq_103_framesto_mask_seq"></a>
### 224x224_5_test       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=1 save_train=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5_test

<a id="combine_seq__224x2245testbear_13_seq_103_framesto_mask_seq"></a>
#### combine_seq       @ 224x224_5_test/bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=1 save_train=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5_test combine_seq=1

<a id="bear11_even_10__224x2245testbear_13_seq_103_framesto_mask_seq"></a>
#### bear_1_1_even_10       @ 224x224_5_test/bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1_even_10 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=1 save_train=0 save_raw_mask=0 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5_test

<a id="bear_13_seq_103_frames_448x448_10_test__bear_13_seq_103_framesto_mask_seq"></a>
### bear_13_seq_103_frames_448x448_10_test       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=448x448 out_border=10 show_img=0 save_test=1 save_train=0 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_448x448_10_test combine_seq=1
<a id="bear_13_seq_103_frames_800x800_25_test__bear_13_seq_103_framesto_mask_seq"></a>
### bear_13_seq_103_frames_800x800_25_test       @ bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=800x800 out_border=25 show_img=0 save_test=1 save_train=0 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_800x800_25_test combine_seq=1

<a id="map_to_bbox__bear_13_seq_103_frames_800x800_25_testbear_13_seq_103_framesto_mask_seq"></a>
#### map_to_bbox       @ bear_13_seq_103_frames_800x800_25_test/bear_13_seq_103_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=800x800 out_border=25 show_img=0 save_test=1 save_train=0 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_800x800_25_mapped_test combine_seq=1 map_to_bbox=1

<a id="bear_13_seq_103_frames_ex1__to_mask_seq"></a>
## bear_13_seq_103_frames_ex1       @ to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/bear_13_seq_103_frames_ex1.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear_13_seq_103_frames_ex1 out_mask_size=0x682 out_border=10 show_img=0 save_test=0 save_raw_mask=0

<a id="coyote9seq_54_frames__to_mask_seq"></a>
## coyote_9_seq_54_frames       @ to_mask_seq

<a id="map_to_bbox__coyote9seq_54_framesto_mask_seq"></a>
### map_to_bbox       @ coyote_9_seq_54_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/coyote_9_seq_54_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/coyote_9_seq_54_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1

<a id="0x682_1__coyote9seq_54_framesto_mask_seq"></a>
### 0x682_1       @ coyote_9_seq_54_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/coyote_9_seq_54_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote_9_seq_54_frames out_mask_size=0x682 out_border=1 show_img=0 save_test=0 save_raw_mask=0 allow_skipping_images=0

<a id="deer_11_seq_56_frames__to_mask_seq"></a>
## deer_11_seq_56_frames       @ to_mask_seq

<a id="map_to_bbox__deer_11_seq_56_framesto_mask_seq"></a>
### map_to_bbox       @ deer_11_seq_56_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/deer_11_seq_56_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/deer_11_seq_56_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1

<a id="deer_12_1__map_to_bboxdeer_11_seq_56_framesto_mask_seq"></a>
#### deer_12_1       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=deer_12_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/deer_11_seq_56_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1

<a id="deer_jesse21__map_to_bboxdeer_11_seq_56_framesto_mask_seq"></a>
#### deer_jesse_2_1       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=deer_jesse_2_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/deer_11_seq_56_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1

<a id="deer_jesse22__map_to_bboxdeer_11_seq_56_framesto_mask_seq"></a>
#### deer_jesse_2_2       @ map_to_bbox/deer_11_seq_56_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=deer_jesse_2_2 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/deer_11_seq_56_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1


<a id="0x682_1__deer_11_seq_56_framesto_mask_seq"></a>
### 0x682_1       @ deer_11_seq_56_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/deer_11_seq_56_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/deer_11_seq_56_frames out_mask_size=0x682 out_border=1 show_img=0 save_test=0 save_raw_mask=0 allow_skipping_images=0

<a id="moose_10_seq_50_frames__to_mask_seq"></a>
## moose_10_seq_50_frames       @ to_mask_seq

<a id="map_to_bbox__moose_10_seq_50_framesto_mask_seq"></a>
### map_to_bbox       @ moose_10_seq_50_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/moose_10_seq_50_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=1 save_raw_mask=1 out_root_path=/data/acamp/acamp20k/masks/moose_10_seq_50_frames save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1 enable_xml_annotations=1


<a id="0x682_1__moose_10_seq_50_framesto_mask_seq"></a>
### 0x682_1       @ moose_10_seq_50_frames/to_mask_seq

python3 to_mask_seq.py seq_paths=../tf_api/moose_10_seq_50_frames.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/moose_10_seq_50_frames out_mask_size=0x682 out_border=1 show_img=0 save_test=0 save_raw_mask=0 allow_skipping_images=0


<a id="p1_deer__to_mask_seq"></a>
## p1_deer       @ to_mask_seq

<a id="0x0_0__p1_deerto_mask_seq"></a>
### 0x0_0       @ p1_deer/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_deer.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/deer out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0

<a id="acamp_all_deer__0x0_0p1_deerto_mask_seq"></a>
#### acamp_all_deer       @ 0x0_0/p1_deer/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_deer.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/deer out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0

<a id="map_to_bbox__p1_deerto_mask_seq"></a>
### map_to_bbox       @ p1_deer/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_deer.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1

<a id="p1_coyote__to_mask_seq"></a>
## p1_coyote       @ to_mask_seq

<a id="0x0_0__p1_coyoteto_mask_seq"></a>
### 0x0_0       @ p1_coyote/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_coyote.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="coyote91__0x0_0p1_coyoteto_mask_seq"></a>
#### coyote_9_1       @ 0x0_0/p1_coyote/to_mask_seq

python3 to_mask_seq.py seq_paths=coyote_9_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="coyote_10_5__0x0_0p1_coyoteto_mask_seq"></a>
#### coyote_10_5       @ 0x0_0/p1_coyote/to_mask_seq

python3 to_mask_seq.py seq_paths=coyote_10_5 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="coyote_b__0x0_0p1_coyoteto_mask_seq"></a>
#### coyote_b       @ 0x0_0/p1_coyote/to_mask_seq

python3 to_mask_seq.py seq_paths=coyote_b class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/coyote out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="map_to_bbox__p1_coyoteto_mask_seq"></a>
### map_to_bbox       @ p1_coyote/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_coyote.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1

<a id="p1_moose__to_mask_seq"></a>
## p1_moose       @ to_mask_seq

<a id="0x0_0__p1_mooseto_mask_seq"></a>
### 0x0_0       @ p1_moose/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_moose.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/moose out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="moose52__0x0_0p1_mooseto_mask_seq"></a>
#### moose_5_2       @ 0x0_0/p1_moose/to_mask_seq

python3 to_mask_seq.py seq_paths=moose_5_2 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/acamp20k_annotations out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 

<a id="map_to_bbox__p1_mooseto_mask_seq"></a>
### map_to_bbox       @ p1_moose/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_moose.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1

<a id="p1_bear__to_mask_seq"></a>
## p1_bear       @ to_mask_seq

<a id="0x0_0__p1_bearto_mask_seq"></a>
### 0x0_0       @ p1_bear/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_bear.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 masks_per_seq=1 

<a id="bear11__0x0_0p1_bearto_mask_seq"></a>
#### bear_1_1       @ 0x0_0/p1_bear/to_mask_seq

python3 to_mask_seq.py seq_paths=bear_1_1 class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=0x0 out_border=0 show_img=1 save_test=0 save_raw_mask=0 masks_per_seq=1 

<a id="map_to_bbox__p1_bearto_mask_seq"></a>
### map_to_bbox       @ p1_bear/to_mask_seq

python3 to_mask_seq.py seq_paths=p1_bear.txt class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k show_img=0 save_test=0 save_raw_mask=0 out_img_dir=__n__ enable_out_suffix=0 map_to_bbox=1

<a id="p1_source_0x682_1__to_mask_seq"></a>
## p1_source_0x682_1       @ to_mask_seq

python3 to_mask_seq.py class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/prototype_1_source out_mask_size=0x682 out_border=1 show_img=1 save_test=0 save_raw_mask=0 map_to_bbox=0 out_img_dir=__n__



<a id="mask_to_record"></a>
# mask_to_record

<a id="bear11__mask_to_record"></a>
## bear_1_1       @ mask_to_record

python2 mask_to_record.py --root_dir=/data/acamp/acamp20k --seq_paths=bear_1_1_masks_0x1365_10 --output_path=/data/acamp/acamp20k/bear_1_1_masks_0x1365_10.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="0x682_50__bear11mask_to_record"></a>
### 0x682_50       @ bear_1_1/mask_to_record

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/masks --seq_paths=bear_1_1_0x682_50 --output_path=/data/acamp/acamp20k/masks/bear_1_1_0x682_50.record --label_map_path=data/wildlife_label_map_mask.pbtxt


<a id="bear_13_seq_103_frames__mask_to_record"></a>
## bear_13_seq_103_frames       @ mask_to_record

<a id="0x682_0__bear_13_seq_103_framesmask_to_record"></a>
### 0x682_0       @ bear_13_seq_103_frames/mask_to_record

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k --seq_paths=bear_13_seq_103_frames.txt --seq_postfix=masks_0x682_0 --output_path=/data/acamp/acamp20k/bear_13_seq_103_frames_0x682_0.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="0x682_10__bear_13_seq_103_framesmask_to_record"></a>
### 0x682_10       @ bear_13_seq_103_frames/mask_to_record

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k --seq_paths=bear_13_seq_103_frames.txt --seq_postfix=masks_0x682_10 --output_path=/data/acamp/acamp20k/bear_13_seq_103_frames_0x682_10.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="bear_13_seq_103_frames_ex1__mask_to_record"></a>
## bear_13_seq_103_frames_ex1       @ mask_to_record

<a id="0x682_1__bear_13_seq_103_frames_ex1mask_to_record"></a>
### 0x682_1       @ bear_13_seq_103_frames_ex1/mask_to_record

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/bear_13_seq_103_frames_ex1_0x682_1 --seq_paths=bear_13_seq_103_frames_ex1.txt --seq_postfix=0x682_1 --output_path=/data/acamp/acamp20k/bear_13_seq_103_frames_ex1_0x682_1.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="coyote9seq_54_frames__mask_to_record"></a>
## coyote_9_seq_54_frames       @ mask_to_record

<a id="0x682_1__coyote9seq_54_framesmask_to_record"></a>
### 0x682_1       @ coyote_9_seq_54_frames/mask_to_record

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/coyote_9_seq_54_frames/masks --seq_paths=coyote_9_seq_54_frames.txt --seq_postfix=0x682_1 --output_path=/data/acamp/acamp20k/coyote_9_seq_54_frames_0x682_1.record --label_map_path=data/wildlife_label_map_mask.pbtxt


<a id="deer_11_seq_56_frames__mask_to_record"></a>
## deer_11_seq_56_frames       @ mask_to_record

<a id="0x682_1__deer_11_seq_56_framesmask_to_record"></a>
### 0x682_1       @ deer_11_seq_56_frames/mask_to_record

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/deer_11_seq_56_frames/masks --seq_paths=deer_11_seq_56_frames.txt --seq_postfix=0x682_1 --output_path=/data/acamp/acamp20k/deer_11_seq_56_frames_0x682_1.record --label_map_path=data/wildlife_label_map_mask.pbtxt

<a id="moose_10_seq_50_frames__mask_to_record"></a>
## moose_10_seq_50_frames       @ mask_to_record

<a id="0x682_1__moose_10_seq_50_framesmask_to_record"></a>
### 0x682_1       @ moose_10_seq_50_frames/mask_to_record

python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/moose_10_seq_50_frames/masks --seq_paths=moose_10_seq_50_frames.txt --seq_postfix=0x682_1 --output_path=/data/acamp/acamp20k/moose_10_seq_50_frames_0x682_1.record --label_map_path=data/wildlife_label_map_mask.pbtxt


<a id="augment_mask"></a>
# augment_mask

<a id="beer11__augment_mask"></a>
## beer_1_1       @ augment_mask

<a id="airport__beer11augment_mask"></a>
### airport       @ beer_1_1/augment_mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0/labels bkg_paths=/data/acamp/acamp20k/backgrounds/airport  aug_seq_prefix=bear_1_1_augmented_mask_airport only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1  bkg_size=1280x720 static_bkg=5 map_to_bbox=1

<a id="acamp_office__beer11augment_mask"></a>
### acamp_office       @ beer_1_1/augment_mask

<a id="0x0_0__acamp_officebeer11augment_mask"></a>
#### 0x0_0       @ acamp_office/beer_1_1/augment_mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear/bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0/labels bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1 bkg_size=1280x720 static_bkg=1 map_to_bbox=1

<a id="0x0_50__acamp_officebeer11augment_mask"></a>
#### 0x0_50       @ acamp_office/beer_1_1/augment_mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear/bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_50/labels bkg_paths=/data/acamp/acamp20k/backgrounds/acamp_office only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1 bkg_size=1280x720 static_bkg=1 map_to_bbox=1 border=50

<a id="general__beer11augment_mask"></a>
### general       @ beer_1_1/augment_mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_1_1 mask_paths=/data/acamp/acamp20k/bear/bear_1_1_masks_0x682_10\labels bkg_paths=/data/acamp/acamp20k/backgrounds/general border=10 aug_seq_prefix=bear_1_1_augmented_mask_backgrounds only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1 static_bkg=3 bkg_size=1280x720

<a id="5_boxes_per_bkg__generalbeer11augment_mask"></a>
#### 5_boxes_per_bkg       @ general/beer_1_1/augment_mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_1_1 mask_paths=/data/acamp/acamp20k/bear\bear_1_1_masks_0x682_10\labels bkg_paths=/data/acamp/acamp20k/backgrounds/general border=10 aug_seq_prefix=bear_1_1_augmented_mask_backgrounds only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1 static_bkg=3 bkg_size=1280x720 boxes_per_bkg=5

<a id="bear_jesse71_w__augment_mask"></a>
## bear_jesse_7_1_w       @ augment_mask

<a id="airport__bear_jesse71_waugment_mask"></a>
### airport       @ bear_jesse_7_1_w/augment_mask

python3 augment_mask.py src_paths=/data/acamp/acamp20k/bear\bear_jesse_7_1_w mask_paths=/data/acamp/acamp20k/masks/bear_jesse_7_1_w_0x0_0/labels bkg_paths=/data/acamp/acamp20k/backgrounds/airport  aug_seq_prefix=bear_1_1_augmented_mask_airport only_one_src_obj=1 aug_seq_size=1000 end_frame_id=100 visualize=1  bkg_size=1280x720 static_bkg=13 map_to_bbox=1 show_bbox=1 show_blended=0

<a id="p1_deer__augment_mask"></a>
## p1_deer       @ augment_mask

<a id="highway__p1_deeraugment_mask"></a>
### highway       @ p1_deer/augment_mask

python3 augment_mask.py src_paths=p1_deer.txt src_root_dir=/data/acamp/acamp20k/deer bkg_paths=/data/acamp/acamp20k/backgrounds/highway only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels aug_seq_prefix=p1_deer src_id=0

<a id="airport__p1_deeraugment_mask"></a>
### airport       @ p1_deer/augment_mask

python3 augment_mask.py src_paths=p1_deer.txt src_root_dir=/data/acamp/acamp20k/deer bkg_paths=/data/acamp/acamp20k/backgrounds/airport only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels save_path=/data/acamp/acamp20k/prototype_1 aug_seq_prefix=p1_deer_airport src_id=0

<a id="p1_coyote__augment_mask"></a>
## p1_coyote       @ augment_mask

<a id="highway__p1_coyoteaugment_mask"></a>
### highway       @ p1_coyote/augment_mask

python3 augment_mask.py src_paths=p1_coyote.txt src_root_dir=/data/acamp/acamp20k/coyote mask_root_dir=/data/acamp/acamp20k/coyote/masks bkg_paths=/data/acamp/acamp20k/backgrounds/highway only_one_src_obj=1 aug_seq_size=1000 visualize=2 bkg_size=1280x720 static_bkg=-1 hist_match_type=0 mask_dir=labels boxes_per_bkg=1 aug_seq_prefix=p1_coyote src_id=0

<a id="airport__p1_coyoteaugment_mask"></a>
### airport       @ p1_coyote/augment_mask

python3 augment_mask.py src_paths=p1_coyote.txt src_root_dir=/data/acamp/acamp20k/coyote bkg_paths=/data/acamp/acamp20k/backgrounds/airport only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels save_path=/data/acamp/acamp20k/prototype_1 aug_seq_prefix=p1_coyote_airport src_id=0

<a id="p1_moose__augment_mask"></a>
## p1_moose       @ augment_mask

<a id="highway__p1_mooseaugment_mask"></a>
### highway       @ p1_moose/augment_mask

python3 augment_mask.py src_paths=p1_moose.txt src_root_dir=/data/acamp/acamp20k/moose bkg_paths=/data/acamp/acamp20k/backgrounds/highway only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels aug_seq_prefix=p1_moose src_id=0

<a id="airport__p1_mooseaugment_mask"></a>
### airport       @ p1_moose/augment_mask

python3 augment_mask.py src_paths=p1_moose.txt src_root_dir=/data/acamp/acamp20k/moose bkg_paths=/data/acamp/acamp20k/backgrounds/airport only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels save_path=/data/acamp/acamp20k/prototype_1 aug_seq_prefix=p1_moose_airport src_id=0

<a id="p1_bear__augment_mask"></a>
## p1_bear       @ augment_mask

<a id="highway__p1_bearaugment_mask"></a>
### highway       @ p1_bear/augment_mask

python3 augment_mask.py src_paths=p1_bear.txt src_root_dir=/data/acamp/acamp20k bkg_paths=/data/acamp/acamp20k/backgrounds/highway only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=0x0_0 mask_dir=labels aug_seq_prefix=p1_bear_highway src_id=0

<a id="airport__p1_bearaugment_mask"></a>
### airport       @ p1_bear/augment_mask

python3 augment_mask.py src_paths=p1_bear.txt src_root_dir=/data/acamp/acamp20k/bear bkg_paths=/data/acamp/acamp20k/backgrounds/airport only_one_src_obj=1 aug_seq_size=1000 visualize=2  bkg_size=1280x720 static_bkg=-1 aug_seq_size=1000 mask_postfix=masks_0x0_0 mask_dir=labels save_path=/data/acamp/acamp20k/prototype_1 aug_seq_prefix=p1_bear_airport src_id=0


<a id="visualize_masks"></a>
# visualize_masks

<a id="bear11__visualize_masks"></a>
### bear_1_1       @ visualize_masks/

<a id="inception_resnet_v2_size_400_max1res_66_233911__bear11visualize_masks"></a>

python3 visualize_masks.py img_paths=bear_1_1 img_root_dir=/data/acamp/acamp20k/bear mask_paths=bear_1_1 mask_paths=/data/acamp/acamp20k/masks/bear_1_1_0x0_0 mask_subdir=labels map_to_bbox=1

<a id="bear_13_seq_103_frames__visualize_masks"></a>
### bear_13_seq_103_frames       @ visualize_masks/

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames.txt img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_190313_1022 mask_subdir=labels map_to_bbox=1 border=0

<a id="bear_13_seq_103_frames_ex1__visualize_masks"></a>
### bear_13_seq_103_frames_ex1       @ visualize_masks/

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames_ex1.txt img_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_ex1 mask_subdir=labels

python3 visualize_masks.py img_paths=../tf_api/bear_13_seq_103_frames_ex1.txt img_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_ex1 mask_subdir=labels

<a id="0x682_10__bear_13_seq_103_frames_ex1visualize_masks"></a>
#### 0x682_10       @ bear_13_seq_103_frames_ex1/visualize_masks/

python3 visualize_masks.py img_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_ex1_0x682_10  img_subdir=images mask_subdir=labels border=10

<a id="0x682_0__bear_13_seq_103_frames_ex1visualize_masks"></a>
#### 0x682_0       @ bear_13_seq_103_frames_ex1/visualize_masks/

python3 visualize_masks.py img_root_dir=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_ex1_0x682_0  img_subdir=images mask_subdir=labels border=0

