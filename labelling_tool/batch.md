<!-- MarkdownTOC -->

- [labelImg](#labelim_g_)
- [xml_to_csv](#xml_to_cs_v_)
- [to_csv](#to_cs_v_)
    - [10k       @ to_csv](#10k___to_csv_)
    - [20k       @ to_csv](#20k___to_csv_)
    - [20k7_new       @ to_csv](#20k7_new___to_csv_)
    - [25k7_new       @ to_csv](#25k7_new___to_csv_)
        - [test_0709       @ 25k7_new/to_csv](#test_0709___25k7_new_to_cs_v_)
    - [cows       @ to_csv](#cows___to_csv_)
    - [horses       @ to_csv](#horses___to_csv_)
        - [horse_16       @ horses/to_csv](#horse_16___horses_to_cs_v_)
        - [horse_24       @ horses/to_csv](#horse_24___horses_to_cs_v_)
        - [horse_25       @ horses/to_csv](#horse_25___horses_to_cs_v_)
    - [bear       @ to_csv](#bear___to_csv_)
        - [bear_1_1       @ bear/to_csv](#bear_1_1___bear_to_cs_v_)
        - [bear_1_1_even_10       @ bear/to_csv](#bear_1_1_even_10___bear_to_cs_v_)
    - [bison       @ to_csv](#bison___to_csv_)
        - [bison_60_w       @ bison/to_csv](#bison_60_w___bison_to_csv_)
        - [bison_42_w       @ bison/to_csv](#bison_42_w___bison_to_csv_)
    - [elk       @ to_csv](#elk___to_csv_)
    - [airport       @ to_csv](#airport___to_csv_)
    - [highway       @ to_csv](#highway___to_csv_)
    - [coyote_10_5       @ to_csv](#coyote_10_5___to_csv_)
    - [coyote_b       @ to_csv](#coyote_b___to_csv_)
    - [p1_highway       @ to_csv](#p1_highway___to_csv_)
    - [prototype_1_vid       @ to_csv](#prototype_1_vid___to_csv_)
- [augment](#augment_)
    - [deer       @ augment](#deer___augmen_t_)
        - [grs       @ deer/augment](#grs___deer_augment_)
        - [deer_jesse_static_1       @ deer/augment](#deer_jesse_static_1___deer_augment_)
        - [acamp20k_static_deer       @ deer/augment](#acamp20k_static_deer___deer_augment_)
    - [coyote       @ augment](#coyote___augmen_t_)
        - [grs       @ coyote/augment](#grs___coyote_augment_)
    - [bear       @ augment](#bear___augmen_t_)
        - [grs       @ bear/augment](#grs___bear_augment_)
    - [p1_deer       @ augment](#p1_deer___augmen_t_)
        - [highway       @ p1_deer/augment](#highway___p1_deer_augmen_t_)
            - [no_mask       @ highway/p1_deer/augment](#no_mask___highway_p1_deer_augmen_t_)
            - [mask_rcnn_resnet101       @ highway/p1_deer/augment](#mask_rcnn_resnet101___highway_p1_deer_augmen_t_)
            - [siam_mask       @ highway/p1_deer/augment](#siam_mask___highway_p1_deer_augmen_t_)
            - [siam_mask_davis       @ highway/p1_deer/augment](#siam_mask_davis___highway_p1_deer_augmen_t_)
        - [airport       @ p1_deer/augment](#airport___p1_deer_augmen_t_)
            - [no_mask       @ airport/p1_deer/augment](#no_mask___airport_p1_deer_augmen_t_)
            - [mask_rcnn_resnet101       @ airport/p1_deer/augment](#mask_rcnn_resnet101___airport_p1_deer_augmen_t_)
            - [siam_mask       @ airport/p1_deer/augment](#siam_mask___airport_p1_deer_augmen_t_)
            - [siam_mask_davis       @ airport/p1_deer/augment](#siam_mask_davis___airport_p1_deer_augmen_t_)
    - [p1_coyote       @ augment](#p1_coyote___augmen_t_)
        - [highway       @ p1_coyote/augment](#highway___p1_coyote_augmen_t_)
            - [no_mask       @ highway/p1_coyote/augment](#no_mask___highway_p1_coyote_augmen_t_)
            - [mask_rcnn_resnet101       @ highway/p1_coyote/augment](#mask_rcnn_resnet101___highway_p1_coyote_augmen_t_)
            - [siam_mask       @ highway/p1_coyote/augment](#siam_mask___highway_p1_coyote_augmen_t_)
            - [siam_mask_davis       @ highway/p1_coyote/augment](#siam_mask_davis___highway_p1_coyote_augmen_t_)
        - [airport       @ p1_coyote/augment](#airport___p1_coyote_augmen_t_)
            - [no_mask       @ airport/p1_coyote/augment](#no_mask___airport_p1_coyote_augmen_t_)
            - [mask_rcnn_resnet101       @ airport/p1_coyote/augment](#mask_rcnn_resnet101___airport_p1_coyote_augmen_t_)
            - [siam_mask       @ airport/p1_coyote/augment](#siam_mask___airport_p1_coyote_augmen_t_)
            - [siam_mask_davis       @ airport/p1_coyote/augment](#siam_mask_davis___airport_p1_coyote_augmen_t_)
    - [p1_bear       @ augment](#p1_bear___augmen_t_)
        - [highway       @ p1_bear/augment](#highway___p1_bear_augmen_t_)
            - [no_mask       @ highway/p1_bear/augment](#no_mask___highway_p1_bear_augmen_t_)
            - [mask_rcnn_resnet101       @ highway/p1_bear/augment](#mask_rcnn_resnet101___highway_p1_bear_augmen_t_)
            - [siam_mask       @ highway/p1_bear/augment](#siam_mask___highway_p1_bear_augmen_t_)
            - [siam_mask_davis       @ highway/p1_bear/augment](#siam_mask_davis___highway_p1_bear_augmen_t_)
        - [airport       @ p1_bear/augment](#airport___p1_bear_augmen_t_)
            - [no_mask       @ airport/p1_bear/augment](#no_mask___airport_p1_bear_augmen_t_)
            - [mask_rcnn_resnet101       @ airport/p1_bear/augment](#mask_rcnn_resnet101___airport_p1_bear_augmen_t_)
            - [siam_mask       @ airport/p1_bear/augment](#siam_mask___airport_p1_bear_augmen_t_)
            - [siam_mask_davis       @ airport/p1_bear/augment](#siam_mask_davis___airport_p1_bear_augmen_t_)
    - [p1_moose       @ augment](#p1_moose___augmen_t_)
        - [highway       @ p1_moose/augment](#highway___p1_moose_augment_)
            - [mask_rcnn_resnet101       @ highway/p1_moose/augment](#mask_rcnn_resnet101___highway_p1_moose_augment_)
            - [siam_mask       @ highway/p1_moose/augment](#siam_mask___highway_p1_moose_augment_)
            - [siam_mask_davis       @ highway/p1_moose/augment](#siam_mask_davis___highway_p1_moose_augment_)
        - [airport       @ p1_moose/augment](#airport___p1_moose_augment_)
            - [mask_rcnn_resnet101       @ airport/p1_moose/augment](#mask_rcnn_resnet101___airport_p1_moose_augment_)
            - [siam_mask       @ airport/p1_moose/augment](#siam_mask___airport_p1_moose_augment_)
            - [siam_mask_davis       @ airport/p1_moose/augment](#siam_mask_davis___airport_p1_moose_augment_)
- [visualize](#visualize_)
    - [5k       @ visualize](#5k___visualiz_e_)
    - [10k       @ visualize](#10k___visualiz_e_)
    - [10k_ar       @ visualize](#10k_ar___visualiz_e_)
    - [10kh       @ visualize](#10kh___visualiz_e_)
    - [20k       @ visualize](#20k___visualiz_e_)
        - [coyote       @ 20k/visualize](#coyote___20k_visualiz_e_)
        - [bear       @ 20k/visualize](#bear___20k_visualiz_e_)
        - [deer       @ 20k/visualize](#deer___20k_visualiz_e_)
            - [deer_jesse_9_       @ deer/20k/visualize](#deer_jesse_9___deer_20k_visualize_)
        - [moose       @ 20k/visualize](#moose___20k_visualiz_e_)
        - [bison       @ 20k/visualize](#bison___20k_visualiz_e_)
        - [horse       @ 20k/visualize](#horse___20k_visualiz_e_)
        - [bear_1_1       @ 20k/visualize](#bear_1_1___20k_visualiz_e_)
            - [masks_0x1365_10       @ bear_1_1/20k/visualize](#masks_0x1365_10___bear_1_1_20k_visualize_)
            - [masks_0x0_0p6       @ bear_1_1/20k/visualize](#masks_0x0_0p6___bear_1_1_20k_visualize_)
            - [bear_1_1_masks_448x448_ar_1p0       @ bear_1_1/20k/visualize](#bear_1_1_masks_448x448_ar_1p0___bear_1_1_20k_visualize_)
        - [coyote_9_seq_54_frames       @ 20k/visualize](#coyote_9_seq_54_frames___20k_visualiz_e_)
        - [deer_11_seq_56_frames       @ 20k/visualize](#deer_11_seq_56_frames___20k_visualiz_e_)
        - [moose_10_seq_50_frames       @ 20k/visualize](#moose_10_seq_50_frames___20k_visualiz_e_)
        - [bear_13_seq_103_frames_224x224_5       @ 20k/visualize](#bear_13_seq_103_frames_224x224_5___20k_visualiz_e_)
        - [bear_13_seq_103_frames_224x224_5_test       @ 20k/visualize](#bear_13_seq_103_frames_224x224_5_test___20k_visualiz_e_)
        - [bear_13_seq_103_frames       @ 20k/visualize](#bear_13_seq_103_frames___20k_visualiz_e_)
        - [test_0709       @ 20k/visualize](#test_0709___20k_visualiz_e_)
    - [results       @ visualize](#results___visualiz_e_)
    - [prototype_1       @ visualize](#prototype_1___visualiz_e_)
        - [source       @ prototype_1/visualize](#source___prototype_1_visualiz_e_)
            - [coyote_9_1,moose_jesse_2_2_w       @ source/prototype_1/visualize](#coyote_9_1_moose_jesse_2_2_w___source_prototype_1_visualize_)
            - [individual       @ source/prototype_1/visualize](#individual___source_prototype_1_visualize_)
            - [stack       @ source/prototype_1/visualize](#stack___source_prototype_1_visualize_)
            - [cropped       @ source/prototype_1/visualize](#cropped___source_prototype_1_visualize_)
                - [coyote       @ cropped/source/prototype_1/visualize](#coyote___cropped_source_prototype_1_visualize_)
                - [moose       @ cropped/source/prototype_1/visualize](#moose___cropped_source_prototype_1_visualize_)
        - [bear_airport       @ prototype_1/visualize](#bear_airport___prototype_1_visualiz_e_)
            - [no_mask_n_vis_2       @ bear_airport/prototype_1/visualize](#no_mask_n_vis_2___bear_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_n_vis_2       @ bear_airport/prototype_1/visualize](#mask_rcnn_resnet101_n_vis_2___bear_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_n_vis_3       @ bear_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_n_vis_3___bear_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4       @ bear_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4___bear_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ bear_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___bear_airport_prototype_1_visualize_)
        - [bear_highway       @ prototype_1/visualize](#bear_highway___prototype_1_visualiz_e_)
            - [no_mask_n_vis_2       @ bear_highway/prototype_1/visualize](#no_mask_n_vis_2___bear_highway_prototype_1_visualize_)
            - [mask_rcnn_resnet101_n_vis_2       @ bear_highway/prototype_1/visualize](#mask_rcnn_resnet101_n_vis_2___bear_highway_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ bear_highway/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___bear_highway_prototype_1_visualize_)
        - [deer_airport       @ prototype_1/visualize](#deer_airport___prototype_1_visualiz_e_)
            - [no_mask_n_vis_2       @ deer_airport/prototype_1/visualize](#no_mask_n_vis_2___deer_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_n_vis_2       @ deer_airport/prototype_1/visualize](#mask_rcnn_resnet101_n_vis_2___deer_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_n_vis_3       @ deer_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_n_vis_3___deer_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4       @ deer_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4___deer_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ deer_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___deer_airport_prototype_1_visualize_)
        - [deer_highway       @ prototype_1/visualize](#deer_highway___prototype_1_visualiz_e_)
            - [no_mask_n_vis_2       @ deer_highway/prototype_1/visualize](#no_mask_n_vis_2___deer_highway_prototype_1_visualize_)
            - [mask_rcnn_resnet101_n_vis_2       @ deer_highway/prototype_1/visualize](#mask_rcnn_resnet101_n_vis_2___deer_highway_prototype_1_visualize_)
        - [coyote_airport       @ prototype_1/visualize](#coyote_airport___prototype_1_visualiz_e_)
            - [no_mask_n_vis_2       @ coyote_airport/prototype_1/visualize](#no_mask_n_vis_2___coyote_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4       @ coyote_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4___coyote_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ coyote_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___coyote_airport_prototype_1_visualize_)
            - [mask_rcnn_resnet101_n_vis_2       @ coyote_airport/prototype_1/visualize](#mask_rcnn_resnet101_n_vis_2___coyote_airport_prototype_1_visualize_)
        - [coyote_highway       @ prototype_1/visualize](#coyote_highway___prototype_1_visualiz_e_)
            - [no_mask_n_vis_2       @ coyote_highway/prototype_1/visualize](#no_mask_n_vis_2___coyote_highway_prototype_1_visualize_)
            - [mask_rcnn_resnet101_n_vis_2       @ coyote_highway/prototype_1/visualize](#mask_rcnn_resnet101_n_vis_2___coyote_highway_prototype_1_visualize_)
            - [mask_rcnn_resnet101_no_mask_n_vis_3       @ coyote_highway/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_n_vis_3___coyote_highway_prototype_1_visualize_)
        - [moose_airport       @ prototype_1/visualize](#moose_airport___prototype_1_visualiz_e_)
            - [no_mask_n_vis_2       @ moose_airport/prototype_1/visualize](#no_mask_n_vis_2___moose_airport_prototype_1_visualiz_e_)
            - [mask_rcnn_resnet101_n_vis_2       @ moose_airport/prototype_1/visualize](#mask_rcnn_resnet101_n_vis_2___moose_airport_prototype_1_visualiz_e_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4       @ moose_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4___moose_airport_prototype_1_visualiz_e_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ moose_airport/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___moose_airport_prototype_1_visualiz_e_)
        - [moose_highway       @ prototype_1/visualize](#moose_highway___prototype_1_visualiz_e_)
            - [no_mask_n_vis_2       @ moose_highway/prototype_1/visualize](#no_mask_n_vis_2___moose_highway_prototype_1_visualiz_e_)
            - [mask_rcnn_resnet101_n_vis_2       @ moose_highway/prototype_1/visualize](#mask_rcnn_resnet101_n_vis_2___moose_highway_prototype_1_visualiz_e_)
            - [mask_rcnn_resnet101_no_mask_n_vis_3       @ moose_highway/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_n_vis_3___moose_highway_prototype_1_visualiz_e_)
            - [mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ moose_highway/prototype_1/visualize](#mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___moose_highway_prototype_1_visualiz_e_)

<!-- /MarkdownTOC -->

<a id="labelim_g_"></a>
# labelImg

python3 labelImg.py --load_prev=1 --mask_disp_size=1000,1000 --mask_magnified_window=1 --mask_show_pts=0 --mask_border_size=10,10

python3 labelImg.py --h

python3 labelImg.py --load_prev=1 --mask.disp_size=1920,1020 --mask.show_magnified_window=1 --mask.show_pts=0 --mask.border_size=15,15 --mask.mag_patch_size=80 --mask.mag_win_size=800

<a id="xml_to_cs_v_"></a>
# xml_to_csv

python xml_to_csv.py base_path=N:\Datasets\Acamp\marcin_180613\images seq_name=static_1

<a id="to_cs_v_"></a>
# to_csv

python3 to_csv.py seq_paths="N:\Datasets\Acamp\marcin_180615\list.txt" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 to_csv.py root_dir="N:\Datasets\Acamp\test_data_annotations" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 to_csv.py seq_paths="N:\Datasets\Acamp\marcin_180613\images\deer\deer_2_5m" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 to_csv.py seq_paths="N:\Datasets\Acamp\acamp20k/deer_jesse_7_4" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 to_csv.py seq_paths="N:\Datasets\Acamp\backgrounds" class_names_path=../labelling_tool/data/predefined_classes_bkg.txt

<a id="10k___to_csv_"></a>
## 10k       @ to_csv

python3 to_csv.py seq_paths=N:\Datasets\Acamp\acamp10k\animals class_names_path=data/predefined_classes_10k.txt

python3 to_csv.py seq_paths="N:\Datasets\Acamp\acamp10k\test\images\moose_21_1" class_names_path=data/predefined_classes_10k.txt

python3 to_csv.py seq_paths="N:\Datasets\Acamp\acamp10k\test\images\moose_21_1" class_names_path=data/predefined_classes_10k.txt

python3 to_csv.py seq_paths=../tf_api/acamp_all_bear.txt root_dir=/data/acamp/acamp20k/bear class_names_path=data/predefined_classes_10k.txt

<a id="20k___to_csv_"></a>
## 20k       @ to_csv

python3 to_csv.py seq_paths=N:\Datasets\Acamp\acamp20k class_names_path=data/predefined_classes_10k.txt

python3 to_csv.py seq_paths=../tf_api/acamp20k_test_180719.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=data/predefined_classes_10k.txt

python3 to_csv.py seq_paths=../tf_api/acamp20k_test_180720.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=data/predefined_classes_10k.txt

python3 to_csv.py seq_paths=../tf_api/acamp20k_test_180720.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=data/predefined_classes_10k.txt

python3 to_csv.py seq_paths=../tf_api/acamp20k_test_180727.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=data/predefined_classes_10k.txt

<a id="20k7_new___to_csv_"></a>
## 20k7_new       @ to_csv

python3 to_csv.py seq_paths=../tf_api/acamp20k7_new_train.txt root_dir=/data/acamp/acamp20k class_names_path=data/predefined_classes_20k7.txt

<a id="25k7_new___to_csv_"></a>
## 25k7_new       @ to_csv

python3 to_csv.py seq_paths=../tf_api/acamp25k7_train_new.txt root_dir=/data/acamp/acamp20k class_names_path=data/predefined_classes_20k7.txt

<a id="test_0709___25k7_new_to_cs_v_"></a>
### test_0709       @ 25k7_new/to_csv

python3 to_csv.py seq_paths=acamp20k_test_0709.txt class_names_path=data/predefined_classes_10k.txt root_dir=N:\Datasets\Acamp\acamp20k

python3 to_csv.py seq_paths=acamp20k_test_0709.txt class_names_path=data/predefined_classes_10k.txt  root_dir=/data/acamp/acamp20k

<a id="cows___to_csv_"></a>
## cows       @ to_csv

python3 to_csv.py seq_paths=acamp_cows.txt class_names_path=data/predefined_classes_cow.txt root_dir=/data/acamp/acamp20k

<a id="horses___to_csv_"></a>
## horses       @ to_csv

python3 to_csv.py seq_paths=acamp_horses.txt class_names_path=data/predefined_classes_horse.txt root_dir=/data/acamp/acamp20k

<a id="horse_16___horses_to_cs_v_"></a>
### horse_16       @ horses/to_csv

python3 to_csv.py seq_paths=horse_16 class_names_path=data/predefined_classes_horse.txt root_dir=/data/acamp/acamp20k

<a id="horse_24___horses_to_cs_v_"></a>
### horse_24       @ horses/to_csv

python3 to_csv.py seq_paths=horse_24 class_names_path=data/predefined_classes_horse.txt root_dir=/data/acamp/acamp20k

<a id="horse_25___horses_to_cs_v_"></a>
### horse_25       @ horses/to_csv

python3 to_csv.py seq_paths=horse_25 class_names_path=data/predefined_classes_horse.txt root_dir=/data/acamp/acamp20k

<a id="bear___to_csv_"></a>
## bear       @ to_csv

<a id="bear_1_1___bear_to_cs_v_"></a>
### bear_1_1       @ bear/to_csv

python3 to_csv.py seq_paths=bear_1_1 class_names_path=data/predefined_classes.txt root_dir=/data/acamp/acamp20k enable_mask=1

<a id="bear_1_1_even_10___bear_to_cs_v_"></a>
### bear_1_1_even_10       @ bear/to_csv

python3 to_csv.py seq_paths=bear_1_1_even_10 class_names_path=data/predefined_classes.txt root_dir=/data/acamp/acamp20k/bear

<a id="bison___to_csv_"></a>
## bison       @ to_csv

python3 to_csv.py class_names_path=data/predefined_classes.txt root_dir=/data/acamp/acamp20k/bison

<a id="bison_60_w___bison_to_csv_"></a>
### bison_60_w       @ bison/to_csv

python3 to_csv.py seq_paths=bison_60_w class_names_path=data/predefined_classes.txt root_dir=/data/acamp/acamp20k/bison

<a id="bison_42_w___bison_to_csv_"></a>
### bison_42_w       @ bison/to_csv

python3 to_csv.py seq_paths=bison_42_w class_names_path=data/predefined_classes.txt root_dir=/data/acamp/acamp20k/bison


<a id="elk___to_csv_"></a>
## elk       @ to_csv

python3 to_csv.py class_names_path=data/predefined_classes.txt root_dir=/data/acamp/acamp20k/elk

<a id="airport___to_csv_"></a>
## airport       @ to_csv

python3 to_csv.py class_names_path=data/predefined_classes_bear.txt seq_paths=/data/acamp/acamp20k/backgrounds\airport

<a id="highway___to_csv_"></a>
## highway       @ to_csv

python3 to_csv.py class_names_path=data/predefined_classes_bear.txt seq_paths=/data/acamp/acamp20k/backgrounds\highway

<a id="coyote_10_5___to_csv_"></a>
## coyote_10_5       @ to_csv

python3 to_csv.py class_names_path=data/predefined_classes.txt seq_paths=/data/acamp/acamp20k/coyote\coyote_10_5

<a id="coyote_b___to_csv_"></a>
## coyote_b       @ to_csv

python3 to_csv.py class_names_path=data/predefined_classes.txt seq_paths=/data/acamp/acamp20k/coyote\coyote_b


<a id="p1_highway___to_csv_"></a>
## p1_highway       @ to_csv

python3 to_csv.py class_names_path=data/predefined_classes.txt root_dir=/data/acamp/acamp20k/prototype_1

<a id="prototype_1_vid___to_csv_"></a>
## prototype_1_vid       @ to_csv

python3 to_csv.py class_names_path=data/predefined_classes.txt root_dir=/data/acamp/acamp20k/prototype_1_vid

<a id="augment_"></a>
# augment

<a id="deer___augmen_t_"></a>
## deer       @ augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/deer src_paths=../tf_api/acamp20k_static_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/general border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=deer_augmented only_one_src_obj=1 aug_seq_size=1000

<a id="grs___deer_augment_"></a>
### grs       @ deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k src_paths=../tf_api/acamp20k_static_deer.txt bkg_path=/data/acamp/backgrounds border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=deer_augmented only_one_src_obj=1 aug_seq_size=1000

<a id="deer_jesse_static_1___deer_augment_"></a>
### deer_jesse_static_1       @ deer/augment

python3 augment.py src_root_dir=N:\Datasets\Acamp\acamp20k/#static\deer_jesse_static_1 bkg_paths=N:\Datasets\Acamp\backgrounds border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0

<a id="acamp20k_static_deer___deer_augment_"></a>
### acamp20k_static_deer       @ deer/augment

python3 augment.py src_root_dir=N:\Datasets\Acamp\acamp20k/#static src_paths=../tf_api/acamp20k_static_deer.txt bkg_paths=N:\Datasets\Acamp\backgrounds border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=deer_augmented only_one_src_obj=1 aug_seq_size=100

<a id="coyote___augmen_t_"></a>
## coyote       @ augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/coyote src_paths=../tf_api/acamp20k_static_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/general border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=coyote_augmented only_one_src_obj=1 aug_seq_size=1000

<a id="grs___coyote_augment_"></a>
### grs       @ coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k src_paths=../tf_api/acamp20k_static_coyote.txt bkg_path=/data/acamp/backgrounds border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=coyote_augmented only_one_src_obj=1 aug_seq_size=1000

<a id="bear___augmen_t_"></a>
## bear       @ augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/bear src_paths=../tf_api/acamp20k_static_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/general border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=bear_augmented only_one_src_obj=1 aug_seq_size=1000

<a id="grs___bear_augment_"></a>
### grs       @ bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k src_paths=../tf_api/acamp20k_static_bear.txt bkg_path=/data/acamp/backgrounds border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=bear_augmented only_one_src_obj=1 aug_seq_size=1000
 
<a id="p1_deer___augmen_t_"></a>
## p1_deer       @ augment

<a id="highway___p1_deer_augmen_t_"></a>
### highway       @ p1_deer/augment

<a id="no_mask___highway_p1_deer_augmen_t_"></a>
#### no_mask       @ highway/p1_deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=p1_deer_h_no_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_deer_h_1/annotations.csv flip_lr_prob=-1 visualize=0 sample_frg_per_bkg=3

<a id="mask_rcnn_resnet101___highway_p1_deer_augmen_t_"></a>
#### mask_rcnn_resnet101       @ highway/p1_deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_deer_h_mask_rcnn_resnet101 only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_deer_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/mask_rcnn/resnet101_deer_11_seq_56_frames_max_1_size_400_res_60_0x682_1_249451_grs_190710_105634 visualize=0 mask_border=1

<a id="siam_mask___highway_p1_deer_augmen_t_"></a>
#### siam_mask       @ highway/p1_deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_deer_h_siam_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_deer_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask/deer visualize=0 mask_border=0 mask_threshold=0

<a id="siam_mask_davis___highway_p1_deer_augmen_t_"></a>
#### siam_mask_davis       @ highway/p1_deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_deer_h_siam_mask_davis only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_deer_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask_davis/deer visualize=0 mask_border=0 mask_threshold=0

<a id="airport___p1_deer_augmen_t_"></a>
### airport       @ p1_deer/augment

<a id="no_mask___airport_p1_deer_augmen_t_"></a>
#### no_mask       @ airport/p1_deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=p1_deer_a_no_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_deer_a_1/annotations.csv flip_lr_prob=-1 visualize=0 sample_frg_per_bkg=3

<a id="mask_rcnn_resnet101___airport_p1_deer_augmen_t_"></a>
#### mask_rcnn_resnet101       @ airport/p1_deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_deer_a_mask_rcnn_resnet101 only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_deer_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/mask_rcnn/resnet101_deer_11_seq_56_frames_max_1_size_400_res_60_0x682_1_249451_grs_190710_105634 visualize=0 mask_border=1

<a id="siam_mask___airport_p1_deer_augmen_t_"></a>
#### siam_mask       @ airport/p1_deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_deer_a_siam_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_deer_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask/deer visualize=0 mask_border=0 mask_threshold=0

<a id="siam_mask_davis___airport_p1_deer_augmen_t_"></a>
#### siam_mask_davis       @ airport/p1_deer/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_deer.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_deer_a_siam_mask_davis only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_deer_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask_davis/deer visualize=0 mask_border=0 mask_threshold=0

<a id="p1_coyote___augmen_t_"></a>
## p1_coyote       @ augment

<a id="highway___p1_coyote_augmen_t_"></a>
### highway       @ p1_coyote/augment

<a id="no_mask___highway_p1_coyote_augmen_t_"></a>
#### no_mask       @ highway/p1_coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=p1_coyote_h_no_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_coyote_h_1/annotations.csv flip_lr_prob=-1 sample_frg_per_bkg=3

<a id="mask_rcnn_resnet101___highway_p1_coyote_augmen_t_"></a>
#### mask_rcnn_resnet101       @ highway/p1_coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_coyote_h_mask_rcnn_resnet101 only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_coyote_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/mask_rcnn/resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1_439063_z370_190710_105926 visualize=0 mask_border=1

<a id="siam_mask___highway_p1_coyote_augmen_t_"></a>
#### siam_mask       @ highway/p1_coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_coyote_h_siam_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_coyote_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask/coyote visualize=0 mask_border=0 mask_threshold=0

<a id="siam_mask_davis___highway_p1_coyote_augmen_t_"></a>
#### siam_mask_davis       @ highway/p1_coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_coyote_h_siam_mask_davis only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_coyote_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask_davis/coyote visualize=0 mask_border=0 mask_threshold=0

<a id="airport___p1_coyote_augmen_t_"></a>
### airport       @ p1_coyote/augment

<a id="no_mask___airport_p1_coyote_augmen_t_"></a>
#### no_mask       @ airport/p1_coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=p1_coyote_a_no_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_coyote_a_1/annotations.csv flip_lr_prob=-1 sample_frg_per_bkg=3

<a id="mask_rcnn_resnet101___airport_p1_coyote_augmen_t_"></a>
#### mask_rcnn_resnet101       @ airport/p1_coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_coyote_a_mask_rcnn_resnet101 only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_coyote_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/mask_rcnn/resnet101_coyote_9_seq_54_frames_max_1_size_400_res_60_0x682_1_439063_z370_190710_105926 visualize=0 mask_border=1

<a id="siam_mask___airport_p1_coyote_augmen_t_"></a>
#### siam_mask       @ airport/p1_coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_coyote_a_siam_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_coyote_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask/coyote visualize=0 mask_border=0 mask_threshold=0

<a id="siam_mask_davis___airport_p1_coyote_augmen_t_"></a>
#### siam_mask_davis       @ airport/p1_coyote/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_coyote.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_coyote_a_siam_mask_davis only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_coyote_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask_davis/coyote visualize=0 mask_border=0 mask_threshold=0

<a id="p1_bear___augmen_t_"></a>
## p1_bear       @ augment

<a id="highway___p1_bear_augmen_t_"></a>
### highway       @ p1_bear/augment

<a id="no_mask___highway_p1_bear_augmen_t_"></a>
#### no_mask       @ highway/p1_bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=p1_bear_h_no_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_bear_h_1/annotations.csv flip_lr_prob=-1 visualize=0 sample_frg_per_bkg=3

<a id="mask_rcnn_resnet101___highway_p1_bear_augmen_t_"></a>
#### mask_rcnn_resnet101       @ highway/p1_bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_bear_h_mask_rcnn_resnet101 only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_bear_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/mask_rcnn/resnet101_bear_13_seq_103_frames_ex1_max_1_size_400_res_60_0x682_1_110431_z370_190701_083521 visualize=0 mask_border=1

<a id="siam_mask___highway_p1_bear_augmen_t_"></a>
#### siam_mask       @ highway/p1_bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_bear_h_siam_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_bear_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask/bear visualize=0 mask_border=0 mask_threshold=0

<a id="siam_mask_davis___highway_p1_bear_augmen_t_"></a>
#### siam_mask_davis       @ highway/p1_bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_bear_h_siam_mask_davis only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_bear_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask_davis/bear visualize=0 mask_border=0 mask_threshold=0

<a id="airport___p1_bear_augmen_t_"></a>
### airport       @ p1_bear/augment

<a id="no_mask___airport_p1_bear_augmen_t_"></a>
#### no_mask       @ airport/p1_bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=p1_bear_a_no_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_bear_a_1/annotations.csv flip_lr_prob=-1 sample_frg_per_bkg=3

<a id="mask_rcnn_resnet101___airport_p1_bear_augmen_t_"></a>
#### mask_rcnn_resnet101       @ airport/p1_bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_bear_a_mask_rcnn_resnet101 only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_bear_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/mask_rcnn/resnet101_bear_13_seq_103_frames_ex1_max_1_size_400_res_60_0x682_1_110431_z370_190701_083521 visualize=0 mask_border=1

<a id="siam_mask___airport_p1_bear_augmen_t_"></a>
#### siam_mask       @ airport/p1_bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_bear_a_siam_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_bear_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask/bear visualize=0 mask_border=0 mask_threshold=0

<a id="siam_mask_davis___airport_p1_bear_augmen_t_"></a>
#### siam_mask_davis       @ airport/p1_bear/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_bear.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_bear_a_siam_mask_davis only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_bear_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask_davis/bear visualize=0 mask_border=0 mask_threshold=0

<a id="p1_moose___augmen_t_"></a>
## p1_moose       @ augment

<a id="highway___p1_moose_augment_"></a>
### highway       @ p1_moose/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_moose.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=p1_moose_h_no_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_moose_h_1/annotations.csv flip_lr_prob=-1 sample_frg_per_bkg=3

<a id="mask_rcnn_resnet101___highway_p1_moose_augment_"></a>
#### mask_rcnn_resnet101       @ highway/p1_moose/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_moose.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_moose_h_mask_rcnn_resnet101 only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_moose_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/mask_rcnn/resnet101_moose_10_seq_50_frames_max_1_size_400_res_60_0x682_1_261401_grs_190710_105652 visualize=0 mask_border=1

<a id="siam_mask___highway_p1_moose_augment_"></a>
#### siam_mask       @ highway/p1_moose/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_moose.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_moose_h_siam_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_moose_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask/moose visualize=0 mask_border=0 mask_threshold=0

<a id="siam_mask_davis___highway_p1_moose_augment_"></a>
#### siam_mask_davis       @ highway/p1_moose/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_moose.txt bkg_path=/data/acamp/acamp20k/backgrounds/highway border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_moose_h_siam_mask_davis only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_moose_h_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask_davis/moose visualize=0 mask_border=0 mask_threshold=0

<a id="airport___p1_moose_augment_"></a>
### airport       @ p1_moose/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_moose.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0.05 make_square=1 mask_type=1 bkg_iou_thresh=0.0 aug_seq_prefix=p1_moose_a_no_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_moose_a_1/annotations.csv flip_lr_prob=-1 sample_frg_per_bkg=3

<a id="mask_rcnn_resnet101___airport_p1_moose_augment_"></a>
#### mask_rcnn_resnet101       @ airport/p1_moose/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_moose.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_moose_a_mask_rcnn_resnet101 only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_moose_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/mask_rcnn/resnet101_moose_10_seq_50_frames_max_1_size_400_res_60_0x682_1_261401_grs_190710_105652 visualize=0 mask_border=1

<a id="siam_mask___airport_p1_moose_augment_"></a>
#### siam_mask       @ airport/p1_moose/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_moose.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_moose_a_siam_mask only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_moose_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask/moose visualize=0 mask_border=0 mask_threshold=0

<a id="siam_mask_davis___airport_p1_moose_augment_"></a>
#### siam_mask_davis       @ airport/p1_moose/augment

python3 augment.py src_root_dir=/data/acamp/acamp20k/prototype_1_source src_paths=p1_moose.txt bkg_path=/data/acamp/acamp20k/backgrounds/airport border_ratio=0 make_square=0 mask_type=4 bkg_iou_thresh=0.0 aug_seq_prefix=p1_moose_a_siam_mask_davis only_one_src_obj=1 aug_seq_size=1000 n_aug=0 inclue_src_in_augmented=0 load_bkg=0 bkg_bboxes_csv=/data/acamp/acamp20k/prototype_1/p1_moose_a_1/annotations.csv flip_lr_prob=-1 mask_root_dir=/results/siam_mask_davis/moose visualize=0 mask_border=0 mask_threshold=0

<a id="visualize_"></a>
# visualize

python3 visualize.py seq_paths="N:\Datasets\Acamp\marcin_180615\list.txt" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 visualize.py seq_paths="N:\Datasets\Acamp\marcin_180613\images\list.txt" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 visualize.py seq_paths="N:\Datasets\Acamp\marcin_180613\images\deer\deer_2_5m" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 visualize.py seq_paths="N:\Datasets\Acamp\acamp20k/#static\deer_augmented_1" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 visualize.py seq_paths=N:\Datasets\Acamp\test_data_annotations class_names_path=../labelling_tool/data/predefined_classes_orig.txt vis.save=1 vis.save_fmt=jpg,'XVID',30 vis.save_prefix=vis

<a id="5k___visualiz_e_"></a>
## 5k       @ visualize

python3 visualize.py seq_paths="N:\Datasets\Acamp\acamp5k\train\images\bear" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 visualize.py seq_paths="N:\Datasets\Acamp\acamp5k\test\images\bear" class_names_path=../labelling_tool/data/predefined_classes_orig.txt

<a id="10k___visualiz_e_"></a>
## 10k       @ visualize

python3 visualize.py root_dir="N:\Datasets\Acamp\acamp10k" class_names_path=data/predefined_classes_10k.txt

python3 visualize.py root_dir=N:\Datasets\Acamp\acamp10k\test\images class_names_path=data/predefined_classes_10k.txt

python3 visualize.py root_dir=N:\Datasets\Acamp\acamp10k\train class_names_path=data/predefined_classes_10k.txt

python3 visualize.py seq_paths=acamp10k_train_biwi.txt class_names_path=data/predefined_classes_10k.txt

python3 visualize.py seq_paths=acamp10k_train_caltech.txt class_names_path=data/predefined_classes_10k.txt

python3 visualize.py seq_paths=acamp10k_train_inria.txt class_names_path=data/predefined_classes_10k.txt

python3 visualize.py seq_paths=acamp10k_train_mot.txt class_names_path=data/predefined_classes_10k.txt

python3 visualize.py seq_paths=acamp10k_train_daimler.txt class_names_path=data/predefined_classes_10k.txt

<a id="10k_ar___visualiz_e_"></a>
## 10k_ar       @ visualize


python3 visualize.py root_dir=N:\Datasets\Acamp\acamp10k\train_ar class_names_path=data/predefined_classes_10k.txt

<a id="10kh___visualiz_e_"></a>
## 10kh       @ visualize

python3 visualize.py seq_paths="H:\UofA\Acamp\code\mAP\acamp10kh_test_images.txt" class_names_path=data/predefined_classes_10k.txt

python3 visualize.py seq_paths="H:\UofA\Acamp\code\mAP\acamp10kh_train_images.txt" class_names_path=data/predefined_classes_10k.txt

<a id="20k___visualiz_e_"></a>
## 20k       @ visualize

python3 visualize.py root_dir="N:\Datasets\Acamp\acamp20k" class_names_path=data/predefined_classes_10k.txt

python3 visualize.py seq_paths=../tf_api/acamp20k_test_180720.txt root_dir="N:\Datasets\Acamp\acamp20k" class_names_path=data/predefined_classes_10k.txt]

python3 visualize.py root_dir=N:\Datasets\Acamp\test_data_annotations class_names_path=data/predefined_classes_10k.txt

<a id="coyote___20k_visualiz_e_"></a>
### coyote       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k_repo/coyote class_names_path=data/predefined_classes_10k.txt

python3 visualize.py seq_paths=/data/acamp/acamp20k_repo/coyote/coyote_c class_names_path=data/predefined_classes_10k.txt

<a id="bear___20k_visualiz_e_"></a>
### bear       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k_repo/bear class_names_path=data/predefined_classes_10k.txt

<a id="deer___20k_visualiz_e_"></a>
### deer       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k_repo/deer class_names_path=data/predefined_classes_10k.txt

<a id="deer_jesse_9___deer_20k_visualize_"></a>
#### deer_jesse_9_       @ deer/20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k_repo/deer class_names_path=data/predefined_classes_10k.txt seq_prefix=deer_jesse_9_

<a id="moose___20k_visualiz_e_"></a>
### moose       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k_repo/moose class_names_path=data/predefined_classes_10k.txt

<a id="bison___20k_visualiz_e_"></a>
### bison       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k_repo/bison class_names_path=data/predefined_classes_bison.txt

<a id="horse___20k_visualiz_e_"></a>
### horse       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k_repo/horse class_names_path=data/predefined_classes_horse.txt

<a id="bear_1_1___20k_visualiz_e_"></a>
### bear_1_1       @ 20k/visualize

<a id="masks_0x1365_10___bear_1_1_20k_visualize_"></a>
#### masks_0x1365_10       @ bear_1_1/20k/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/bear\bear_1_1\masks_0x1365_10\images class_names_path=data/predefined_classes_10k.txt

<a id="masks_0x0_0p6___bear_1_1_20k_visualize_"></a>
#### masks_0x0_0p6       @ bear_1_1/20k/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/bear\bear_1_1_masks_0x0_0p6\images class_names_path=data/predefined_classes_10k.txt


<a id="bear_1_1_masks_448x448_ar_1p0___bear_1_1_20k_visualize_"></a>
#### bear_1_1_masks_448x448_ar_1p0       @ bear_1_1/20k/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/bear\bear_1_1_masks_448x448_ar_1p0\images class_names_path=data/predefined_classes_10k.txt


<a id="coyote_9_seq_54_frames___20k_visualiz_e_"></a>
### coyote_9_seq_54_frames       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k/masks/coyote_9_seq_54_frames class_names_path=data/predefined_classes_10k.txt enable_masks=1

<a id="deer_11_seq_56_frames___20k_visualiz_e_"></a>
### deer_11_seq_56_frames       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k/masks/deer_11_seq_56_frames class_names_path=data/predefined_classes_10k.txt enable_masks=1

<a id="moose_10_seq_50_frames___20k_visualiz_e_"></a>
### moose_10_seq_50_frames       @ 20k/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k/masks/moose_10_seq_50_frames class_names_path=data/predefined_classes_10k.txt enable_masks=1

<a id="bear_13_seq_103_frames_224x224_5___20k_visualiz_e_"></a>
### bear_13_seq_103_frames_224x224_5       @ 20k/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/masks\bear_13_seq_103_frames_224x224_5\images class_names_path=data/predefined_classes_10k.txt

<a id="bear_13_seq_103_frames_224x224_5_test___20k_visualiz_e_"></a>
### bear_13_seq_103_frames_224x224_5_test       @ 20k/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/masks/bear_13_seq_103_frames_224x224_5_test class_names_path=data/predefined_classes_10k.txt

<a id="bear_13_seq_103_frames___20k_visualiz_e_"></a>
### bear_13_seq_103_frames       @ 20k/visualize

python3 visualize.py root_dir=E:\Datasets\acamp\acamp20k\masks\bear_13_seq_103_frames class_names_path=data/predefined_classes_10k.txt enable_masks=1

<a id="test_0709___20k_visualiz_e_"></a>
### test_0709       @ 20k/visualize

python3 visualize.py seq_paths=acamp20k_test_0709.txt class_names_path=data/predefined_classes_10k.txt

<a id="results___visualiz_e_"></a>
## results       @ visualize

python3 visualize.py root_dir="N:\Datasets\Acamp\acamp5k\test\images" csv_root_dir=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_100_on_5k class_names_path=../labelling_tool/data/predefined_classes_orig.txt

python3 visualize.py root_dir="N:\Datasets\Acamp\acamp5k\test\images" csv_root_dir=H:\UofA\Acamp\code\results\faster_rcnn_resnet101\faster_rcnn_resnet101_coco_2018_01_28_100_on_5k class_names_path=../labelling_tool/data/predefined_classes_orig.txt

<a id="prototype_1___visualiz_e_"></a>
## prototype_1       @ visualize

python3 visualize.py root_dir=/data/acamp/p1_no_mask class_names_path=data/predefined_classes_10k.txt

<a id="source___prototype_1_visualiz_e_"></a>
### source       @ prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k/prototype_1_source class_names_path=data/predefined_classes_10k.txt save=1 save_fmt=jpg

<a id="coyote_9_1_moose_jesse_2_2_w___source_prototype_1_visualize_"></a>
#### coyote_9_1,moose_jesse_2_2_w       @ source/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k/prototype_1_source seq_paths=coyote_9_1,moose_jesse_2_2_w class_names_path=data/predefined_classes_10k.txt save=1 save_fmt=jpg n_vis=2 vis.write_frame_id=0

<a id="individual___source_prototype_1_visualize_"></a>
#### individual       @ source/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k/prototype_1_source seq_paths=moose_jesse_2_2_w class_names_path=data/predefined_classes_10k.txt save=1 save_fmt=jpg vis.write_frame_id=0

python3 visualize.py root_dir=/data/acamp/acamp20k/prototype_1_source seq_paths=coyote_9_1 class_names_path=data/predefined_classes_10k.txt save=1 save_fmt=jpg vis.write_frame_id=0

<a id="stack___source_prototype_1_visualize_"></a>
#### stack       @ source/prototype_1/visualize

python3 stackVideos.py root_dir=H:\UofA\Acamp\code\labelling_tool\log src_paths=vis_190822_0800,airport,vis_190822_0901,hw ext=jpg grid_size=1x4

<a id="cropped___source_prototype_1_visualize_"></a>
#### cropped       @ source/prototype_1/visualize

<a id="coyote___cropped_source_prototype_1_visualize_"></a>
##### coyote       @ cropped/source/prototype_1/visualize

python3 stackVideos.py root_dir=H:\UofA\Acamp\code\labelling_tool\log src_paths=vis_190822_0800,airport,syn_coyote_airport_3_crop_image000082 ext=jpg grid_size=1x3

<a id="moose___cropped_source_prototype_1_visualize_"></a>
##### moose       @ cropped/source/prototype_1/visualize

python3 stackVideos.py root_dir=H:\UofA\Acamp\code\labelling_tool\log src_paths=vis_190822_0901,hw_crop,syn_moose_hw_image000072 ext=jpg grid_size=1x3


<a id="bear_airport___prototype_1_visualiz_e_"></a>
### bear_airport       @ prototype_1/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/prototype_1/p1_bear_a_1 class_names_path=data/predefined_classes_10k.txt

<a id="no_mask_n_vis_2___bear_airport_prototype_1_visualize_"></a>
#### no_mask_n_vis_2       @ bear_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_no_mask/p1_bear_a_no_mask_1,prototype_1/p1_bear_a_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_n_vis_2___bear_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_n_vis_2       @ bear_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_mask_rcnn_resnet101/p1_bear_a_mask_rcnn_resnet101_1,prototype_1/p1_bear_a_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_no_mask_n_vis_3___bear_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_n_vis_3       @ bear_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_bear_a_1,p1_mask_rcnn_resnet101/p1_bear_a_mask_rcnn_resnet101_1,p1_no_mask/p1_bear_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=3 vis_size=1024x768 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_masknvis_3__bear_airport vis.text_fmt=green,0,5,1.0,1 labels=manual,mask_rcnn,no_mask

<a id="mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4___bear_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4       @ bear_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_bear_a_1,p1_mask_rcnn_resnet101/p1_bear_a_mask_rcnn_resnet101_1,p1_siam_mask/p1_bear_a_siam_mask_1,p1_no_mask/p1_bear_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 vis_size=1920x1080 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_masknvis_4__bear_airport  labels=manual,mask_rcnn,siam_mask,no_mask

<a id="mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___bear_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ bear_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_bear_a_1,p1_mask_rcnn_resnet101/p1_bear_a_mask_rcnn_resnet101_1,p1_siam_mask_davis/p1_bear_a_siam_mask_davis_1,p1_no_mask/p1_bear_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 vis_size=1920x1080 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_mask_davisnvis_4__bear_airport  labels=manual,mask_rcnn,siam_mask_davis,no_mask

<a id="bear_highway___prototype_1_visualiz_e_"></a>
### bear_highway       @ prototype_1/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/prototype_1/p1_bear_h_1 class_names_path=data/predefined_classes_10k.txt

<a id="no_mask_n_vis_2___bear_highway_prototype_1_visualize_"></a>
#### no_mask_n_vis_2       @ bear_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_no_mask/p1_bear_h_no_mask_1,prototype_1/p1_bear_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_n_vis_2___bear_highway_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_n_vis_2       @ bear_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_mask_rcnn_resnet101/p1_bear_h_mask_rcnn_resnet101_1,prototype_1/p1_bear_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___bear_highway_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ bear_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_bear_h_1,p1_mask_rcnn_resnet101/p1_bear_h_mask_rcnn_resnet101_1,p1_siam_mask_davis/p1_bear_h_siam_mask_davis_1,p1_no_mask/p1_bear_h_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 vis_size=1920x1080 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_mask_davisnvis_4__bear_highway  labels=manual,mask_rcnn,siam_mask_davis,no_mask

<a id="deer_airport___prototype_1_visualiz_e_"></a>
### deer_airport       @ prototype_1/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/prototype_1/p1_deer_a_1 class_names_path=data/predefined_classes_10k.txt

<a id="no_mask_n_vis_2___deer_airport_prototype_1_visualize_"></a>
#### no_mask_n_vis_2       @ deer_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_no_mask/p1_deer_a_no_mask_1,prototype_1/p1_deer_a_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_n_vis_2___deer_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_n_vis_2       @ deer_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_mask_rcnn_resnet101/p1_deer_h_mask_rcnn_resnet101_1,prototype_1/p1_deer_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_no_mask_n_vis_3___deer_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_n_vis_3       @ deer_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_deer_a_1,p1_mask_rcnn_resnet101/p1_deer_a_mask_rcnn_resnet101_1,p1_no_mask/p1_deer_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=3 vis_size=1024x768 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_masknvis_3__deer_airport  labels=manual,mask_rcnn,no_mask

<a id="mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4___deer_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4       @ deer_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_deer_a_1,p1_mask_rcnn_resnet101/p1_deer_a_mask_rcnn_resnet101_1,p1_siam_mask/p1_deer_a_siam_mask_1,p1_no_mask/p1_deer_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 vis_size=1920x1080 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_masknvis_4__deer_airport  labels=manual,mask_rcnn,siam_mask,no_mask

<a id="mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___deer_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ deer_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_deer_a_1,p1_mask_rcnn_resnet101/p1_deer_a_mask_rcnn_resnet101_1,p1_siam_mask_davis/p1_deer_a_siam_mask_davis_1,p1_no_mask/p1_deer_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 vis_size=1920x1080 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_mask_davisnvis_4__deer_airport  labels=manual,mask_rcnn,siam_mask_davis,no_mask

<a id="deer_highway___prototype_1_visualiz_e_"></a>
### deer_highway       @ prototype_1/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/prototype_1/p1_deer_h_1 class_names_path=data/predefined_classes_10k.txt

<a id="no_mask_n_vis_2___deer_highway_prototype_1_visualize_"></a>
#### no_mask_n_vis_2       @ deer_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_no_mask/p1_deer_h_no_mask_1,prototype_1/p1_deer_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_n_vis_2___deer_highway_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_n_vis_2       @ deer_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_mask_rcnn_resnet101/p1_deer_h_mask_rcnn_resnet101_1,prototype_1/p1_deer_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="coyote_airport___prototype_1_visualiz_e_"></a>
### coyote_airport       @ prototype_1/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/prototype_1/p1_coyote_a_1 class_names_path=data/predefined_classes_10k.txt

<a id="no_mask_n_vis_2___coyote_airport_prototype_1_visualize_"></a>
#### no_mask_n_vis_2       @ coyote_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_no_mask/p1_coyote_a_no_mask_1,prototype_1/p1_coyote_a_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4___coyote_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4       @ coyote_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_coyote_a_1,p1_mask_rcnn_resnet101/p1_coyote_a_mask_rcnn_resnet101_1,p1_siam_mask/p1_coyote_a_siam_mask_1,p1_no_mask/p1_coyote_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 vis_size=1920x1080 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_masknvis_4__coyote_airport  labels=manual,mask_rcnn,siam_mask,no_mask

<a id="mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___coyote_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ coyote_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_coyote_a_1,p1_mask_rcnn_resnet101/p1_coyote_a_mask_rcnn_resnet101_1,p1_siam_mask_davis/p1_coyote_a_siam_mask_davis_1,p1_no_mask/p1_coyote_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_mask_davisnvis_4__coyote_airport  labels=manual,mask_rcnn,siam_mask,no_mask crop_size=1080x360

<a id="mask_rcnn_resnet101_n_vis_2___coyote_airport_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_n_vis_2       @ coyote_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_mask_rcnn_resnet101/p1_coyote_a_mask_rcnn_resnet101_1,prototype_1/p1_coyote_a_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="coyote_highway___prototype_1_visualiz_e_"></a>
### coyote_highway       @ prototype_1/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/prototype_1/p1_coyote_h_1 class_names_path=data/predefined_classes_10k.txt

<a id="no_mask_n_vis_2___coyote_highway_prototype_1_visualize_"></a>
#### no_mask_n_vis_2       @ coyote_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_no_mask/p1_coyote_h_no_mask_1,prototype_1/p1_coyote_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_n_vis_2___coyote_highway_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_n_vis_2       @ coyote_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_mask_rcnn_resnet101/p1_coyote_h_mask_rcnn_resnet101_1,prototype_1/p1_coyote_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_no_mask_n_vis_3___coyote_highway_prototype_1_visualize_"></a>
#### mask_rcnn_resnet101_no_mask_n_vis_3       @ coyote_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_coyote_h_1,p1_mask_rcnn_resnet101/p1_coyote_h_mask_rcnn_resnet101_1,p1_no_mask/p1_coyote_h_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=3 vis_size=1024x768 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_masknvis_3__coyote_highway labels=manual,mask_rcnn,no_mask

<a id="moose_airport___prototype_1_visualiz_e_"></a>
### moose_airport       @ prototype_1/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/prototype_1/p1_moose_a_1 class_names_path=data/predefined_classes_10k.txt

<a id="no_mask_n_vis_2___moose_airport_prototype_1_visualiz_e_"></a>
#### no_mask_n_vis_2       @ moose_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_no_mask/p1_moose_a_no_mask_1,prototype_1/p1_moose_a_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080


<a id="mask_rcnn_resnet101_n_vis_2___moose_airport_prototype_1_visualiz_e_"></a>
#### mask_rcnn_resnet101_n_vis_2       @ moose_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_mask_rcnn_resnet101/p1_moose_a_mask_rcnn_resnet101_1,prototype_1/p1_moose_a_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4___moose_airport_prototype_1_visualiz_e_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_n_vis_4       @ moose_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_moose_a_1,p1_mask_rcnn_resnet101/p1_moose_a_mask_rcnn_resnet101_1,p1_siam_mask/p1_moose_a_siam_mask_1,p1_no_mask/p1_moose_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 vis_size=1920x1080 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_masknvis_4__moose_airport  labels=manual,mask_rcnn,siam_mask,no_mask

<a id="mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___moose_airport_prototype_1_visualiz_e_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ moose_airport/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_moose_a_1,p1_mask_rcnn_resnet101/p1_moose_a_mask_rcnn_resnet101_1,p1_siam_mask_davis/p1_moose_a_siam_mask_davis_1,p1_no_mask/p1_moose_a_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 vis_size=1920x1080 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_mask_davisnvis_4__moose_airport  labels=manual,mask_rcnn,siam_mask_davis,no_mask


<a id="moose_highway___prototype_1_visualiz_e_"></a>
### moose_highway       @ prototype_1/visualize

python3 visualize.py seq_paths=/data/acamp/acamp20k/prototype_1/p1_moose_h_1 class_names_path=data/predefined_classes_10k.txt

<a id="no_mask_n_vis_2___moose_highway_prototype_1_visualiz_e_"></a>
#### no_mask_n_vis_2       @ moose_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_no_mask/p1_moose_h_no_mask_1,prototype_1/p1_moose_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_n_vis_2___moose_highway_prototype_1_visualiz_e_"></a>
#### mask_rcnn_resnet101_n_vis_2       @ moose_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=p1_mask_rcnn_resnet101/p1_moose_h_mask_rcnn_resnet101_1,prototype_1/p1_moose_h_1 class_names_path=data/predefined_classes_10k.txt n_vis=2 vis_size=1920x1080

<a id="mask_rcnn_resnet101_no_mask_n_vis_3___moose_highway_prototype_1_visualiz_e_"></a>
#### mask_rcnn_resnet101_no_mask_n_vis_3       @ moose_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_moose_h_1,p1_mask_rcnn_resnet101/p1_moose_h_mask_rcnn_resnet101_1,p1_no_mask/p1_moose_h_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=3 vis_size=1024x768 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_masknvis_3__moose_highway labels=manual,mask_rcnn,no_mask

<a id="mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4___moose_highway_prototype_1_visualiz_e_"></a>
#### mask_rcnn_resnet101_no_mask_siam_mask_davis_n_vis_4       @ moose_highway/prototype_1/visualize

python3 visualize.py root_dir=/data/acamp/acamp20k seq_paths=prototype_1/p1_moose_h_1,p1_mask_rcnn_resnet101/p1_moose_h_mask_rcnn_resnet101_1,p1_siam_mask_davis/p1_moose_h_siam_mask_davis_1,p1_no_mask/p1_moose_h_no_mask_1 class_names_path=data/predefined_classes_10k.txt n_vis=4 save=1 save_fmt=jpg save_dir=mask_rcnn_resnet101_no_mask_siam_mask_davisnvis_4__moose_hw  labels=manual,mask_rcnn,siam_mask,no_mask crop_size=900x360



