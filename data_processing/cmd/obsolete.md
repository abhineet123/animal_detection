<!-- MarkdownTOC -->

- [AR](#a_r_)
    - [GRS       @ AR](#grs___ar_)
    - [VBU       @ AR](#vbu___ar_)
- [caltech](#caltech_)
    - [only_person       @ caltech](#only_person___caltec_h_)
    - [temporal_subsampling       @ caltech](#temporal_subsampling___caltec_h_)
        - [dummy       @ temporal_subsampling/caltech](#dummy___temporal_subsampling_caltech_)
- [kitti](#kitti_)
- [daimler](#daimler_)
- [tud_briussels](#tud_briussels_)
- [inria](#inria_)
    - [new_labels       @ inria](#new_labels___inri_a_)
    - [conversion       @ inria](#conversion___inri_a_)
- [biwi](#biw_i_)
    - [sunny_day-img-left       @ biwi](#sunny_day_img_left___biwi_)
    - [seq03-img-left       @ biwi](#seq03_img_left___biwi_)
    - [seq04-img-left       @ biwi](#seq04_img_left___biwi_)
    - [seq02-img-left       @ biwi](#seq02_img_left___biwi_)
    - [seq01-img-left       @ biwi](#seq01_img_left___biwi_)
    - [lp-left       @ biwi](#lp_left___biwi_)
- [voc_to_csv](#voc_to_cs_v_)
    - [grs       @ voc_to_csv](#grs___voc_to_csv_)
- [coco_to_csv](#coco_to_csv_)
    - [10k       @ coco_to_csv](#10k___coco_to_cs_v_)
        - [grs       @ 10k/coco_to_csv](#grs___10k_coco_to_cs_v_)
    - [384       @ coco_to_csv](#384___coco_to_cs_v_)
        - [10k_2       @ 384/coco_to_csv](#10k_2___384_coco_to_cs_v_)
        - [grs       @ 384/coco_to_csv](#grs___384_coco_to_cs_v_)
        - [10k_3       @ 384/coco_to_csv](#10k_3___384_coco_to_cs_v_)
        - [grs       @ 384/coco_to_csv](#grs___384_coco_to_cs_v__1)
        - [10k_4       @ 384/coco_to_csv](#10k_4___384_coco_to_cs_v_)
        - [grs       @ 384/coco_to_csv](#grs___384_coco_to_cs_v__2)
        - [5k       @ 384/coco_to_csv](#5k___384_coco_to_cs_v_)
        - [grs       @ 384/coco_to_csv](#grs___384_coco_to_cs_v__3)
    - [1k       @ coco_to_csv](#1k___coco_to_cs_v_)
        - [grs       @ 1k/coco_to_csv](#grs___1k_coco_to_csv_)
    - [1.6k       @ coco_to_csv](#1_6k___coco_to_cs_v_)
    - [val2017       @ coco_to_csv](#val2017___coco_to_cs_v_)

<!-- /MarkdownTOC -->

<a id="a_r_"></a>
# AR
<a id="grs___ar_"></a>
## GRS       @ AR-->obsolete
python temporal_subsampling.py list_file_name=acamp10k_train.txt root_dir=/data/acamp/acamp10k out_root_dir=/data/acamp/acamp10k/train_ar vis_width=1024 vis_height=768 fixed_ar=1 save_raw=1 show_img=0

<a id="vbu___ar_"></a>
## VBU       @ AR-->obsolete
python temporal_subsampling.py root_dir=G:\Datasets\Acamp\acamp10k\train out_root_dir=G:\Datasets\Acamp\acamp10k\train_ar vis_width=1024 vis_height=768 fixed_ar=1 save_raw=1 show_img=0

python temporal_subsampling.py list_file_name=acamp10kh_train.txt root_dir=G:\Datasets\Acamp\acamp10k\train out_root_dir=G:\Datasets\Acamp\acamp10k\train_ar vis_width=1024 vis_height=768 fixed_ar=1 save_raw=1 show_img=1

<a id="caltech_"></a>
# caltech

python caltech_to_csv.py root_dir=G:\Datasets\Caltech show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set00 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set01 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set02 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set03 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set04 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set05 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set06 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set07 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set08 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set09 show_img=1 ignore_occl=0

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set10 show_img=1 ignore_occl=0

<a id="only_person___caltec_h_"></a>
## only_person       @ caltech-->obsolete

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set00 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set01 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set02 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set03 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set04 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set05 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set06 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set07 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set08 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set09 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

python caltech_to_csv.py root_dir=G:\Datasets\Caltech\set10 show_img=1 ext=jpg vis_width=1280 vis_height=960 ignore_occl=1 only_person=2 vis_root=vis_only_person_2_no_occl save_raw=1

<a id="temporal_subsampling___caltec_h_"></a>
## temporal_subsampling       @ caltech-->obsolete

python temporal_subsampling.py root_dir=G:\Datasets\Caltech_old\vis_only_person_2_no_occl list_file_name="G:\Datasets\Caltech_old\vis_only_person_2_no_occl\list.txt" show_img=1 start_id=0 frame_gap=3 ext=jpg

python temporal_subsampling.py root_dir=G:\Datasets\Caltech_old\vis_only_person_2_no_occl list_file_name=caltech_ss_0_3.txt show_img=1 start_id=0 frame_gap=3 ext=jpg save_raw=1 out_root_dir=G:\Datasets\Acamp\acamp10k out_postfix=ss_0_3 show_img=0

python temporal_subsampling.py root_dir=G:\Datasets\Caltech_old\vis_only_person_2_no_occl list_file_name=caltech_ss_0_3.txt show_img=1 start_id=1 frame_gap=3 ext=jpg save_raw=1 out_root_dir=G:\Datasets\Acamp\acamp10k out_postfix=ss_1_3 show_img=0

<a id="dummy___temporal_subsampling_caltech_"></a>
### dummy       @ temporal_subsampling/caltech-->obsolete

python temporal_subsampling.py root_dir=G:\Datasets\Caltech_old\vis_only_person_2_no_occl list_file_name=caltech_ss_dummy.txt show_img=1 start_id=0 frame_gap=1 ext=jpg save_raw=1 out_root_dir=G:\Datasets\Acamp\acamp10k show_img=0


<a id="kitti_"></a>
# kitti

python kitti_to_csv.py root_dir=G:\Datasets\KITTI\training show_img=1 ignore_dc=0

python kitti_to_csv.py root_dir=G:\Datasets\KITTI\training show_img=1 ignore_dc=1 save_raw=0 ext=jpg only_person=1 vis_root=vis_only_person_no_dc

python kitti_to_csv.py root_dir=G:\Datasets\KITTI\training show_img=1 ignore_dc=1 save_raw=1 ext=jpg only_person=1 vis_root=raw_only_person_no_dc


<a id="daimler_"></a>
# daimler

python daimler_to_csv.py root_dir=G:\Datasets\daimler\DaimlerBenchmark show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg

python daimler_to_csv.py root_dir=G:\Datasets\daimler\DaimlerBenchmark show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 only_person=1 save_raw=0 ext=mkv

<a id="tud_briussels_"></a>
# tud_briussels

python tud_brussels_to_csv.py img_path=G:\Datasets\tud_brussels\TUD-Brussels  ann_path="G:\Datasets\tud_brussels\TUD-Brussels\annotation.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=1 ext=jpg mode=1

python tud_brussels_to_csv.py img_path=G:\Datasets\tud_brussels\TUD-Brussels  ann_path="G:\Datasets\tud_brussels\TUD-Brussels\annotation.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=1

python tud_brussels_to_csv.py img_path=G:\Datasets\tud_brussels\TUD-MotionPairs\positive  ann_path="G:\Datasets\tud_brussels\TUD-MotionPairs\positive\train-pos.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=1 ext=jpg mode=0

<a id="inria_"></a>
# inria

<a id="new_labels___inri_a_"></a>
## new_labels       @ inria-->obsolete

vbb('vbbSaveTxt', vbb('vbbLoad', 'G:\Datasets\INRIA\INRIAPerson\InriaNewTestLabels.vbb' ), 'N:\\Datasets\\INRIA\\test.txt', 0 )

vbb('vbbSaveTxt', vbb('vbbLoad', 'G:\Datasets\INRIA\INRIAPerson\InriaNewTrainingLabels.vbb' ), 'N:\\Datasets\\INRIA\\train.txt', 0 )

<a id="conversion___inri_a_"></a>
## conversion       @ inria-->obsolete

python caltech_to_csv.py root_dir=G:\Datasets\INRIA show_img=1 ignore_occl=0 img_ext=png ext=jpg ignore_occl=1 only_person=1 vis_root=vis_only_person_no_occl save_raw=0

python caltech_to_csv.py root_dir=G:\Datasets\INRIA show_img=0 ignore_occl=0 img_ext=png ext=jpg ignore_occl=0 only_person=1 save_raw=1


<a id="biw_i_"></a>
# biwi

<a id="sunny_day_img_left___biwi_"></a>
## sunny_day-img-left       @ biwi-->obsolete

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\sunny_day-img-left  ann_path="G:\Datasets\BIWI\sunny_day-img-left\annotations.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\sunny_day-img-left  ann_path="G:\Datasets\BIWI\sunny_day-img-left\refined.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\sunny_day-img-left  ann_path="G:\Datasets\BIWI\sunny_day-img-left\refined.idl" show_img=0 ignore_occl=0 save_raw=1 ext=jpg mode=0 out_root_dir=G:\Datasets\Acamp\acamp10k\biwi

<a id="seq03_img_left___biwi_"></a>
## seq03-img-left       @ biwi-->obsolete

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq03-img-left  ann_path="G:\Datasets\BIWI\seq03-img-left\bahnhof-annot.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq03-img-left  ann_path="G:\Datasets\BIWI\seq03-img-left\refined.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq03-img-left  ann_path="G:\Datasets\BIWI\seq03-img-left\refined.idl" show_img=0 ignore_occl=0 save_raw=1 ext=jpg mode=0 out_root_dir=G:\Datasets\Acamp\acamp10k\biwi


<a id="seq04_img_left___biwi_"></a>
## seq04-img-left       @ biwi-->obsolete

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq04-img-left  ann_path="G:\Datasets\BIWI\seq04-img-left\jelmoli-annot.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq04-img-left  ann_path="G:\Datasets\BIWI\seq04-img-left\eth02.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq04-img-left  ann_path="G:\Datasets\BIWI\seq04-img-left\eth02.idl" show_img=0 ignore_occl=0  save_raw=1 ext=jpg mode=0 out_root_dir=G:\Datasets\Acamp\acamp10k\biwi

<a id="seq02_img_left___biwi_"></a>
## seq02-img-left       @ biwi-->obsolete

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq02-img-left  ann_path="G:\Datasets\BIWI\seq02-img-left\linthescher-annot.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq02-img-left  ann_path="G:\Datasets\BIWI\seq02-img-left\linthescher-annot.idl" show_img=0 ignore_occl=0 save_raw=1 ext=jpg mode=0 out_root_dir=G:\Datasets\Acamp\acamp10k\biwi

<a id="seq01_img_left___biwi_"></a>
## seq01-img-left       @ biwi-->obsolete

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq01-img-left  ann_path="G:\Datasets\BIWI\seq01-img-left\crossing-annot.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\seq01-img-left  ann_path="G:\Datasets\BIWI\seq01-img-left\crossing-annot.idl" show_img=0 ignore_occl=0 save_raw=1 ext=jpg mode=0 out_root_dir=G:\Datasets\Acamp\acamp10k\biwi

<a id="lp_left___biwi_"></a>
## lp-left       @ biwi-->obsolete

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\lp-left  ann_path="G:\Datasets\BIWI\lp-left\lp-annot.idl" show_img=1 ignore_occl=0 vis_width=1280 vis_height=960 save_raw=0 ext=jpg mode=0

python tud_brussels_to_csv.py db_type=1 img_path=G:\Datasets\BIWI\lp-left  ann_path="G:\Datasets\BIWI\lp-left\lp-annot.idl" show_img=0 ignore_occl=0  save_raw=1 ext=jpg mode=0 out_root_dir=G:\Datasets\Acamp\acamp10k\biwi

<a id="voc_to_cs_v_"></a>
# voc_to_csv

python voc_to_csv.py root_dir=G:\Datasets\VOC2012\JPEGImages show_img=1 vis_width=1280 vis_height=960 only_person=0 save_raw=0 ext=jpg

python voc_to_csv.py root_dir=G:\Datasets\VOC2012\JPEGImages show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg show_img=0

<a id="grs___voc_to_csv_"></a>
## grs       @ voc_to_csv-->obsolete

python voc_to_csv.py root_dir=/data/voc2012/JPEGImages show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg show_img=0


<a id="coco_to_csv_"></a>
# coco_to_csv

python coco_test.py

python coco_to_csv.py root_dir=G:\Datasets\COCO17 data_type=train2017 show_img=1 vis_width=1280 vis_height=960 only_person=0 save_raw=0 ext=jpg

python coco_to_csv.py root_dir=G:\Datasets\COCO17 data_type=train2017 show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=0 n_frames=10000 frame_gap=6 show_img=0

<a id="10k___coco_to_cs_v_"></a>
## 10k       @ coco_to_csv-->obsolete

python coco_to_csv.py root_dir=G:\Datasets\COCO17 data_type=train2017 save_path=G:\Datasets\Acamp\acamp20k\human_coco1710k show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=0 n_frames=10000 frame_gap=6 show_img=0

<a id="grs___10k_coco_to_cs_v_"></a>
### grs       @ 10k/coco_to_csv-->obsolete

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=0 n_frames=10000 frame_gap=6 show_img=0

<a id="384___coco_to_cs_v_"></a>
## 384       @ coco_to_csv-->obsolete

python coco_to_csv.py root_dir=G:\Datasets\COCO17 data_type=train2017 show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=0 n_frames=384 frame_gap=6 show_img=0


<a id="10k_2___384_coco_to_cs_v_"></a>
### 10k_2       @ 384/coco_to_csv-->obsolete

<a id="grs___384_coco_to_cs_v_"></a>
### grs       @ 384/coco_to_csv-->obsolete

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 save_path=/data/acamp/acamp20k/human_coco1710k_2 show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=1 n_frames=10000 frame_gap=6 show_img=0

<a id="10k_3___384_coco_to_cs_v_"></a>
### 10k_3       @ 384/coco_to_csv-->obsolete

<a id="grs___384_coco_to_cs_v__1"></a>
### grs       @ 384/coco_to_csv-->obsolete

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 save_path=/data/acamp/acamp20k/human_coco1710k_3 show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=2 n_frames=10000 frame_gap=6 show_img=0

<a id="10k_4___384_coco_to_cs_v_"></a>
### 10k_4       @ 384/coco_to_csv-->obsolete

<a id="grs___384_coco_to_cs_v__2"></a>
### grs       @ 384/coco_to_csv-->obsolete

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 save_path=/data/acamp/acamp20k/human_coco1710k_4 show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=3 n_frames=10000 frame_gap=6 show_img=0

<a id="5k___384_coco_to_cs_v_"></a>
### 5k       @ 384/coco_to_csv-->obsolete

<a id="grs___384_coco_to_cs_v__3"></a>
### grs       @ 384/coco_to_csv-->obsolete

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 save_path=/data/acamp/acamp20k/human_coco175k vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=2 n_frames=5000 frame_gap=6 show_img=0

<a id="1k___coco_to_cs_v_"></a>
## 1k       @ coco_to_csv-->obsolete

python coco_to_csv.py root_dir=G:\Datasets\COCO17 data_type=train2017 save_path=G:\Datasets\Acamp\acamp20k\human_coco17_2_1k show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=2 n_frames=1000 frame_gap=6 show_img=0

python coco_to_csv.py root_dir=G:\Datasets\COCO17 data_type=train2017 save_path=G:\Datasets\Acamp\acamp20k\human_coco17_3_1k show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=3 n_frames=1000 frame_gap=6 show_img=0

<a id="grs___1k_coco_to_csv_"></a>
### grs       @ 1k/coco_to_csv-->obsolete

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 save_path=/data/acamp/acamp20k/human_coco17_2_1k show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=2 n_frames=1000 frame_gap=6 show_img=0

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 save_path=/data/acamp/acamp20k/human_coco17_3_1k show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=3 n_frames=1000 frame_gap=6 show_img=0

<a id="1_6k___coco_to_cs_v_"></a>
## 1.6k       @ coco_to_csv-->obsolete

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 save_path=/data/acamp/acamp20k/human_coco17_1600 show_img=0 only_person=1 save_raw=1 ext=jpg start_id=0 n_frames=1600 frame_gap=6 show_img=0

python coco_to_csv.py root_dir=/data/coco2017 data_type=train2017 save_path=/data/acamp/acamp20k/human_coco17_1600_2 show_img=0 only_person=1 save_raw=1 ext=jpg start_id=1 n_frames=1600 frame_gap=6 show_img=0

<a id="val2017___coco_to_cs_v_"></a>
## val2017       @ coco_to_csv-->obsolete

python coco_to_csv.py root_dir=G:\Datasets\COCO17 data_type=val2017 show_img=1 vis_width=1280 vis_height=960 only_person=1 save_raw=1 ext=jpg start_id=0 n_frames=0 frame_gap=1 show_img=0
