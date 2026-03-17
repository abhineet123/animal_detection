<!-- MarkdownTOC -->

- [xml_to_csv](#xml_to_cs_v_)
    - [10k       @ xml_to_csv](#10k___xml_to_csv_)
    - [20k       @ xml_to_csv](#20k___xml_to_csv_)
    - [20k7_new       @ xml_to_csv](#20k7_new___xml_to_csv_)
    - [25k7_new       @ xml_to_csv](#25k7_new___xml_to_csv_)
        - [test_0709       @ 25k7_new/xml_to_csv](#test_0709___25k7_new_xml_to_cs_v_)
    - [cows       @ xml_to_csv](#cows___xml_to_csv_)
    - [horses       @ xml_to_csv](#horses___xml_to_csv_)
        - [horse_16       @ horses/xml_to_csv](#horse_16___horses_xml_to_cs_v_)
        - [horse_24       @ horses/xml_to_csv](#horse_24___horses_xml_to_cs_v_)
        - [horse_25       @ horses/xml_to_csv](#horse_25___horses_xml_to_cs_v_)
    - [bear       @ xml_to_csv](#bear___xml_to_csv_)
        - [bear_1_1       @ bear/xml_to_csv](#bear_1_1___bear_xml_to_cs_v_)
        - [bear_1_1_even_10       @ bear/xml_to_csv](#bear_1_1_even_10___bear_xml_to_cs_v_)
    - [bison       @ xml_to_csv](#bison___xml_to_csv_)
        - [bison_60_w       @ bison/xml_to_csv](#bison_60_w___bison_xml_to_csv_)
        - [bison_42_w       @ bison/xml_to_csv](#bison_42_w___bison_xml_to_csv_)
    - [elk       @ xml_to_csv](#elk___xml_to_csv_)
    - [airport       @ xml_to_csv](#airport___xml_to_csv_)
    - [highway       @ xml_to_csv](#highway___xml_to_csv_)
    - [coyote_10_5       @ xml_to_csv](#coyote_10_5___xml_to_csv_)
    - [coyote_b       @ xml_to_csv](#coyote_b___xml_to_csv_)
    - [p1_highway       @ xml_to_csv](#p1_highway___xml_to_csv_)
    - [prototype_1_vid       @ xml_to_csv](#prototype_1_vid___xml_to_csv_)
    - [1_in_10_8_class       @ xml_to_csv](#1_in_10_8_class___xml_to_csv_)
    - [1_in_2_all_static       @ xml_to_csv](#1_in_2_all_static___xml_to_csv_)

<!-- /MarkdownTOC -->

<a id="xml_to_cs_v_"></a>
# xml_to_csv
python xml_to_csv.py base_path=N:\Datasets\Acamp\marcin_180613\images seq_name=static_1

python3 xml_to_csv.py seq_paths="N:\Datasets\Acamp\marcin_180615\list.txt" class_names_path=lists/classes/predefined_classes_orig.txt

python3 xml_to_csv.py root_dir="N:\Datasets\Acamp\test_data_annotations" class_names_path=lists/classes/predefined_classes_orig.txt

python3 xml_to_csv.py seq_paths="N:\Datasets\Acamp\marcin_180613\images\deer\deer_2_5m" class_names_path=lists/classes/predefined_classes_orig.txt

python3 xml_to_csv.py seq_paths="N:\Datasets\Acamp\acamp20k/deer_jesse_7_4" class_names_path=lists/classes/predefined_classes_orig.txt

python3 xml_to_csv.py seq_paths="N:\Datasets\Acamp\backgrounds" class_names_path=lists/classes/predefined_classes_bkg.txt

<a id="10k___xml_to_csv_"></a>
## 10k       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=N:\Datasets\Acamp\acamp10k\animals class_names_path=lists/classes/predefined_classes_10k.txt

python3 xml_to_csv.py seq_paths="N:\Datasets\Acamp\acamp10k\test\images\moose_21_1" class_names_path=lists/classes/predefined_classes_10k.txt

python3 xml_to_csv.py seq_paths="N:\Datasets\Acamp\acamp10k\test\images\moose_21_1" class_names_path=lists/classes/predefined_classes_10k.txt

python3 xml_to_csv.py seq_paths=../tf_api/acamp_all_bear.txt root_dir=/data/acamp/acamp20k/bear class_names_path=lists/classes/predefined_classes_10k.txt

<a id="20k___xml_to_csv_"></a>
## 20k       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=N:\Datasets\Acamp\acamp20k class_names_path=lists/classes/predefined_classes_10k.txt

python3 xml_to_csv.py seq_paths=../tf_api/acamp20k_test_180719.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=lists/classes/predefined_classes_10k.txt

python3 xml_to_csv.py seq_paths=../tf_api/acamp20k_test_180720.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=lists/classes/predefined_classes_10k.txt

python3 xml_to_csv.py seq_paths=../tf_api/acamp20k_test_180720.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=lists/classes/predefined_classes_10k.txt

python3 xml_to_csv.py seq_paths=../tf_api/acamp20k_test_180727.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=lists/classes/predefined_classes_10k.txt

<a id="20k7_new___xml_to_csv_"></a>
## 20k7_new       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=../tf_api/acamp20k7_new_train.txt root_dir=/data/acamp/acamp20k class_names_path=lists/classes/predefined_classes_20k7.txt

<a id="25k7_new___xml_to_csv_"></a>
## 25k7_new       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=../tf_api/acamp25k7_train_new.txt root_dir=/data/acamp/acamp20k class_names_path=lists/classes/predefined_classes_20k7.txt

<a id="test_0709___25k7_new_xml_to_cs_v_"></a>
### test_0709       @ 25k7_new/xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=acamp20k_test_0709.txt class_names_path=lists/classes/predefined_classes_10k.txt root_dir=N:\Datasets\Acamp\acamp20k

python3 xml_to_csv.py seq_paths=acamp20k_test_0709.txt class_names_path=lists/classes/predefined_classes_10k.txt  root_dir=/data/acamp/acamp20k

<a id="cows___xml_to_csv_"></a>
## cows       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=acamp_cows.txt class_names_path=lists/classes/predefined_classes_cow.txt root_dir=/data/acamp/acamp20k

<a id="horses___xml_to_csv_"></a>
## horses       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=acamp_horses.txt class_names_path=lists/classes/predefined_classes_horse.txt root_dir=/data/acamp/acamp20k

<a id="horse_16___horses_xml_to_cs_v_"></a>
### horse_16       @ horses/xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=horse_16 class_names_path=lists/classes/predefined_classes_horse.txt root_dir=/data/acamp/acamp20k

<a id="horse_24___horses_xml_to_cs_v_"></a>
### horse_24       @ horses/xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=horse_24 class_names_path=lists/classes/predefined_classes_horse.txt root_dir=/data/acamp/acamp20k

<a id="horse_25___horses_xml_to_cs_v_"></a>
### horse_25       @ horses/xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=horse_25 class_names_path=lists/classes/predefined_classes_horse.txt root_dir=/data/acamp/acamp20k

<a id="bear___xml_to_csv_"></a>
## bear       @ xml_to_csv-->xml_acamp
<a id="bear_1_1___bear_xml_to_cs_v_"></a>
### bear_1_1       @ bear/xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=bear_1_1 class_names_path=lists/classes/predefined_classes.txt root_dir=/data/acamp/acamp20k enable_mask=1

<a id="bear_1_1_even_10___bear_xml_to_cs_v_"></a>
### bear_1_1_even_10       @ bear/xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=bear_1_1_even_10 class_names_path=lists/classes/predefined_classes.txt root_dir=/data/acamp/acamp20k/bear

<a id="bison___xml_to_csv_"></a>
## bison       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes.txt root_dir=/data/acamp/acamp20k/bison

<a id="bison_60_w___bison_xml_to_csv_"></a>
### bison_60_w       @ bison/xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=bison_60_w class_names_path=lists/classes/predefined_classes.txt root_dir=/data/acamp/acamp20k/bison

<a id="bison_42_w___bison_xml_to_csv_"></a>
### bison_42_w       @ bison/xml_to_csv-->xml_acamp
python3 xml_to_csv.py seq_paths=bison_42_w class_names_path=lists/classes/predefined_classes.txt root_dir=/data/acamp/acamp20k/bison

<a id="elk___xml_to_csv_"></a>
## elk       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes.txt root_dir=/data/acamp/acamp20k/elk

<a id="airport___xml_to_csv_"></a>
## airport       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes_bear.txt seq_paths=/data/acamp/acamp20k/backgrounds\airport

<a id="highway___xml_to_csv_"></a>
## highway       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes_bear.txt seq_paths=/data/acamp/acamp20k/backgrounds\highway

<a id="coyote_10_5___xml_to_csv_"></a>
## coyote_10_5       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes.txt seq_paths=/data/acamp/acamp20k/coyote\coyote_10_5

<a id="coyote_b___xml_to_csv_"></a>
## coyote_b       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes.txt seq_paths=/data/acamp/acamp20k/coyote\coyote_b

<a id="p1_highway___xml_to_csv_"></a>
## p1_highway       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes.txt root_dir=/data/acamp/acamp20k/prototype_1

<a id="prototype_1_vid___xml_to_csv_"></a>
## prototype_1_vid       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes.txt root_dir=/data/acamp/acamp20k/prototype_1_vid

<a id="1_in_10_8_class___xml_to_csv_"></a>
## 1_in_10_8_class       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py class_names_path=lists/classes/predefined_classes_4k8.txt root_dir=/data/acamp/acamp20k load_samples=1 load_samples_root=/data/acamp/acamp20k/1_in_10_8_class seq_paths=../tf_api/acamp_all_8_class.txt

<a id="1_in_2_all_static___xml_to_csv_"></a>
## 1_in_2_all_static       @ xml_to_csv-->xml_acamp
python3 xml_to_csv.py root_dir=/data/acamp/acamp20k load_samples=1 load_samples_root=/data/acamp/acamp20k/1_in_2_all_static seq_paths=../tf_api/acamp_static_3_class.txt
