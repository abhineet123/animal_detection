
<a id="yolo"></a>
# yolo

<a id="train__yolo"></a>
## train       @ yolo

python to_txt.py type=1 list_file="N:\Datasets\Acamp\acamp5k\train\images\acamp5k_train_seq_list.txt" class_names_path=../labelling_tool/data/predefined_classes_orig.txt out_dir="N:\Datasets\Acamp\acamp5k\train\labels"

<a id="test__yolo"></a>
## test       @ yolo

python to_txt.py type=1 list_file="N:\Datasets\Acamp\acamp5k\test\images\acamp5k_test_seq_list.txt" class_names_path=../labelling_tool/data/predefined_classes_orig.txt out_dir="N:\Datasets\Acamp\acamp5k\test\labels"


<a id="map"></a>
# map

python to_txt.py type=0 list_file=N:\Datasets\Acamp\acamp10k\test\images class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp10k\test\labels"

python to_txt.py type=0 file_name="N:\Datasets\Acamp\acamp10k\test\images\deer_1_3" class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp10k\test\labels"

python to_txt.py type=0 file_name="N:\Datasets\Acamp\acamp10k\test\images\human_mot17i_MOT17-04" class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp10k\test\labels"

python to_txt.py type=0 file_name="N:\Datasets\Acamp\bear_1_1\test\images\human_mot17i_MOT17-11" class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp10k\test\labels"

python to_txt.py type=0 file_name="N:\Datasets\Acamp\acamp10k\test\images\moose_21_1" class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp10k\test\labels"

python to_txt.py type=0 list_file=acamp10k_test.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp10k\test\labels"

python to_txt.py type=0 list_file=../tf_api/acamp20k_test_180727.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp20k\test\labels"

python to_txt.py type=0 list_file=../tf_api/acamp20k3_coco_test.txt root_dir=N:\Datasets\Acamp\acamp20k class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp20k\test\labels"

python to_txt.py type=0 list_file=../tf_api/acamp5k_test.txt root_dir=N:\Datasets\Acamp\acamp5k\test\images class_names_path=data/predefined_classes_10k.txt out_dir="N:\Datasets\Acamp\acamp5k\test\labels"
''