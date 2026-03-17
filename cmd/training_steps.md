Following steps can be followed to train and test a Mask RCNN model implemented
in the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) or a [Tensorflow implementation of Sharpmask](https://github.com/aby2s/sharpmask) using a new mask dataset produced by the labeling tool.    
Throughout, we will be following the example of training and testing on a datasset with two sequences - *bear_jesse_32_1_w* and *bear_jesse_38_1_w* - that are assumed to be present under */data/acamp/acamp20k*

# Mask RCNN

1. RetinaNet needs an updated version of the API so the training commands here are for that version too.
Please clone [this repo](https://github.com/abhineet123/models) to the home directory and set it up following the instructions in the `Protobuf Compilation`, `Manual protobuf-compiler installation and usage` and `Add Libraries to PYTHONPATH`
 sections of [this](https://github.com/abhineet123/models/blob/master/research/object_detection/g3doc/installation.md) file.

2. Convert the xml annotations produced by the labeling tool into mask
sequences. This can be done individually for each sequence using the `Save Masks` button on the tool or as a batch operation for multiple sequences using [labelling_tool/to_mask_seq.py](labelling_tool/to_mask_seq.py).
Sample commands for using  it are in the `mask_seq` section of [labelling_tool/batch.md](labelling_tool/batch.md). This step might also require the creation of a new class names file similar to those in [labelling_tool/data](labelling_tool/data) if the dataset contains new classes that have not already been trained on.  

    __Example:__  
`
python3 to_mask_seq.py seq_paths=bear_jesse_32_1_w,bear_jesse_38_1_w class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k out_mask_size=0x682 out_border=10 show_img=0 save_test=1 save_raw_mask=0
`    

    This will create four mask sequences in */data/acamp/acamp20k/masks* - two each for labeled (*bear_jesse_32_1_w_0x682_10*, *bear_jesse_38_1_w_0x682_10*) and unlabeled (*bear_jesse_32_1_w_0x682_10_test*, *bear_jesse_38_1_w_0x682_10_test*) frames that will respectively be used for training and testing.

3. Convert the mask sequences to a tfrecord file using [tf_api/mask_to_record.py](tf_api/mask_to_record.py).
This also does batch processing and its sample commands can be found in the *mask_to_record* section of [labeling_tool/batch.md](labeling_tool/batch.md).  

    __Example:__  
`
python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/masks --seq_paths=bear_jesse_32_1_w,bear_jesse_38_1_w --seq_postfix=0x682_10 --output_path=/data/acamp/acamp20k/bear_jesse_32_1_w_38_1_w_0x682_10.record --label_map_path=data/wildlife_label_map_mask.pbtxt
`  

    This will create a tfrecord file called */data/acamp/acamp20k/bear_jesse_32_1_w_38_1_w_0x682_10.record*.

4. There should be no need to create a new protobuf label map file since we only need single class models whose label is in [data/wildlife_label_map_mask.pbtxt](data/wildlife_label_map_mask.pbtxt)

5. Download the appropriate pretrained model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) under the `COCO-trained models` section.
Extract the archive to *tf_api/pre_trained_models/*.

6. Create a new protobuf config file by copying and editing an existing *.config* file under [tf_api/configs/mask_rcnn_resnet101_atrous_coco_2018_01_28](tf_api/configs/mask_rcnn_resnet101_atrous_coco_2018_01_28) (for the *resnet101* variant of the model; config files for *inception_v2*, *resnet50* and *inception_resnet_v2* are also available under the respective folders).
Let the new config file thus created be named  *mask_rcnn_resnet101_bear_jesse_32_1_w_38_1_w_0x682_10_max_1_size_400_res_60.config*.    
Taking [mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60.config](tf_api/configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_1_1_100_max_1_size_400_res_60.config) file as an example, following changes need to be made for the example training scenario:
    - `input_path` under `train_input_reader/tf_record_input_reader` in line 136 where the path of the tfecord file created in step 3 would go.
    - [optional] `mask_height` and `mask_width` under `second_stage_box_predictor/mask_rcnn_box_predictor` in lines 84 and 85 might be increased if more precise masks are needed; the maximum values for these fields depend on the amount of available GPU memory and can only be determind experimentally; set them to some higher value, e.g. 90 and start the training - it will crash if it is too high, decrease it slightly and try again
    - [optional] `min_dimension` and `max_dimension` under `image_resizer/keep_aspect_ratio_resizer` in lines 7 and 8 might be changed along with the mask dimensions to help make higher mask dimensions possible
    - [optional] `input_path` and `label_map_path` under `eval_input_reader` in lines 147 and 151 might be changed too if evaluation is to be performed while training.
    - [optional] set `fine_tune_checkpoint` under `train_config` in line 129 to the location of the pretrained model obtained in step 5 if it is different from the default
    - [optional] `batch_size` under `train_config` in line 104 might be increased if GPU memory permits
    - [optional] more advanced changes might be made if experimentation with the model is to be done. Details about the various fields in the config files and their possible values can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).

7. Run [~/models/research/object_detection/legacy/train.py](https://github.com/abhineet123/models/blob/master/research/object_detection/legacy/train.py) __from *tf_api/*__ to start training.
Sample commands for running it are in the `train` section of [tf_api/mask_rcnn.md](tf_api/mask_rcnn.md) under individual subsections for each of the four variants.    

    __Example:__  
`
CUDA_VISIBLE_DEVICES=0 python3 ~/models/research/object_detection/legacy/train.py --logtostderr --pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_jesse_32_1_w_38_1_w_0x682_10_max_1_size_400_res_60.config --train_dir=trained/mask_rcnn_resnet101_bear_jesse_32_1_w_38_1_w_0x682_10_max_1_size_400_res_60 --n_steps=1000000 --save_interval_secs=600
`

    Optionally, evaluation can be performed too by running [~/models/research/object_detection/legacy/eval.py](https://github.com/abhineet123/models/blob/master/research/object_detection/legacy/eval.py). This needs a second GPU.

8. Training will be done when the loss stops decreasing. Number of steps needed will depend on the size of the training set and the model variant but should be around 200K-500K.
Loss behavior can be monitored by running tensorboard with the training directory as the log directory and opening the URL provided by the command.  

    __Example:__  
`
tensorboard --logdir=trained/mask_rcnn_resnet101_bear_jesse_32_1_w_38_1_w_0x682_10_max_1_size_400_res_60
`

9. Once training is done, the model can be tested on unlabeled images by running [tf_api/tf_api_eval.py](tf_api/tf_api_eval.py).
Sample commands are in the `tf_api_eval` section of the [tf_api/mask_rcnn.md](tf_api/mask_rcnn.md).
Note that, unlike the training steps, the protobuf label file used here should be the one specific to the animal whose model is being tested, e.g. [wildlife_label_map_bear.pbtxt](tf_api/data/wildlife_label_map_bear.pbtxt) for bear instead of the generic [wildlife_label_map_mask.pbtxt](tf_api/data/wildlife_label_map_mask.pbtxt). 
This is needed to allow the generated masks to be correctly imported in the labeling tool.    
    
    __Example:__  
`
CUDA_VISIBLE_DEVICES=0 python3 tf_api_eval.py ckpt_path=trained/mask_rcnn_resnet101_bear_jesse_32_1_w_38_1_w_0x682_10_max_1_size_400_res_60 labels_path=data/wildlife_label_map_bear.pbtxt n_frames=0 batch_size=1 show_img=0 n_classes=1  eval_every=0 root_dir=/data/acamp/acamp20k/bear/masks seq_paths=bear_jesse_32_1_w_0x682_10_test,bear_jesse_38_1_w_0x682_10_test pipeline_config_path=configs/mask_rcnn_resnet101_atrous_coco_2018_01_28/mask_rcnn_resnet101_bear_jesse_32_1_w_38_1_w_0x682_10_max_1_size_400_res_60.config sampling_ratio=1 random_sampling=0 sleep_time=10 save_video=0 save_det=0 enable_masks=1 out_postfix=test
`

10. The generated masks can be visualized by running [labelling_tool/visualize_masks.py](labelling_tool/visualize_masks.py).
Sample commands are in the `visualize` section of [tf_api/mask_rcnn.md](tf_api/mask_rcnn.md).  
    
    __Example:__  
`
python3 visualize_masks.py img_paths=bear_jesse_32_1_w,bear_jesse_38_1_w img_root_dir=/data/acamp/acamp20k/bear mask_root_dir=../tf_api/trained/mask_rcnn_resnet101_bear_jesse_32_1_w_38_1_w_0x682_10_max_1_size_400_res_60/200000_on_test map_to_bbox=1 combine=1 border=10 include_orig=1 include_binary=1 out_size=0x0 write_text=0
`

    This example assumes that the training was stopped after 200000 steps, hence the generated masks being present under the subfolder *200000_on_test* in the training folder.


# Sharpmask

1. Same as step 2 of Mask RCNN except that the mask size should be smaller since the output size of Sharpmask is same as its input size. 

    __Example:__  
`
python3 to_mask_seq.py seq_paths=bear_jesse_32_1_w,bear_jesse_38_1_w class_names_path=data/predefined_classes_mask.txt root_dir=/data/acamp/acamp20k/bear out_mask_size=224x224 out_border=5 show_img=0 save_test=1 save_raw_mask=0
`    

    This will create four mask sequences in */data/acamp/acamp20k/bear/masks* - two each for labeled (*bear_jesse_32_1_w_224x224_5*, *bear_jesse_38_1_w_224x224_5*) and unlabeled (*bear_jesse_32_1_w_224x224_5_test*, *bear_jesse_38_1_w_224x224_5_test*) frames that will respectively be used for training and testing.

2. Same as step 3 of Mask RCNN except that `db_type=1` must be specified.
Sample commands are in `mask_to_record` section of [sharpmask_tf\sharpmask_tf.md](sharpmask_tf\sharpmask_tf.md).

    __Example:__  
`
python3 mask_to_record.py --root_dir=/data/acamp/acamp20k/masks --seq_paths=bear_jesse_32_1_w,bear_jesse_38_1_w --seq_postfix=224x224_5 --output_path=/data/acamp/acamp20k/bear_jesse_32_1_w_38_1_w_224x224_5.record --label_map_path=data/wildlife_label_map_mask.pbtxt db_type=1
`

3. Download the pretrained model from [here](http://download.tensorflow.org/models/official/resnet_v2_imagenet_checkpoint.tar.gz) and extract to *sharpmask_tf/pretrained/*.


4. Run [sharpmask_tf/run_sharpmask_tf.py](sharpmask_tf/run_sharpmask_tf.py) with `--train` option to start training. Sample commands are in the `train` section of [sharpmask_tf\sharpmask_tf.md](sharpmask_tf\sharpmask_tf.md).

    __Example:__  
`
CUDA_VISIBLE_DEVICES=0 python3 run_sharpmask_tf.py --model all --train --train_path=/data/acamp/acamp20k/bear_jesse_32_1_w_38_1_w_224x224_5.record --checkpoint_path=trained/bear_jesse_32_1_w_38_1_w_224x224_5_batch_50 --resnet_ckpt=pretrained/resnet_v2_imagenet_checkpoint/model.ckpt-250200 --batch_size=50 --restore
`

    Number of training images should idealy be divisible by `batch_size` so it can be changed accordingly.

5. Run [sharpmask_tf/run_sharpmask_tf.py](sharpmask_tf/run_sharpmask_tf.py) with `--evaluate` option to start training. Sample commands are in the `evaluate` section of [sharpmask_tf\sharpmask_tf.md](sharpmask_tf\sharpmask_tf.md).

    __Example:__  
`
CUDA_VISIBLE_DEVICES=0 python3 run_sharpmask_tf.py --model sharpmask --evaluate  --checkpoint_path=trained/bear_jesse_32_1_w_38_1_w_224x224_5_batch_50 --batch_size=50 --restore --root_dir=/data/acamp/acamp20k/masks --seq_paths=bear_jesse_32_1_w_224x224_5_test,bear_jesse_38_1_w_224x224_5_test
`

6. The generated masks can be visualized by running [labelling_tool/visualize_masks.py](labelling_tool/visualize_masks.py).
Sample commands are in the `visualize` section of [sharpmask_tf\sharpmask_tf.md](sharpmask_tf\sharpmask_tf.md).  
    
    __Example:__  
`
python3 visualize_masks.py img_paths=bear_jesse_32_1_w,bear_jesse_38_1_w img_root_dir=/data/acamp/acamp20k mask_root_dir=../sharpmask_tf/trained/bear_jesse_32_1_w_38_1_w_224x224_5_batch_50/sharpmask_eval map_to_bbox=1 border=5 include_orig=1 include_binary=1 combine=1 out_size=0x0 write_text=0
`