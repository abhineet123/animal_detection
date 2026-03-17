REM crop patch by detection
python3 "%~dp0/tf_api_test.py" ckpt_path=pre_trained_models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person, n_objs_to_include=1 save_patches=1 extend_vertically=1 patch_ar=0.89 patch_out_root_dir=#patches_src root_dir=%1 extend_vertically=%2 patch_ar=%3

REM cd /D %curr_dir%
REM cd /D H:\UofA\Acamp\code\tf_api

REM set PYTHONPATH=%PYTHONPATH%;H:\UofA\Acamp\code\tf_api & 