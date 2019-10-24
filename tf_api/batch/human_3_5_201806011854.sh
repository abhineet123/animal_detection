CUDA_VISIBLE_DEVICES=0 python3 infer.py ckpt_path=pre_trained_models/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb  batch_size=1 labels_path=data/mscoco_label_map.pbtxt src_path=videos/human3 save_path=results/faster_rcnn_nas_coco_2018_01_28_ots/human3.mp4  save_fmt=2 n_frames=0 codec=H264

CUDA_VISIBLE_DEVICES=0 python3 infer.py ckpt_path=pre_trained_models/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb  batch_size=1 labels_path=data/mscoco_label_map.pbtxt src_path=videos/human5 save_path=results/faster_rcnn_nas_coco_2018_01_28_ots/human5.mkv  save_fmt=2 n_frames=0 codec=H264

