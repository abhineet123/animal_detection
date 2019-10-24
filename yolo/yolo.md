# install

## e5g

make CUDA_INCLUDE=/usr/local/cuda-8.0/cuda/include CUDA_LIB=/usr/local/cuda-8.0/lib64 CUDNN_LIB=/usr/local/cuda-9.0/lib64 GPU=0 CUDNN=0

# data conversion

python3 csv_to_yolo.py input_path=../object_detection/data/train.csv img_path=../object_detection/images/train output_path=../object_detection/labels/train

python3 csv_to_yolo.py input_path=../object_detection/data/test.csv img_path=../object_detection/images/test output_path=../object_detection/labels/test

## count_files

python2 /home/abhineet/PTF/countFileInSubfolders.py file_ext=jpg out_file=/home/abhineet/acamp/acamp_code/yolo/darknet/acamp5k_yolo_train_list.txt folder_name=/data/acamp/acamp5k/train/images

python2 /home/abhineet/PTF/countFileInSubfolders.py file_ext=jpg out_file=/home/abhineet/acamp/acamp_code/yolo/darknet/acamp5k_yolo_test_list.txt folder_name=/data/acamp/acamp5k/test/images

# training

## v3

./darknet detector train /home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg /home/abhineet/acamp/acamp_code/yolo/cfg/yolov3.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/darknet53.conv.74 -gpus 0,1

./darknet detector train /home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg /home/abhineet/acamp/acamp_code/yolo/cfg/yolov3-train.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/darknet19_448.conv.23 -gpus 1

## v2

./darknet detector train /home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg /home/abhineet/acamp/acamp_code/yolo/cfg/yolov2.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/darknet53.conv.74 -gpus 0,1

./darknet detector train /home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg /home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/darknet19_448.conv.23 -gpus 0,1

### 5k

./darknet detector train /home/abhineet/acamp/acamp_code/yolo/cfg/acamp5k.cfg /home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/darknet19_448.conv.23 -gpus 0

## v1

./darknet detector train /home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg /home/abhineet/acamp/acamp_code/yolo/cfg/yolov1.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/darknet53.conv.74 -gpus 0,1

# testing python

## 100_on_5k

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg root_dir=/data/acamp/acamp5k/test/images save_dir=results/yolov2_100_on_5k/ gpu=0

## 5k

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/yolov2-acamp_40000.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp5k.cfg root_dir=/data/acamp/acamp5k/test/images save_dir=results/acamp5k_yolov2/ gpu=1

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/yolov2-acamp_20000.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp5k.cfg root_dir=/data/acamp/acamp5k/test/images save_dir=results/acamp5k_yolov2_20K/ gpu=1

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/yolov2-acamp_10000.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp5k.cfg root_dir=/data/acamp/acamp5k/test/images save_dir=results/acamp5k_yolov2_10K/ gpu=1

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/yolov2-acamp_900.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp5k.cfg root_dir=/data/acamp/acamp5k/test/images save_dir=results/acamp5k_yolov2_900/ gpu=1

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/yolov2-acamp_100.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp5k.cfg root_dir=/data/acamp/acamp5k/test/images save_dir=results/acamp5k_yolov2_100/ gpu=1

## test_images

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_1.mp4 save_dir=results/yolov2/

## bear

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/test/ save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_2.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_3.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_4.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_5.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_6.mp4 save_dir=results/yolov2/

## zoo_180529

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_0.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_1.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_2.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_3.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_4.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_8.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_9.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_10.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_11.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_12.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_13.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_14.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_15.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_16.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_17.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_18.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_19.MPG save_dir=results/yolov2/


python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_19.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/bear_zoo_180529_21.MPG save_dir=results/yolov2/


## deer

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_1.mp4 save_dir=results/yolov2/


python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_2.mp4 save_dir=results/yolov2/


python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_3.mp4 save_dir=results/yolov2/

## zoo_180529

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_0.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_1.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_2.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_3.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_4.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_5.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_6.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_7.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_8.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_9.MPG save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/deer_zoo_180529_10.MPG save_dir=results/yolov2/

## moose

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/moose_1.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/moose_2.mp4 save_dir=results/yolov2/

## coyote

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_1.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_2.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_3.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_4.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_5.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_6.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_7.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_8.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_9.mp4 save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_10.mp4 save_dir=results/yolov2/

### doc

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_1_1.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_1_2.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_1_3.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_3_1.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_3_2.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_3_3.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_3_4.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_4_1.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_4_2.mkv save_dir=results/yolov2/

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/coyote_doc_4_3.mkv save_dir=results/yolov2/

## marcin_180608

### bear

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_3 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_4 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_5 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_1_6 save_dir=results/yolov2/marcin


python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_3 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_4 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/bear_5 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/grizzly_bear_6_1 save_dir=results/yolov2/marcin

### moose

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_1_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_4_3 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_3 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_4 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_5 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_6 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_7 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_8 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_9 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_10 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_12 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_5_13 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_6_3 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_7_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_8_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_9_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_10_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_11_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_13_1 save_dir=results/yolov2/marcin

#### missing first time around

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_2_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/moose_12_3 save_dir=results/yolov2/marcin

### coyote

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_2 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_1_3 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_1 save_dir=results/yolov2/marcin

python3 detector_video.py weights_path=/home/abhineet/acamp/acamp_code/yolo/darknet/backup/acamp100/yolov2-train_final.weights cfg_path=/home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg meta_path=/home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg file_name=/home/abhineet/acamp/acamp_code/object_detection/videos/marcin/coyote_2_2 save_dir=results/yolov2/marcin

### zip

zr yolov2_marcin_bear_moose results/yolov2/marcin/

# testing c

## coco pretrained

./darknet detector test /home/abhineet/acamp/acamp_code/yolo/cfg/coco.data /home/abhineet/acamp/acamp_code/yolo/cfg/yolov3.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/yolov3.weights /home/abhineet/acamp/acamp_code/object_detection/videos/human/image000001.jpg -out /home/abhineet/acamp/acamp_code/yolo/results/human_image000001

./darknet detector test /home/abhineet/acamp/acamp_code/yolo/cfg/coco.data /home/abhineet/acamp/acamp_code/yolo/cfg/yolov3.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/yolov3.weights /home/abhineet/acamp/acamp_code/object_detection/videos/human/image000003.jpg -out /home/abhineet/acamp/acamp_code/yolo/results/human_image000003 -thresh 0.1

./darknet detector test /home/abhineet/acamp/acamp_code/yolo/cfg/coco.data /home/abhineet/acamp/acamp_code/yolo/cfg/yolov3.cfg /home/abhineet/acamp/acamp_code/yolo/pretrained/yolov3.weights /home/abhineet/acamp/acamp_code/object_detection/videos/human/image000002.png -out /home/abhineet/acamp/acamp_code/yolo/results/human_image000002 -thresh 0.1

## v2

./darknet detector test /home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg /home/abhineet/acamp/acamp_code/yolo/cfg/yolov2.cfg /home/abhineet/acamp/acamp_code/yolo/backup/yolov3_800.weights /home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_1_1/image000001.jpg -out /home/abhineet/acamp/acamp_code/yolo/results/grizzly_bear_1_1_image000001

./darknet detector test /home/abhineet/acamp/acamp_code/yolo/cfg/acamp.cfg /home/abhineet/acamp/acamp_code/yolo/cfg/yolov2-acamp.cfg /home/abhineet/acamp/acamp_code/yolo/backup/darknet19_448.conv.23 -gpus 0,1




