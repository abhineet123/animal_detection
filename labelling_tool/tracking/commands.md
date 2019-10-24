# issues
`
tensorflow/core/kernels/conv_ops.cc:717] Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms)
`

sudo apt-get purge nvidia*
sudo apt-get install libcuda1-390 nvidia-390 nvidia-390-dev nvidia-prime nvidia-settings

sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp35-cp35m-linux_x86_64.whl

## solution       @ issues

sudo rm -rf ~/.nv*
https://devtalk.nvidia.com/default/topic/1029297/gpu-accelerated-libraries/cudnn_status_internal_error-when-using-convolution/


# local commands

## siam_mask       @ local_commands

python3 tracking/Server.py --cfg=tracking/cfg/params.cfg --patch_tracker.tracker_type=3

## da_siam_rpn_tracker       @ local_commands

python3 tracking/Server.py --cfg=tracking/cfg/params.cfg --patch_tracker.tracker_type=4

python3 tracking/Server.py --cfg=/home/abhineet/acamp_code/labelling_tool/tracking/cfg/params.cfg --server.patch_tracker.mtf_cfg_dir=/home/abhineet/acamp_code/labelling_tool/tracking/cfg/mtf

python3 Server.py --cfg=/home/abhineet/acamp_code/labelling_tool/tracking/cfg/params.cfg --server.patch_tracker.mtf_cfg_dir=/home/abhineet/acamp_code/labelling_tool/tracking/cfg/mtf

Command "/home/abhineet/ve-tracking/bin/python2.7 -u -c "import setuptools, tokenize;__file__='/tmp/pip-install-Kgvpst/matplotlib/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" install --record /tmp/pip-record-5Q2dfh/install-record.txt --single-version-externally-managed --compile --install-headers /home/abhineet/ve-tracking/include/site/python2.7/matplotlib" failed with error code 1 in /tmp/pip-install-Kgvpst/matplotlib/

source ~/ve-tracking/bin/activate

  Could not find a version that satisfies the requirement tensorflow-cpu==1.1.0 (from -r requirements.txt (line 2)) (from versions: )
No matching distribution found for tensorflow-cpu==1.1.0 (from -r requirements.txt (line 2))

python2 Server.py --cfg=/home/abhineet/acamp_code/labelling_tool/tracking/cfg/params.cfg --server.patch_tracker.mtf_cfg_dir=/home/abhineet/acamp_code/labelling_tool/tracking/cfg/mtf

python3 Server.py --cfg=/home/abhineet/acamp_code/labelling_tool/tracking/cfg/params.cfg --patch_tracker.mtf_cfg_dir=/home/abhineet/acamp_code/labelling_tool/tracking/cfg/mtf

python3 Server.py --cfg=/home/abhineet/H/UofA/Acamp/code/labelling_tool/tracking/cfg/params.cfg --patch_tracker.mtf_cfg_dir=/home/abhineet/H/UofA/Acamp/code/labelling_tool/tracking/cfg/mtf

# remote commands

python3 Server.py --mode=1 --cfg=/home/abhineet/acamp/acamp_code/labelling_tool/tracking/cfg/params.cfg --img_path=/home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_1_1 --id_number=1 --init_frame_id=0 --init_bbox=281,102,426,257 --patch_tracker.show=0

python3 Server.py --mode=1 --cfg=/home/abhineet/acamp_code_non_root/labelling_tool/tracking/cfg/remote_params.cfg --img_path=/home/abhineet/acamp/acamp_code/object_detection/videos/grizzly_bear_1_4 --id_number=1 --init_frame_id=0 --init_bbox=204,122,412,239 --patch_tracker.tracker_type=2 --patch_tracker.cv_tracker_type=2 --patch_tracker.show=0
