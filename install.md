<!-- MarkdownTOC -->

- [install nvidia drivers](#install_nvidia_driver_s_)
  - [418_for_cuda_10.1       @ install_nvidia_drivers](#418_for_cuda_10_1___install_nvidia_drivers_)
  - [410_for_cuda_10.0       @ install_nvidia_drivers](#410_for_cuda_10_0___install_nvidia_drivers_)
  - [396_for_cuda_9       @ install_nvidia_drivers](#396_for_cuda_9___install_nvidia_drivers_)
- [install cuda](#install_cuda_)
  - [ubuntu_16.04       @ install_cuda](#ubuntu_16_04___install_cuda_)
    - [cuda_10.0_for_tensorflow_1.14       @ ubuntu_16.04/install_cuda](#cuda_10_0_for_tensorflow_1_14___ubuntu_16_04_install_cud_a_)
    - [tensorflow_official_jjj       @ ubuntu_16.04/install_cuda](#tensorflow_official_jjj___ubuntu_16_04_install_cud_a_)
  - [ubuntu_18.04       @ install_cuda](#ubuntu_18_04___install_cuda_)
    - [tensorflow_official       @ ubuntu_18.04/install_cuda](#tensorflow_official___ubuntu_18_04_install_cud_a_)
- [install cudnn](#install_cudn_n_)
  - [cuda_10.0       @ install_cudnn](#cuda_10_0___install_cudn_n_)
    - [cudnn_7.4.1       @ cuda_10.0/install_cudnn](#cudnn_7_4_1___cuda_10_0_install_cudn_n_)
  - [cuda_10.1       @ install_cudnn](#cuda_10_1___install_cudn_n_)
    - [cudnn_7.6.1       @ cuda_10.1/install_cudnn](#cudnn_7_6_1___cuda_10_1_install_cudn_n_)
      - [check_version       @ cudnn_7.6.1/cuda_10.1/install_cudnn](#check_version___cudnn_7_6_1_cuda_10_1_install_cudn_n_)
  - [cuda_9.0/cudnn_7_for_tensorflow_1.6_\(assuming_Ubuntu_16.04\)       @ install_cudnn](#cuda_9_0_cudnn_7_for_tensorflow_1_6__assuming_ubuntu_16_04____install_cudn_n_)
  - [cuda_8.0/cudnn_6_for_tensorflow_1.4_\(assuming_Ubuntu_14.04\)       @ install_cudnn](#cuda_8_0_cudnn_6_for_tensorflow_1_4__assuming_ubuntu_14_04____install_cudn_n_)
- [install protobuf compiler](#install_protobuf_compiler_)
- [update pip](#update_pi_p_)
- [setup python 3](#setup_python_3_)
  - [install_core_library       @ setup_python_3](#install_core_library___setup_python_3_)
    - [Ubuntu_16.04       @ install_core_library/setup_python_3](#ubuntu_16_04___install_core_library_setup_python_3_)
      - [python_3.6       @ Ubuntu_16.04/install_core_library/setup_python_3](#python_3_6___ubuntu_16_04_install_core_library_setup_python_3_)
    - [Ubuntu_14.04       @ install_core_library/setup_python_3](#ubuntu_14_04___install_core_library_setup_python_3_)
  - [install_packages       @ setup_python_3](#install_packages___setup_python_3_)
  - [install_opencv       @ setup_python_3](#install_opencv___setup_python_3_)
    - [4.1.0       @ install_opencv/setup_python_3](#4_1_0___install_opencv_setup_python_3_)
    - [3.4.5       @ install_opencv/setup_python_3](#3_4_5___install_opencv_setup_python_3_)
      - [uninstall       @ 3.4.5/install_opencv/setup_python_3](#uninstall___3_4_5_install_opencv_setup_python_3_)
  - [tensorflow       @ setup_python_3](#tensorflow___setup_python_3_)
    - [1.14       @ tensorflow/setup_python_3](#1_14___tensorflow_setup_python_3_)
      - [from_source       @ 1.14/tensorflow/setup_python_3](#from_source___1_14_tensorflow_setup_python_3_)
    - [v1.6_for_cuda_9.0       @ tensorflow/setup_python_3](#v1_6_for_cuda_9_0___tensorflow_setup_python_3_)
      - [python_3.5       @ v1.6_for_cuda_9.0/tensorflow/setup_python_3](#python_3_5___v1_6_for_cuda_9_0_tensorflow_setup_python_3_)
      - [python_3.6       @ v1.6_for_cuda_9.0/tensorflow/setup_python_3](#python_3_6___v1_6_for_cuda_9_0_tensorflow_setup_python_3_)
    - [v1.6_for_cuda_8.0       @ tensorflow/setup_python_3](#v1_6_for_cuda_8_0___tensorflow_setup_python_3_)
    - [v1.4_for_cuda_8.0       @ tensorflow/setup_python_3](#v1_4_for_cuda_8_0___tensorflow_setup_python_3_)
  - [pytorch_and_vis_tools       @ setup_python_3](#pytorch_and_vis_tools___setup_python_3_)
    - [linux_python3.6/cuda_10.0       @ pytorch_and_vis_tools/setup_python_3](#linux_python3_6_cuda_10_0___pytorch_and_vis_tools_setup_python_3_)
      - [apex       @ linux_python3.6/cuda_10.0/pytorch_and_vis_tools/setup_python_3](#apex___linux_python3_6_cuda_10_0_pytorch_and_vis_tools_setup_python_3_)
    - [windows_python3.7/cuda_10.0       @ pytorch_and_vis_tools/setup_python_3](#windows_python3_7_cuda_10_0___pytorch_and_vis_tools_setup_python_3_)
  - [theano_and_keras       @ setup_python_3](#theano_and_keras___setup_python_3_)
- [setup python 2](#setup_python_2_)
  - [install_packages       @ setup_python_2](#install_packages___setup_python_2_)
  - [install_opencv       @ setup_python_2](#install_opencv___setup_python_2_)
    - [4.1.0       @ install_opencv/setup_python_2](#4_1_0___install_opencv_setup_python_2_)
    - [3.4.5       @ install_opencv/setup_python_2](#3_4_5___install_opencv_setup_python_2_)
  - [tensorflow       @ setup_python_2](#tensorflow___setup_python_2_)
    - [1.14       @ tensorflow/setup_python_2](#1_14___tensorflow_setup_python_2_)
    - [v1.6_for_cuda_9.0       @ tensorflow/setup_python_2](#v1_6_for_cuda_9_0___tensorflow_setup_python_2_)
    - [v1.6_for_cuda_8.0       @ tensorflow/setup_python_2](#v1_6_for_cuda_8_0___tensorflow_setup_python_2_)
    - [v1.4_for_cuda_8.0       @ tensorflow/setup_python_2](#v1_4_for_cuda_8_0___tensorflow_setup_python_2_)
    - [windows/python3.7       @ tensorflow/setup_python_2](#windows_python3_7___tensorflow_setup_python_2_)
      - [no_gpu       @ windows/python3.7/tensorflow/setup_python_2](#no_gpu___windows_python3_7_tensorflow_setup_python_2_)
- [install imagemagick 7](#install_imagemagick_7_)
- [install_pycharm](#install_pycharm_)
- [install_jpeg4py](#install_jpeg4py_)

<!-- /MarkdownTOC -->

<a id="install_nvidia_driver_s_"></a>
# install nvidia drivers
1. uninstall noveau drivers:
```
sudo apt-get purge nvidia*
```
2. blacklist noveau drivers using instructions here:

https://askubuntu.com/questions/841876/how-to-disable-nouveau-kernel-driver


According to the NVIDIA developer zone: 

Create a file

```
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```
with the following contents:

```
blacklist nouveau
options nouveau modeset=0
```
Regenerate the kernel initramfs:

```
sudo update-initramfs -u
```
and finally: reboot

```
sudo shutdown -r now
```

3. install nvidia drivers:
```
sudo add-apt-repository ppa:graphics-drivers

sudo apt-get update 

```

<a id="418_for_cuda_10_1___install_nvidia_drivers_"></a>
## 418_for_cuda_10.1       @ install_nvidia_drivers
```
sudo apt-get install nvidia-418
```
<a id="410_for_cuda_10_0___install_nvidia_drivers_"></a>
## 410_for_cuda_10.0       @ install_nvidia_drivers
```
sudo apt-get install nvidia-410
```

<a id="396_for_cuda_9___install_nvidia_drivers_"></a>
## 396_for_cuda_9       @ install_nvidia_drivers
```
sudo apt-get install nvidia-396
```

4. restart
```
sudo shutdown -r now
```

<a id="install_cuda_"></a>
# install cuda

<a id="ubuntu_16_04___install_cuda_"></a>
## ubuntu_16.04       @ install_cuda

<a id="cuda_10_0_for_tensorflow_1_14___ubuntu_16_04_install_cud_a_"></a>
### cuda_10.0_for_tensorflow_1.14       @ ubuntu_16.04/install_cuda
```
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64
sudo dpkg -i cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub

sudo apt-get update
sudo apt-get install cuda-10-0
```

<a id="tensorflow_official_jjj___ubuntu_16_04_install_cud_a_"></a>
### tensorflow_official_jjj       @ ubuntu_16.04/install_cuda

```
# Add NVIDIA package repositories
# Add HTTPS support for apt-key
sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
# Issue with driver install requires creating /usr/lib/nvidia
sudo mkdir /usr/lib/nvidia

# doesn't work
# sudo apt-get install --no-install-recommends nvidia-driver-418

sudo apt-get install --no-install-recommends nvidia-418

# Reboot. Check that GPUs are visible using the command: nvidia-smi
sudo shutdown -r now
watch nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.2.24-1+cuda10.0  \
    libcudnn7-dev=7.6.2.24-1+cuda10.0


# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 \
    libnvinfer-dev=5.1.5-1+cuda10.0

```

<a id="ubuntu_18_04___install_cuda_"></a>
## ubuntu_18.04       @ install_cuda
```
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub 
sudo apt-get install cuda-toolkit-10-0
```

<a id="tensorflow_official___ubuntu_18_04_install_cud_a_"></a>
### tensorflow_official       @ ubuntu_18.04/install_cuda

```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-418
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.0.64-1+cuda10.0  \
    libcudnn7-dev=7.6.0.64-1+cuda10.0

sudo apt-get install cuda-10-0
sudo apt-get install --no-install-recommends libcudnn7=7.6.0.64-1+cuda10.0
sudo apt-get install --no-install-recommends libcudnn7-dev=7.6.0.64-1+cuda10.0


sudo apt-get install --no-install-recommends cuda-10-0



sudo apt-get install --no-install-recommends libcudnn7=7.4.1.64-1+cuda10.0 
sudo apt-get install --no-install-recommends libcudnn7-dev=7.4.1.64-1+cuda10.0 
    

# Install TensorRT. Requires that libcudnn7 is installed above.
sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 \
    && sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.1.5-1+cuda10.0

```
<a id="install_cudn_n_"></a>
# install cudnn

<a id="cuda_10_0___install_cudn_n_"></a>
## cuda_10.0       @ install_cudnn

<a id="cudnn_7_4_1___cuda_10_0_install_cudn_n_"></a>
### cudnn_7.4.1       @ cuda_10.0/install_cudnn
download: Download cuDNN v7.4.1 (Nov 8, 2018), for CUDA 10.0 from
```
https://developer.nvidia.com/rdp/cudnn-archive
```
```
apt install ./libcudnn7_7.6.1.34-1+cuda10.0_amd64.deb
```
<a id="cuda_10_1___install_cudn_n_"></a>
## cuda_10.1       @ install_cudnn
```
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1604-10-1-local-10.1.168-418.67_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-10-1-local-10.1.168-418.67_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.168-418.67/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-10-1
```

<a id="cudnn_7_6_1___cuda_10_1_install_cudn_n_"></a>
### cudnn_7.6.1       @ cuda_10.1/install_cudnn
download: cuDNN Runtime Library for Ubuntu16.04 (Deb) from 
```
https://developer.nvidia.com/rdp/cudnn-download
```
```
apt install ./libcudnn7_7.6.1.34-1+cuda10.1_amd64.deb
```

<a id="check_version___cudnn_7_6_1_cuda_10_1_install_cudn_n_"></a>
#### check_version       @ cudnn_7.6.1/cuda_10.1/install_cudnn

```
cat /usr/local/cuda/version.txt
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
cat /usr/local/cuda-10.0/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

<a id="cuda_9_0_cudnn_7_for_tensorflow_1_6__assuming_ubuntu_16_04____install_cudn_n_"></a>
## cuda_9.0/cudnn_7_for_tensorflow_1.6_(assuming_Ubuntu_16.04)       @ install_cudnn

1. download the local run file for cuda 9.0 from here:
```
wget https://developer.nvidia.com/cuda-90-download-archive
```
select:
```
Linux->x86_64->Ubuntu->16.04->runfile(local)
```
Download the base installer and all three patches
https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal
2. install it using:

```
sudo chmod +x cuda_9.0.176_384.81_linux.run
sudo ./cuda_9.0.176_384.81_linux.run
```

select no to installing nvidia driver and yes to everything else.
```
sudo chmod +x cuda_9.0.176.1_linux.run
sudo ./cuda_9.0.176.1_linux.run

sudo chmod +x cuda_9.0.176.2_linux.run
sudo ./cuda_9.0.176.2_linux.run

sudo chmod +x cuda_9.0.176.3_linux.run
sudo ./cuda_9.0.176.3_linux.run
```

3. download cudnn 7 for cuda 9.0 from here:

https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-linux-x64-v7

you will have to create an nvidia account to access the downloads

4. install it using:

```
tar -xvzf cudnn-9.0-linux-x64-v7.tgz

sudo mv cuda/include/* /usr/local/cuda-9.0/include/

sudo mv cuda/lib64/* /usr/local/cuda-9.0/lib64/
```

5. Add following lines to `~/.bashrc`:

```
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib:/usr/lib64:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/cuda/lib64:/usr/local/cuda-9.0/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib64:/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/cuda/lib64:/usr/local/cuda-9.0/targets/x86_64-linux/lib
export PATH=$PATH:$HOME/bin:$HOME/.local/bin:$HOME/bin:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:$HOME/scripts:/usr/local/cuda-9.0/bin:/usr/local/cuda-9.0/cuda/include:/usr/local/cuda-9.0/targets/x86_64-linux/include
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0
export PYTHONPATH=$PYTHONPATH:$HOME/models/research:$HOME/models/research/slim
export CUDNN_PATH=/usr/local/cuda-9.0/cuda/lib64/libcudnn.so.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/hdf5/serial
```

<a id="cuda_8_0_cudnn_6_for_tensorflow_1_4__assuming_ubuntu_14_04____install_cudn_n_"></a>
## cuda_8.0/cudnn_6_for_tensorflow_1.4_(assuming_Ubuntu_14.04)       @ install_cudnn


1. download the local run file for cuda 8.0 from here:
```
https://developer.nvidia.com/cuda-80-ga2-download-archive
```
select:
```
Linux->x86_64->Ubuntu->14.04->runfile(local)
```
Download the base installer and the patch.

2. install it using:

```
sudo chmod +x cuda_8.0.61_375.26_linux.run
sudo ./cuda_8.0.61_375.26_linux.run
```

select no to installing nvidia driver and yes to everything else.
```
sudo chmod +x cuda_8.0.61.2_linux.run
sudo ./cuda_8.0.61.2_linux.run
```

3. download cudnn 6 for cuda 8.0 from here:

https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-linux-x64-v7

you will have to create an nvidia account to access the downloads

4. install it using:

```
tar -xvzf cudnn-8.0-linux-x64-v6.0.tgz

sudo mv cuda/include/* /usr/local/cuda-8.0/include/

sudo mv cuda/lib64/* /usr/local/cuda-8.0/lib64/
```

5. Add following lines to `~/.bashrc`:

```
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib:/usr/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/cuda/lib64:/usr/local/cuda-8.0/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib64:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/cuda/lib64:/usr/local/cuda-8.0/targets/x86_64-linux/lib
export PATH=$PATH:$HOME/bin:$HOME/.local/bin:$HOME/bin:$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:$HOME/scripts:/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/cuda/include:/usr/local/cuda-8.0/targets/x86_64-linux/include
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0
export PYTHONPATH=$PYTHONPATH:$HOME/models/research:$HOME/models/research/slim
export CUDNN_PATH=/usr/local/cuda-8.0/cuda/lib64/libcudnn.so.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/hdf5/serial
```

<a id="install_protobuf_compiler_"></a>
# install protobuf compiler

```
sudo apt-get install protobuf-compiler
```

<a id="update_pi_p_"></a>
# update pip

```
wget https://bootstrap.pypa.io/get-pip.py

python get-pip.py

python3 get-pip.py
```

<a id="setup_python_3_"></a>
# setup python 3

<a id="install_core_library___setup_python_3_"></a>
## install_core_library       @ setup_python_3

<a id="ubuntu_16_04___install_core_library_setup_python_3_"></a>
### Ubuntu_16.04       @ install_core_library/setup_python_3
```
apt-get install python3-dev

```

<a id="python_3_6___ubuntu_16_04_install_core_library_setup_python_3_"></a>
#### python_3.6       @ Ubuntu_16.04/install_core_library/setup_python_3

```
apt-get install python3.6-dev

```

<a id="ubuntu_14_04___install_core_library_setup_python_3_"></a>
### Ubuntu_14.04       @ install_core_library/setup_python_3

On Ubuntu 14.04, running the above command usually installs python 3.4.3 which is too old to run the labeling tool and some of the batch scripts.
Following commands should be used to install python 3.5 instead:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.5
```

<a id="install_packages___setup_python_3_"></a>
## install_packages       @ setup_python_3

```
apt-get install python3-tk
pip3 install cython numpy scipy sklearn scikit-image pandas matplotlib screeninfo imageio pillow imutils prettytable color_transfer lxml tabulate tqdm paramiko xlwt contextlib2 paramparse
pip3 install pycocotools
pip3 install pyqt5
pip3 install -U Pillow
apt-get install python3-apt

```

**Note**: 

1. PyQt5 is known to have compatibility issues with the version of freetype library that comes with Ubuntu 14.04 that may prevent the labeling tool from working.

2. There might be a PIL version related error. Fix it by uninstalling the old version:

```
sudo apt install python3-pil
```

and running the pip3 command again

<a id="install_opencv___setup_python_3_"></a>
## install_opencv       @ setup_python_3

<a id="4_1_0___install_opencv_setup_python_3_"></a>
### 4.1.0       @ install_opencv/setup_python_3

```
pip3 install opencv-python==4.1.0.25
pip3 install opencv-contrib-python==4.1.0.25
```

<a id="3_4_5___install_opencv_setup_python_3_"></a>
### 3.4.5       @ install_opencv/setup_python_3

```
pip3 install opencv-python==3.4.5.20 opencv-contrib-python==3.4.5.20
```
<a id="uninstall___3_4_5_install_opencv_setup_python_3_"></a>
#### uninstall       @ 3.4.5/install_opencv/setup_python_3

```
pip3 uninstall opencv-python opencv-contrib-python
```


opencv 4 might have compatibility issues so opencv 3 is recommended

<a id="tensorflow___setup_python_3_"></a>
## tensorflow       @ setup_python_3


<a id="1_14___tensorflow_setup_python_3_"></a>
### 1.14       @ tensorflow/setup_python_3

```
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.14.0-cp36-cp36m-linux_x86_64.whl
```

<a id="from_source___1_14_tensorflow_setup_python_3_"></a>
#### from_source       @ 1.14/tensorflow/setup_python_3

```

pip3 install keras_applications==1.0.4 --no-deps
pip3 install keras_preprocessing==1.0.2 --no-deps

pip2 install keras_applications==1.0.4 --no-deps
pip2 install keras_preprocessing==1.0.2 --no-deps

wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh
chmod +x bazel-0.25.2-installer-linux-x86_64.sh
./bazel-0.25.2-installer-linux-x86_64.sh --user

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.14
./configure

bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip3 install /tmp/tensorflow_pkg/tensorflow-1.14.0-cp36-cp36m-linux_x86_64.whl
pip3 install /tmp/tensorflow_pkg/tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl
```

<a id="v1_6_for_cuda_9_0___tensorflow_setup_python_3_"></a>
### v1.6_for_cuda_9.0       @ tensorflow/setup_python_3

<a id="python_3_5___v1_6_for_cuda_9_0_tensorflow_setup_python_3_"></a>
#### python_3.5       @ v1.6_for_cuda_9.0/tensorflow/setup_python_3

```
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp35-cp35m-linux_x86_64.whl

```

<a id="python_3_6___v1_6_for_cuda_9_0_tensorflow_setup_python_3_"></a>
#### python_3.6       @ v1.6_for_cuda_9.0/tensorflow/setup_python_3

```
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl
```

<a id="v1_6_for_cuda_8_0___tensorflow_setup_python_3_"></a>
### v1.6_for_cuda_8.0       @ tensorflow/setup_python_3
Tensorflow does not provide an official installer for v1.6 that supports cuda 8.0 so a prebuilt installer available [here](https://drive.google.com/open?id=1m0hDMsmRn1LufagccUP5I9RoOiJPc-ML) must be used instead.

```
pip2 install --upgrade tensorflow-1.6.0-cp35-cp35m-linux_x86_64.whl
```

Please ensure that a file called ```/usr/local/cuda/lib64/libcudnn.6.0``` is available on the system. If not, run this *before* running the pip command:

```
ln -s /usr/local/cuda-8.0/lib64/libcudnn.6 /usr/local/cuda-8.0/lib64/libcudnn.6.0
```

<a id="v1_4_for_cuda_8_0___tensorflow_setup_python_3_"></a>
### v1.4_for_cuda_8.0       @ tensorflow/setup_python_3

```
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp34-cp34m-linux_x86_64.whl
```



<a id="pytorch_and_vis_tools___setup_python_3_"></a>
## pytorch_and_vis_tools       @ setup_python_3

versions > 1.0.0 have compatibility issues with yolo 3 causing NaN loss

```

pip3 install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision==0.2.2
pip3 install packaging tensorboardX visdom

pip2 install torch torchvision tensorboardX visdom
```


<a id="linux_python3_6_cuda_10_0___pytorch_and_vis_tools_setup_python_3_"></a>
### linux_python3.6/cuda_10.0       @ pytorch_and_vis_tools/setup_python_3

```
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

<a id="apex___linux_python3_6_cuda_10_0_pytorch_and_vis_tools_setup_python_3_"></a>
#### apex       @ linux_python3.6/cuda_10.0/pytorch_and_vis_tools/setup_python_3

```
git clone https://www.github.com/nvidia/apex
cd apex
python setup.py install
```

<a id="windows_python3_7_cuda_10_0___pytorch_and_vis_tools_setup_python_3_"></a>
### windows_python3.7/cuda_10.0       @ pytorch_and_vis_tools/setup_python_3

```
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl

```

<a id="theano_and_keras___setup_python_3_"></a>
## theano_and_keras       @ setup_python_3

```

git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray

mkdir Build
cd Build

cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
make
make install
cd ..

python2 setup.py build
python2 setup.py install

python3 setup.py build
python3 setup.py install

pip2 install Theano
pip2 install keras

pip3 install Theano
pip3 install keras

```


<a id="setup_python_2_"></a>
# setup python 2

```
sudo apt-get install python-dev
```

<a id="install_packages___setup_python_2_"></a>
## install_packages       @ setup_python_2

```
apt-get install python-tk
pip2 install cython numpy scipy sklearn scikit-image pandas matplotlib screeninfo imageio pillow imutils prettytable color_transfer lxml tabulate paramiko xlwt contextlib2 paramparse
pip2 install pycocotools
pip2 install PyQt4
```

<a id="install_opencv___setup_python_2_"></a>
## install_opencv       @ setup_python_2

<a id="4_1_0___install_opencv_setup_python_2_"></a>
### 4.1.0       @ install_opencv/setup_python_2

```
pip2 install opencv-python==4.1.0.25 opencv-contrib-python==4.1.0.25
```

<a id="3_4_5___install_opencv_setup_python_2_"></a>
### 3.4.5       @ install_opencv/setup_python_2

```
pip2 install opencv-python==3.4.5.20 opencv-contrib-python==3.4.5.20
```

opencv 4 might have compatibility issues so opencv 3 is recommended

<a id="tensorflow___setup_python_2_"></a>
## tensorflow       @ setup_python_2

<a id="1_14___tensorflow_setup_python_2_"></a>
### 1.14       @ tensorflow/setup_python_2
```
pip2 install tensorflow_gpu==1.14.0
```

<a id="v1_6_for_cuda_9_0___tensorflow_setup_python_2_"></a>
### v1.6_for_cuda_9.0       @ tensorflow/setup_python_2
```
pip2 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp27-none-linux_x86_64.whl
```

<a id="v1_6_for_cuda_8_0___tensorflow_setup_python_2_"></a>
### v1.6_for_cuda_8.0       @ tensorflow/setup_python_2
Tensorflow does not provide an official installer for v1.6 that supports cuda 8.0 so a prebuilt installer available [here](https://drive.google.com/open?id=11F69SNYVE4nY7Pfw6XOgQtfS_1fCFnaJ) must be used.

```
pip2 install --upgrade tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl
```
Please ensure that a file called ```/usr/local/cuda-8.0/lib64/libcudnn.6.0``` is available on the system. If not, run this *before* running the pip command:

```
ln -s /usr/local/cuda-8.0/lib64/libcudnn.6 /usr/local/cuda-8.0/lib64/libcudnn.6.0
```

<a id="v1_4_for_cuda_8_0___tensorflow_setup_python_2_"></a>
### v1.4_for_cuda_8.0       @ tensorflow/setup_python_2

```
pip2 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp27-none-linux_x86_64.whl
```

<a id="windows_python3_7___tensorflow_setup_python_2_"></a>
### windows/python3.7       @ tensorflow/setup_python_2

```
pip3 install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.13.1-cp37-cp37m-win_amd64.whl

pip3 install --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.6.0-cp37-cp37m-win_amd64.whl
```

<a id="no_gpu___windows_python3_7_tensorflow_setup_python_2_"></a>
#### no_gpu       @ windows/python3.7/tensorflow/setup_python_2

```
pip3 install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.13.1-cp37-cp37m-win_amd64.whl

pip3 install --upgrade https://download.lfd.uci.edu/pythonlibs/t4jqbe6o/libsvm-3.23-cp37-cp37m-win_amd64.whl
```

<a id="install_imagemagick_7_"></a>
# install imagemagick 7

```
wget https://www.imagemagick.org/download/ImageMagick.tar.gz
tar xvzf ImageMagick.tar.gz
cd ImageMagick-7.0.8-60/
./configure 
make -j8
sudo make install 
sudo ldconfig /usr/local/lib

```

7.0.8-60 might need adapting based on the latest available version that gets downloaded.

<a id="install_pycharm_"></a>
# install_pycharm

```
sudo snap install pycharm-community --classic
```


<a id="install_jpeg4py_"></a>
# install_jpeg4py

```
pip3 install jpeg4py 
sudo apt-get install libturbojpeg
```







