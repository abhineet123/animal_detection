#!/usr/bin/env bash

sudo apt-get -y update
sudo apt-get -y upgrade

sudo apt-get -y install build-essential checkinstall cmake pkg-config yasm
sudo apt-get -y install git gfortran
sudo apt-get -y install libjpeg8-dev libjasper-dev libpng12-dev 

sudo apt-get -y install libtiff5-dev
 
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt-get -y install libxine2-dev libv4l-dev
sudo apt-get -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get -y install qt5-default libgtk2.0-dev libtbb-dev
sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt-get -y install libvorbis-dev libxvidcore-dev
sudo apt-get -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get -y install x264 v4l-utils
 
# Optional dependencies
sudo apt-get -y install libprotobuf-dev protobuf-compiler
sudo apt-get -y install libgoogle-glog-dev libgflags-dev
sudo apt-get -y install libgphoto2-dev libhdf5-dev doxygen

sudo apt-get -y install pyqt5-dev-tools

wget https://github.com/opencv/opencv/archive/3.4.2.zip
unzip 3.4.2.zip
rm 3.4.2.zip
wget https://github.com/opencv/opencv_contrib/archive/3.4.2.zip
unzip 3.4.2.zip
rm 3.4.2.zip

cd opencv-3.4.2
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.2/modules \
	  -D WITH_CUDA=ON ..

