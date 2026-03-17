#!/usr/bin/env bash
sudo apt-get update
sudo apt-get install build-essential cmake git pkg-config -y
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libgtk2.0-dev -y
sudo apt-get install libatlas-base-dev gfortran -y
cd ~
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.0.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D WITH_CUDA=OFF \
	..
make -j
sudo make install
sudo ldconfig

PYTHON_VERSION=$(python3 -c "import platform; print(platform.python_version()[:3])")
PYTHON_VERSION_NO_DOT=$(python3 -c "import platform; print(platform.python_version()[:3:2])")
sudo ln -s /usr/local/lib/python${PYTHON_VERSION}/site-packages/cv2.cpython-${PYTHON_VERSION_NO_DOT}m.so /usr/lib/python3/dist-packages/cv2.so