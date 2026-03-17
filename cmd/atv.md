sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 71A1D0EFCFEB6281FD0437C93D5919B448457EE0

sudo apt-get update

sudo apt-get install ros-kinetic-desktop-full


sudo rosdep init

rosdep update

echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

source ~/.bashrc

sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential


sudo apt-get install ros-kinetic-jsk-recognition

sudo apt-get install ros-kinetic-jsk-visualization

sudo apt-get install ros-kinetic-ros-control

sudo apt-get install ros-kinetic-control-toolbox

sudo apt-get install ros-kinetic-nmea-msgs

sudo apt-get install ros-kinetic-grid-map

sudo apt-get install ros-kinetic-gps-common

wget https://download.qt.io/archive/qt/5.8/5.8.0/qt-opensource-linux-x64-5.8.0.run

chmod +x qt-opensource-linux-x64-5.8.0.run

./qt-opensource-linux-x64-5.8.0.run

catkin_make
