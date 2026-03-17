For library:
 
PREREQUISITES
    CUDA 7.5 or higher version and a GPU of compute capability 3.0 or higher are required.

ALL PLATFORMS
    Extract the cuDNN archive to a directory of your choice, referred to below as <installpath>.
    Then follow the platform-specific instructions as follows.

LINUX
    cd <installpath>/lib
    export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
    Add <installpath> to your build and link process by adding -I<installpath>/include to your compile line and -L<installpath>/lib -lcudnn to your link line.

OS X
    cd <installpath>/lib
    export DYLD_LIBRARY_PATH=`pwd`:$DYLD_LIBRARY_PATH
    Add <installpath> to your build and link process by adding -I<installpath>/include to your compile line and -L<installpath>/lib -lcudnn to your link line.

WINDOWS
    Add <installpath>\bin to the PATH environment variable.
    In your Visual Studio project properties, add <installpath>\include to the Include Directories and Library Directories lists and add cudnn.lib to Linker->Input->Additional Dependencies.

ANDROID
    adb root
    Create a target directory on the Android device:
        adb shell "mkdir -p <target dir>"
    Copy cuDNN library files over to the Android device:
        adb push <installpath> <target dir>
    Export LD_LIBRARY_PATH on target:
        cd <target dir>/lib
        export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
 
For deb:
 
PREREQUISITES

    CUDA 7.5 or higher version and a GPU of compute capability 3.0 or higher are required.
 
SUPPORTED PLATFORMS
    Ubuntu 14.04, Ubuntu 16.04, POWER8
 
Then follow the platform-specific instructions as follows.
 
1.  Install Runtime library
sudo dpkg -i $(runtime library deb)
 
2.  Install developer library
sudo dpkg -i $(developer library deb)
 
3.  Install code samples and user guide
sudo dpkg -i $(document library deb)
