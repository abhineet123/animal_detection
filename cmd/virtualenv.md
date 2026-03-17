# python3

```
virtualenv --no-site-packages -p python3  python3

source python3/bin/activate
```

# python2

```
virtualenv --no-site-packages -p python2  ~/python2

source python2/bin/activate
```

# opencv

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D PYTHON_EXECUTABLE=$HOME/abhineet/python2/bin/python \
      -D PYTHON_PACKAGES_PATH=$HOME/abhineet/python2/lib/python2.7/site-packages \
      -D PYTHON3_EXECUTABLE=$HOME/abhineet/python3/bin/python \
      -D PYTHON3_PACKAGES_PATH=$HOME/abhineet/python3/lib/python3.5/site-packages \	  
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.1/modules \
	  -D WITH_CUDA=ON ..
 ``` 