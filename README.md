# Human Activity Recognition

This module detects Real time human activities such as throwing, jumping, jumping_jacks, boxing, sitting. 

**FPS 20-25 on Nvidia AGX Xavier**

## Install

### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+

```bash
$ pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v411 tensorflow-gpu
```

```bash
$cd Human-Activity-Recognition
$pip3 install requirements.txt
$sudo apt-get install swig
$cd tf_pose/pafprocess
$swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
$cd ../..
```

### Realtime Webcam

```
$ python3 run_webcam.py webcam0 abc abc
```

### Prerecorded Video

```
$ python3 run_webcam.py video-path abc abc
```

