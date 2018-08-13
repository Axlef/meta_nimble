# Skeleton Detection and Gesture Recognition

This repository contains the implementation of an architecture for skeleton detection and tracking, followed by simple gesture and action recognition.

![alt-text-1](https://raw.githubusercontent.com/Axlef/meta_nimble/master/img/tracker.png) ![alt-text-2](https://raw.githubusercontent.com/Axlef/meta_nimble/master/img//clapping.png)

## Requirements

* Ubuntu 16.04 (not tested on other version or linux distrib)
* TensorRT 3
* Python 3.4 or superior

## Installation

First clone this meta repository with all the submodules:

```bash
git clone --recurse-submodules https://github.com/Axlef/meta_nimble.git
```

This system is mainly for use with the Jetson TX2, with already JetPack 3.2 installed. For other systems (compatible TensorRT), you may have to manually install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html).

The installation of each module is detailed in their README, please refer to them for instructions. This repository only contains scripts on how to use the whole pipeline, and not the module individually.

You first have to install Openpose and the python wrapper PyOpenpose. They are both packaged as a library (headers and `.so` libs). Install Openpose and PyOpenpose with their repositories instructions, then add the install path to your `LD_LIBRARY_PATH`.

The tracker and the gesture detector are both python-only package, so there's not much to install. Refer to their repositories on how to install their dependencies (with the `requirements.txt`). Finally, add the `tracker` folder and `nimble` folder to your `PYTHONPATH`. In your `.bashrc` file, add the following lines:

```bash
export PYTHONPATH=${PYTHONPATH}:/PATH_TO_META_FOLDER/nimble
export PYTHONPATH=${PYTHONPATH}:/PATH_TO_META_FOLDER/tracker
```

## Quick Run

Openpose requires a pre-trained model, converted to TensorRT format. Also note that this modified version of Openpose with TensorRT only supports a fixed resolution image, `240x320` (width x height) in the current code. Please refer to the openpose-rt README for instructions on how to update the code for another resolution and generate the associated model. Ensure that the TensorRT model is located in `openpose-models/240x320/tensorrt` with the name `rt_model.gie`.

The gesture detector (i.e. Nimble) also requires a pre-trained model. The instructions for training a model can be found in the README of the actual nimble repository. Pre-trained models are also available for different action sets. The folder `nimble-models` in the nimble repository already contains pre-trained models (one model for a set of action).

Multiple scripts are available depending on which modules you want to use (e.g. skeleton extraction only, skeleton detection and tracking etc.).

### Skeleton detection only

Detection only of the body-poses of people from a camera feed. Require only Openpose and PyOpenpose.

```bash
python3 visualization.py --openpose_dir=openpose-models/
```

### Skeleton detection and tracking

Skeletons detection and id tracking of people from a camera feed. Require Openpose, PyOpenpose and the tracker.

```bash
python3 visualization_tracker.py --openpose_dir=openpose-models/
```

### Skeleton detection and gesture recognition

Skeletons detection for gesture recognition (depending on the action set used). Require Openpose, PyOpenpose and nimble.

```bash
python3 recognition_multiprocess.py --openpose_dir=openpose-models --model=nimble-models/model_set_1/model_l2_detection --threshold=0.8
```

Currently, the action set is fixed in the script (not parameterized). You need to manually update the variable `mapping` to reflect the action set you are using. For that, you can find in the nimble model folder the mapping in a `actions.txt` file.

### Skeleton detection and gesture recognition with feedback by Pepper

Skeleton detection for gesture recognition, with feedback from Pepper using the `Versatile` module (require Choregraphe). Require Openpose, PyOpenpose and nimble.

```bash
python3 recognition_multiprocess_interaction.py --openpose_dir=openpose-models/240x320 --model=nimble-models/model_set_1/model_l2_detection --threshold=0.8 --robot_ip=IP_ROBOT
```

Currently, the action set is fixed in the script (not parameterized). You need to manually update the variable `mapping` to reflect the action set you are using. For that, you can find in the nimble model folder the mapping in a `actions.txt` file.
