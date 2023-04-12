# Super Slo Mo TF2 

[![tensorflow](https://aleen42.github.io/badges/src/tensorflow.svg)](https://www.tensorflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Tensorflow 2 implementation of ["Super SloMo: High Quality Estimation of Multiple Intermediate Frames 
for Video Interpolation" by Jiang H., Sun D., Jampani V., Yang M., Learned-Miller E. and Kautz J.](https://arxiv.org/abs/1712.00080)

<p align="center">
  <img width="552" height="310.4" src="resources/rally1_259_12.gif">
</p>

## Setup

The code is based on Tensorflow 2.1. To install all the needed dependency, run

##### Conda

```bash
conda env create -f environment.yml
source activate super-slomo
```

##### Pip 

```bash
python3 -m venv super-slomo
source super-slomo/bin/activate
pip install -r requirements.txt
```

## Inference

You can download the pre-trained model [here](https://www.dropbox.com/s/l35juwrsvcaw565/chckpnt259.zip). This model is trained for 259 epochs on 
the adobe240fps dataset. It uses the single frame prediction mode. 

To generate a slomo video run:

```bash
python super-slomo/inference.py path/to/source/video path/to/slomo/video --model path/to/chckpnt259/ckpt-259 --n_frames 20 --fps 480
```

## Train

#### Data Extraction

Before the training phase, the frames must be extracted from the original video sources. 
This code uses the adobe240fps dataset to train the model. To extract frames, run the following command:

```bash
python super-slomo/frame_extraction.py path/to/dataset path/to/destination 
```

It will use ffmepg to extract the frames and put them in the destination folder, grouped in folders of 12 consecutive frames.
If ffmpeg is not available, it falls back to slower opencv.

For info run:
```bash
python super-slomo/frame_extraction.py -h
```

#### Train the model

You can start to train the model by running:

```bash
python super-slomo/train.py path/to/frames --model path/to/checkpoints --epochs 100 --batch-size 32
```

If the `model` directory contains a checkpoint, the model will continue to train from that epoch until the total number 
of epochs provided is reached

You can also visualize the training with tensorboard, using the following command

```bash
tensorboard --logdir log --port 6006
```

and go to [https://localhost:6006](https://localhost:6006).


For info run:
```bash
python super-slomo/train.py -h
```

##### Multi-frame model

The model above predicts only one frame at time, due to hardware limitations. If you can access to powerful GPUs,
you can predict more frame with a single sample (like in the original paper). To start, clone the multi-frame branch

```bash
git clone --branch multi-frame https://github.com/Riccorl/Super-SloMo-tf2.git 
```

then, follow the instructions above to setup and extract the frames. The training command has one additional parameter `--frames`
to control the number of frames to predict:

```bash
python super-slomo/train.py path/to/frames --model path/to/checkpoints --epochs 100 --batch-size 32 --frames 9
```

## Useful links

#### Dataset links

* [Adobe 240fps](https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring)
* [Need for Speed dataset](https://ci2cv.net/nfs/index.html)
* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

#### Random notes

* [Evaluation script](https://people.cs.umass.edu/~hzjiang/projects/superslomo/UCF101_results.zip)

#### References

* [Paper](https://arxiv.org/abs/1712.00080)
* [Project Page](https://people.cs.umass.edu/~hzjiang/projects/superslomo/)
* [PyTorch implementation](https://github.com/MayankSingal/Super-SlowMo)
