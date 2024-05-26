# Test-Time Adaptation Coordinator for Object Detection: Leveraging In-Situ Monitoring Without Ground Truth

This repository contains the introduction of our constructed continual evolving data sequences, as well as the code for NeurIPS 2024 submitted paper: ["Test-Time Adaptation Coordinator for Object Detection: Leveraging In-Situ Monitoring Without Ground Truth"](). 

## Outline:
* [Overview](#1)
* [Data Sequence Construction](#2)
* [TACo Implementation](#3)

The rest of the repository is organized as follows. [**Section 1**](#1) gives a brief overview of TACo. [**Section 2**](#2) introduces our organized data sequences from public datasets. [**Section 3**](#3) briefly introduces the implementation of the TACo.

## 1. <span id="1"> Overview</span> 
<p align="center"><img src="https://github.com/TestTimeCoordinatorTACo/TACo/blob/main/images/TACo.jpg" width="580"\></p>
<p align="center"><strong>Figure 1. Overview of TACo design.</strong></p> 

TACo is a test-time adaptation coordinator for object detection. An overview of TACo is shown in Figure 1. We consider the continual test-time adaptation scenario. The process starts with target data sequentially provided from a continually evolving environment. An off-the-shelf source pre-trained network initializes the target network. We incorporate a monitoring head to judiciously schedule the adaptation process. Additionally, a dynamic domain bank is deployed to store model checkpoints from different detected domains and to make strategic decisions regarding when to store, aggregate, and restore model parameters. This method improves computational efficiency while maintaining or even enhancing model performance.

## 2. <span id="2"> Data Sequence Construction</span>
To evaluate our proposed method, we utilize three public datasets well-suited for object detection tasks: [ACDC](https://acdc.vision.ee.ethz.ch), [Cityscapes](https://www.cityscapes-dataset.com), and [KITTI](https://www.cvlibs.net/datasets/kitti/eval_3dobject.php). ACDC dataset includes fog, nighttime, rain, and snow scenarios. We apply the [data augmentation techniques](https://github.com/astra-vision/rain-rendering) to create varying target domains including fog, rain, and snow scenarios within Cityscapes and KITTI datasets. Together with the original images, they contains 4 scenarios, including, clear (original), fog, rain, and snow. The augmented samples are shown below:

<p align="center">
  <img src="https://github.com/TestTimeCoordinatorTACo/TACo/blob/main/images/origin.png" width="500" alt="Alt text" title="Figure 2. Example images of clear, snow, fog and rain domains in KITTI dataset.">
  <img src="https://github.com/TestTimeCoordinatorTACo/TACo/blob/main/images/snow.png" width="500" alt="Alt text" title="Figure 2. Example images of clear, snow, fog and rain domains in KITTI dataset.">
  <img src="https://github.com/TestTimeCoordinatorTACo/TACo/blob/main/images/fog.png" width="500" alt="Alt text" title="Figure 2. Example images of clear, snow, fog and rain domains in KITTI dataset.">
  <img src="https://github.com/TestTimeCoordinatorTACo/TACo/blob/main/images/rain.png" width="500" alt="Alt text" title="Figure 2. Example images of clear, snow, fog and rain domains in KITTI dataset.">
</p>
<p align="center">
    <strong>Figure 2. Example images of clear, snow, fog and rain domains in KITTI dataset.<strong>
</p>

We randomly sample sequences of 20 scenarios of KITTI dataset using repetition to simulate realistic short-term. We also randomly sample sequences of 40 scenarios of ACDC and Cityscapes datasets using repetition to simulate realistic long-term adaptation processes.

## 3. <span id="3"> TACo Implementation</span>
The experiments are conducted on a server with 4 RTX TITAN GPUs. For the object detection model used in the teacher-student learning framework, we employ Fast R-CNN. For all the experiments, we set batch size as $32$, learning rate as $0.01$, length of prediction score time window $P$ as $30$, step threshold $T$ as $5$, slope range $sl$ as $0.1$. The TACo implementation is based on [CoTTA](https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py) and [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher/tree/main). Please setup the prerequisites as follows.
```
# create conda env
conda create -n detectron2 python=3.6
# activate the enviorment
conda activate detectron2
# install PyTorch >=1.5 with GPU
conda install pytorch torchvision -c pytorch
```

Once the environment is set up, you can run TACo, execute the following command:
```
python train_net.py --num-gpus 1 --config configs/xxx/faster_rcnn_R_50_FPN_xxx_run1.yaml
```
Refer to the ```configs``` directory for a full list of options and default hyperparameters.
