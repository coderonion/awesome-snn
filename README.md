# Awesome-SNN
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repository lists some awesome SNN(Spiking Neural Network) projects.

## Contents
- [Awesome-SNN](#awesome-SNN)
    - [Frameworks](#frameworks)
    - [Applications](#applications)
      - [Object Detection](#object-detection)
      - [Adversarial Attack and Defense](#adversarial-attack-and-defense)
    - [Others](#others)

## Frameworks

  - [bindsnet](https://github.com/BindsNET/bindsnet) <img src="https://img.shields.io/github/stars/BindsNET/bindsnet?style=social"/>) : Simulation of spiking neural networks (SNNs) using PyTorch.

  - [Spiking-Neural-Network](https://github.com/Shikhargupta/Spiking-Neural-Network) <img src="https://img.shields.io/github/stars/Shikhargupta/Spiking-Neural-Network?style=social"/>) : This is the python implementation of hardware efficient spiking neural network.

  - [norse](https://github.com/norse/norse) <img src="https://img.shields.io/github/stars/norse/norse?style=social"/>) : Deep learning with spiking neural networks (SNNs) in PyTorch.

  - [spikingjelly](https://github.com/fangwei123456/spikingjelly) <img src="https://img.shields.io/github/stars/fangwei123456/spikingjelly?style=social"/>) : SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch.

  - [snn_toolbox](https://github.com/NeuromorphicProcessorProject/snn_toolbox) <img src="https://img.shields.io/github/stars/NeuromorphicProcessorProject/snn_toolbox?style=social"/>) : Toolbox for converting analog to spiking neural networks (ANN to SNN), and running them in a spiking neuron simulator.

  - [Spiking-Neural-Network-SNN-with-PyTorch-where-Backpropagation-engenders-STDP](https://github.com/guillaume-chevalier/Spiking-Neural-Network-SNN-with-PyTorch-where-Backpropagation-engenders-STDP) <img src="https://img.shields.io/github/stars/guillaume-chevalier/Spiking-Neural-Network-SNN-with-PyTorch-where-Backpropagation-engenders-STDP?style=social"/>) : Spiking Neural Network (SNN) with PyTorch : towards bridging the gap between deep learning and the human brain.  

  - [PySNN](https://github.com/BasBuller/PySNN) <img src="https://img.shields.io/github/stars/BasBuller/PySNN?style=social"/>) : Efficient Spiking Neural Network framework, built on top of PyTorch for GPU acceleration.

  - [yjwu17/STBP-for-training-SpikingNN](https://github.com/yjwu17/STBP-for-training-SpikingNN) <img src="https://img.shields.io/github/stars/yjwu17/STBP-for-training-SpikingNN?style=social"/>) : Spatio-temporal BP for spiking neural networks.

  - [thiswinex/STBP-simple](https://github.com/thiswinex/STBP-simple) <img src="https://img.shields.io/github/stars/thiswinex/STBP-simple?style=social"/>) : A simple direct training implement for SNNs using Spatio-Temporal Backpropagation.

  - [ZLkanyo009/STBP-train-and-compression](https://github.com/ZLkanyo009/STBP-train-and-compression) <img src="https://img.shields.io/github/stars/ZLkanyo009/STBP-train-and-compression?style=social"/>) : STBP is a way to train SNN with datasets by Backward propagation.Using this Repositories allows you to train SNNS with STBP and quantize SNNS with QAT to deploy to neuromorphological chips like Loihi and Tianjic.  

  - [SPAIC](https://github.com/ZhejianglabNCRC/SPAIC) <img src="https://img.shields.io/github/stars/ZhejianglabNCRC/SPAIC?style=social"/>) : Spike-based artificial intelligence computing platform. "Darwin-S: A Reference Software Architecture for Brain-Inspired Computers". (**[IEEE Computer 2022](https://ieeexplore.ieee.org/abstract/document/9771131)**)


## Applications

  - ### Object Detection

    - [Spiking-YOLOv3](https://github.com/cwq159/PyTorch-Spiking-YOLOv3) <img src="https://img.shields.io/github/stars/cwq159/PyTorch-Spiking-YOLOv3?style=social"/> : A PyTorch implementation of Spiking-YOLOv3. Two branches are provided, based on two common PyTorch implementation of YOLOv3([ultralytics/yolov3](https://github.com/ultralytics/yolov3) & [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)), with support for Spiking-YOLOv3-Tiny at present. (**[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6787)**)

    - [fjcu-ee-islab/Spiking_Converted_YOLOv4](https://github.com/fjcu-ee-islab/Spiking_Converted_YOLOv4) <img src="https://img.shields.io/github/stars/fjcu-ee-islab/Spiking_Converted_YOLOv4?style=social"/> : Object Detection Based on Dynamic Vision Sensor with Spiking Neural Network.

    - [Zaabon/spiking_yolo](https://github.com/Zaabon/spiking_yolo) <img src="https://img.shields.io/github/stars/Zaabon/spiking_yolo?style=social"/> : This project is a combined neural network utilizing an spiking CNN with backpropagation and YOLOv3 for object detection.

    - [Dignity-ghost/PyTorch-Spiking-YOLOv3](https://github.com/Dignity-ghost/PyTorch-Spiking-YOLOv3) <img src="https://img.shields.io/github/stars/Dignity-ghost/PyTorch-Spiking-YOLOv3?style=social"/> : A modified repository based on [Spiking-YOLOv3](https://github.com/cwq159/PyTorch-Spiking-YOLOv3) and [YOLOv3](https://pjreddie.com/darknet/yolo), which makes it suitable for VOC-dataset and YOLOv2.

  - ### Adversarial Attack and Defense

    - [ssharmin/spikingNN-adversarial-attack](https://github.com/ssharmin/spikingNN-adversarial-attack) <img src="https://img.shields.io/github/stars/ssharmin/spikingNN-adversarial-attack?style=social"/> : FGSM and PGD adversarial attack on Spiking Neural Network (SNN).


## Others

  - [XDUSPONGE/SNN_benchmark](https://github.com/XDUSPONGE/SNN_benchmark) <img src="https://img.shields.io/github/stars/XDUSPONGE/SNN_benchmark?style=social"/> : Spiking Neural Network Paper List. 

  - [amirHossein-Ebrahimi/awesome-computational-neuro-science](https://github.com/amirHossein-Ebrahimi/awesome-computational-neuro-science) <img src="https://img.shields.io/github/stars/amirHossein-Ebrahimi/awesome-computational-neuro-science?style=social"/> : A curated list of awesome Go frameworks, libraries, and software + First class pure python Tutorial Series for Spiking Neural Networks 🔥. 

  - [vvvityaaa/awesome-spiking-neural-networks](https://github.com/vvvityaaa/awesome-spiking-neural-networks) <img src="https://img.shields.io/github/stars/vvvityaaa/awesome-spiking-neural-networks?style=social"/> : A curated list of materials for Spiking Neural Networks, 3rd generation of artificial neural networks. 

  - [TheAwesomeAndy/Awesome_SNN_Reads](https://github.com/TheAwesomeAndy/Awesome_SNN_Reads) <img src="https://img.shields.io/github/stars/TheAwesomeAndy/Awesome_SNN_Reads?style=social"/> : A list of little and big reads about SNN's. 


