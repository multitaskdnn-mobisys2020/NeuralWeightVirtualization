# Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization

## Introduction
This is an open-source repository of the paper titled "***Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization***", which is submitted to [MobiSys 2020](https://www.sigmobile.org/mobisys/2020/). It enables fast and scalable in-memory multitask deep learning on memory-constrained embedded systems by (1) packing multiple DNNs into a fixed-sized main memory whose combined memory requirement is larger than the main memory, and (2) enabling fast in-memory execution of the DNNs. 

This repository implements (1) *virtualization of weight parameters* of multiple heterogeneous DNNs of arbitrary network architectures, and (2) *in-memory execution and context-switching* of deep neural network (DNN) tasks. For the reviewers' convenience, we provide a step-by-step guideline of the weight virtualization and in-memory execution of the five DNN that are used for the multitask learning IoT device, one of the application systems we implemented in the paper. The five DNNs are [MNIST](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), [GoogleSpeechCommands (GSC)](https://arxiv.org/abs/1804.03209), [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf), [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf).

&nbsp;
## Software Install and Setup
Neural weight virtualization is implemented by using Python, TensorFlow, and NVIDIA CUDA (custom TensorFlow operation). The TensorFlow version should be lower than or equal to 1.13.2; the latest version (1.14) seems to have a problem of executing custom operations. We used Tensorflow 1.13.1 and Python 2.7. A GPU is required to perform the weight virtualization (i.e., weight-page matching and optimization) as well as in-memory execution of GPU RAM.    

**Step 1.** Install [Python (>= 2.7)](https://www.python.org/downloads/).

**Step 2.** Install [Tensorflow (<= 1.13.2)](https://www.tensorflow.org/).

**Step 3.** Clone this NeuralWeightVirtualization repository.
```sh
$ git clone https://github.com/multitaskdnn-mobisys2020/NeuralWeightVirtualization.git
Cloning into 'NeuralWeightVirtualization'...
remote: Enumerating objects: 225, done.
remote: Counting objects: 100% (225/225), done.
remote: Compressing objects: 100% (178/178), done.
remote: Total 225 (delta 90), reused 164 (delta 42), pack-reused 0
Receiving objects: 100% (225/225), 11.81 MiB | 15.90 MiB/s, done.
Resolving deltas: 100% (90/90), done.
```
