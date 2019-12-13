# Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization

## Introduction
This is an open-source repository of the paper titled "***Fast and Scalable In-memory Deep Multitask Learning via Neural Weight Virtualization***", which is submitted to [MobiSys 2020](https://www.sigmobile.org/mobisys/2020/). It enables fast and scalable in-memory multitask deep learning on memory-constrained embedded systems by (1) packing multiple DNNs into a fixed-sized main memory whose combined memory requirement is larger than the main memory, and (2) enabling fast in-memory execution of the DNNs. 

This repository implements (1) *virtualization of weight parameters* of multiple heterogeneous DNNs of arbitrary network architectures, and (2) *in-memory execution and context-switching* of deep neural network (DNN) tasks. For the reviewers' convenience, we provide a step-by-step guideline of the weight virtualization and in-memory execution of the five DNN that are used for the multitask learning IoT device, one of the application systems we implemented in the paper; the size of those DNNs are small so that the entire process of weight virtualization can be easily demonstrated in a reasonable time. The five DNNs are [MNIST](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), [GoogleSpeechCommands (GSC)](https://arxiv.org/abs/1804.03209), [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf), [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf).

&nbsp;
## Software Install and Setup
Neural weight virtualization is implemented by using Python, TensorFlow, and NVIDIA CUDA (custom TensorFlow operation). The TensorFlow version should be lower than or equal to 1.13.2; the latest version (1.14) seems to have a problem of executing custom operations. We used Python 2.7, Tensorflow 1.13.1, and CUDA 10.0. A GPU is required to perform the weight virtualization (i.e., weight-page matching and optimization) as well as in-memory execution of GPU RAM. We used NVIDIA RTX 20280 Ti GPU.

**Step 1.** Install [Python (>= 2.7)](https://www.python.org/downloads/).

**Step 2.** Install [Tensorflow (<= 1.13.2)](https://www.tensorflow.org/).

**Step 3.** Install [NVIDIA CUDA (>= 10.0)](https://developer.nvidia.com/cuda-downloads).

**Step 4.** Clone this NeuralWeightVirtualization repository.
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

## 1) Download datasets (Preliminary 1)
Download the datasets for the five DNNs by executing the downloading script (download_dataset.sh). The script uses [curl](https://curl.haxx.se/download.html) for downloading the datasets. 
```sh
$ ./download_dataset.sh 
[1/4] Downloading CIFAR10 dataset...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2604      0 --:--:-- --:--:-- --:--:--  2604
100  234M    0  234M    0     0  63.2M      0 --:--:--  0:00:03 --:--:-- 71.2M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    937      0 --:--:-- --:--:-- --:--:--   934
100  781k  100  781k    0     0  1260k      0 --:--:-- --:--:-- --:--:-- 1260k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2675      0 --:--:-- --:--:-- --:--:--  2675
100 1171M    0 1171M    0     0  67.8M      0 --:--:--  0:00:17 --:--:-- 46.5M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    665      0 --:--:-- --:--:-- --:--:--   664
100 3906k    0 3906k    0     0  4706k      0 --:--:-- --:--:-- --:--:-- 4706k
...
...
...
[4/4] Downloading SVHN dataset...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2791      0 --:--:-- --:--:-- --:--:--  2791
100  305M    0  305M    0     0  79.1M      0 --:--:--  0:00:03 --:--:-- 90.4M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    885      0 --:--:-- --:--:-- --:--:--   885
100 1017k  100 1017k    0     0  1473k      0 --:--:-- --:--:-- --:--:-- 1473k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2639      0 --:--:-- --:--:-- --:--:--  2621
100  772M    0  772M    0     0  84.7M      0 --:--:--  0:00:09 --:--:-- 93.4M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    958      0 --:--:-- --:--:-- --:--:--   955
100 2575k    0 2575k    0     0  3974k      0 --:--:-- --:--:-- --:--:-- 3974k
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0    174      0 --:--:--  0:00:02 --:--:--   174
100 85.8M    0 85.8M    0     0  26.1M      0 --:--:--  0:00:03 --:--:-- 96.3M
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   388    0   388    0     0   2380      0 --:--:-- --:--:-- --:--:--  2380
100  286k  100  286k    0     0   788k      0 --:--:-- --:--:-- --:--:--  788k
```

## 2) Prepare (and train) DNN models (Preliminary 2)
Next, we need to obtain and train DNN models for the five datasets. For the reviewers' convenience, pre-trained DNN models  
