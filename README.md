# ipruning
IPruning: Ising Energy-based Pruning of Deep Convolutional Neural Networks

The paper will be available online upon acceptance. 

## Requirements
- Python 3.7+
- PyTorch 1.3+
- torchvision
- CUDA 10+
- numpy

## Datasets
The following setup of benchmark datasets are used: 

(i) Fashion (gray images in 10 classes, 54k train, 6k validation, and 10k test);

(ii) Kuzushiji (gray images in 10 classes, 54k train, 6k validation, and 10k test); 

(iii) CIFAR-10 (color images in 10 classes, 45k train, 5k validation, and 10k test);

(iv) CIFAR-100 (color images in 100 classes, 45k train, 5k validation, and 10k test);

(v) Flowers (102 flower categories; each class has between 40 and 258 images; 10 images from each class for validation and 10 for test). 

The horizontal flip and Cutout augmentation methods are used for training on CIFAR and Flowers datasets. Input images are resized to 32x32 for ResNets and for 224x224 AlexNet and SqueezeNetv1.1. 

A sample of data structur is presented in data directory for Fashion dataset.

The dataloader file is uner untils directory. The Fashion and Kuzushiji are normalized in [0,1] and the other images are normalized in this setup: (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Models
The models are located in the nets directory. We mainly used the standard torchvision source codes: 

### ResNets

https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

### SqueezeNet

https://pytorch.org/hub/pytorch_vision_squeezenet/

### AlexNet

https://pytorch.org/hub/pytorch_vision_alexnet/

### Deep Compression

https://github.com/mightydeveloper/Deep-Compression-PyTorch

## Hyperparameters

We have conducted a high level hyper-parameters search in following space:

- Initial learning rate: {1,0.1,0.01}
- Adaptive learning rate gamma: {0.1,0.5,0.9}
- Learnign rate step: {25,50}
- Batch size: {64,128}
- Optimizer: {SGD, Aadadelta}
- Weight decay: {0.00001,0.000001}
- Number of epochs: {200,400}
- Early convergence threshold: {50,100,150,200}
- Initial probabilty of binary states: {0.2,0.4,0.6,0.8}


The parameters for most experiemnts are:

- Learning rate: Initial leanring rate of 1 with adaptive step learning rate decaly with gamma 0.1 at every 50 epoch 
- Optimizer: Adadelta with rho=0.9, eps=1e-06, weight_decay=0.00001
- Batch-size: 128
- Validation dataset: 10% of the training dataset selected randomly
- Number of candidate states: 8
- Early convergence threshold: 100
- Number of epochs: 200
- Initial probabilty of binary states: 0.5
- Augmentation: CropOut + RandomRotation in [0,180] for CIFAR and Flowers datasets

Some hyper-parameters analysis are provided in the paper.

### How to Train
`python3 pidropout.py -nnmodel resnet34 -dataset flowers -NS 9 -batch-size 64`

Arguments:

    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 14)')
    
    parser.add_argument('--lr', type=float, default=1., metavar='LR', help='learning rate (default: 1.0)')
    
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.7)')
    
    parser.add_argument('--lr_stepsize', type=float, default=50, metavar='M', help='Learning rate step gamma (default: 0.7)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    
    parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-ls', action='store_true', default=False, help='For Saving the current Model')
    
    parser.add_argument('--save_model', action='store_true', default=True, help='For Saving the current Model')
    
    parser.add_argument('--NS', default=32, help='pop size')
    
    parser.add_argument('--stopcounter', default=10, help='stopcounter')
    
    parser.add_argument('--threshold_early_state', default=5, help='threshold_early_state')
    
    parser.add_argument('--pre_trained_f', default=False, help='load pretrained weights')
    
    parser.add_argument('--scheduler_m', default='StepLR', help='lr scheduler')
    
    parser.add_argument('--optimizer_m', default="Adadelta", help='optimizer model')
    
    parser.add_argument('--nnmodel', default="resnet34", help='original neural net to prune {'resnet18','resnet34','resnet50','resnet101'} ')
    
    parser.add_argument('--dataset', default="flowers", help='dataset {'fashion','kuzushiji','cifar10','cifar100','flowers'}')
    
    parser.add_argument('--model', default="ising", help='ising vs. original {'ising','simple'}')
    

## Results
The results are average of five independant executions. More results are provided in the paper.

<img src="https://github.com/sparsifai/ipruning/blob/master/pngs/res.png" data-canonical-src="https://github.com/sparsifai/ipruning/blob/master/pngs/res.png" width="400" height="380" />

## Docker
A docker container will be pushed asap.

## Parallel Implementation
The current version of the optimization phase is written in NumPy as a POC for fast implementation. The running time is slower than the original model. A parallel version will be published, after publication of the paper. 


