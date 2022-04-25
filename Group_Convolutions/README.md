# Group Convolutions & Separable Group Convolutions

## Baseline
We get the ResNet-34 baseline model from Pytorch model zoo.


## Main Training
### Imagenette
```
python group_convolutions_main_imagenette.py --resume [CHECKPOINT] --scratch[CONFIGURATION] --save [PATH TO SAVE THE RESULTS] [DATASET]
```
### ImageNet
```
python group_convolutions_main_ImageNet.py --resume [CHECKPOINT] --scratch[CONFIGURATION] --save [PATH TO SAVE THE RESULTS] [DATASET]
```
### Cifar-10
```
python group_convolutions_cifar10.py --resume [CHECKPOINT] --scratch[CONFIGURATION] --save [PATH TO SAVE THE RESULTS] [DATASET] 
```

Alternatively you can modify dataset path directly from the arguments, but for number of Group Convolutions you have to 
change the variable `number of groups` on the constructor of the resnet-34 model. There are two main constructors
the normal renset which only modifies the classic Group Convolutions and the resnet_separable_group_conv
which replaces classic group convolutions with depth-wise & point-wise convolutions.



## Models
###Network|Training method|Top-1|Top-5|Pre-Trained
#:---:|:---:|:---:|:---:|:---:
The pruned configurations as well as the pre-trained models are inside the folders of l1-norm-pruning sub-folder of the project.
Unfortunately due to the huge amount of experiments not all of them are saved in a proper way.

