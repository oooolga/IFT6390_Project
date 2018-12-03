# IFT6390_Project

## Contact Information
Olga (Ge Ya) Xu

olga.xu823@gmail.com

## Dependencies
* torchvision==0.2.0
* matplotlib==2.2.2
* six==1.11.0
* torch.egg==info
* tqdm==4.23.0
* numpy==1.14.2
* Pillow==5.3.0
* torch==0.4.1.post2

## Code Setup
```
git clone https://github.com/oooolga/IFT6390_Project.git
cd IFT6390_Project
mkdir saved_models results
```

## Train Models
```
python main.py ...
```
usage: main.py [-h] [-lr LEARNING_RATE] [--batch_size BATCH_SIZE]<br />
_______________[--test_batch_size TEST_BATCH_SIZE] [--epochs EPOCHS]<br />
_______________[--seed SEED] [--weight_decay WEIGHT_DECAY] --model_name<br />
_______________MODEL_NAME [--load_model LOAD_MODEL] [--optimizer {Adam,SGD}]<br />
_______________[--load_all_train] [--dataset {CIFAR,FMNIST,EMNIST}]<br />
_______________[--model {CNN,NN,Regression}] [--plot_freq PLOT_FREQ]<br />

optional arguments:<br />
__-h, --help____________show this help message and exit<br />
__-lr LEARNING_RATE, --learning_rate LEARNING_RATE<br />
________________________Learning rate.<br />
__--batch_size BATCH_SIZE<br />
________________________Mini-batch size for training.<br />
__--test_batch_size TEST_BATCH_SIZE<br />
________________________Mini-batch size for testing.<br />
__--epochs EPOCHS_______Total number of epochs.<br />
__--seed SEED___________Random number seed.<br />
__--weight_decay WEIGHT_DECAY<br />
________________________Weight decay.<br />
  --model_name MODEL_NAME<br />
________________________Model name.<br />
__--load_model LOAD_MODEL<br />
________________________Load model path.<br />
__--optimizer {Adam,SGD}<br />
________________________Optimizer type.<br />
__--load_all_train______Load all data as train flag.<br />
__--dataset {CIFAR,FMNIST,EMNIST}<br />
________________________Dataset choice.<br />
__--model {CNN,NN,Regression}<br />
________________________Model type.<br />
__--plot_freq PLOT_FREQ<br />
________________________plot_freq<br />


## Examples
### Training a CNN model on CIFAR with SGD
```
python main.py -lr 0.01 --epochs 100 --weight_decay 5e-4 --model_name CIFAR_CNN --model CNN --optimizer SGD --dataset CIFAR
```
Model type: CNN<br />
Dataset: CIFAR<br />
Optimizer type: SGD<br />
Learning rate: 0.01<br />
Total number of epochs: 100<br />
Learning rate: 0.01<br />
Weight decay: 0.0005<br />
Batch size: 50<br />
Plot frequency: 5<br />

### Training CIFAR by loading a pre-trained model
```
python main.py -lr 0.01 --epochs 100 --weight_decay 5e-4 --model_name CIFAR_CNN --model CNN --optimizer SGD --dataset CIFAR --load_model saved_models/CIFAR_CNN.pt
```