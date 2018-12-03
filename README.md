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
               [--test_batch_size TEST_BATCH_SIZE] [--epochs EPOCHS]<br />
               [--seed SEED] [--weight_decay WEIGHT_DECAY] --model_name<br />
               MODEL_NAME [--load_model LOAD_MODEL] [--optimizer {Adam,SGD}]<br />
               [--load_all_train] [--dataset {CIFAR,FMNIST,EMNIST}]<br />
               [--model {CNN,NN,Regression}] [--plot_freq PLOT_FREQ]<br />

optional arguments:<br />
  -h, --help            show this help message and exit<br />
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE<br />
                        Learning rate.<br />
  --batch_size BATCH_SIZE<br />
                        Mini-batch size for training.<br />
  --test_batch_size TEST_BATCH_SIZE<br />
                        Mini-batch size for testing.<br />
  --epochs EPOCHS       Total number of epochs.<br />
  --seed SEED           Random number seed.<br />
  --weight_decay WEIGHT_DECAY<br />
                        Weight decay.<br />
  --model_name MODEL_NAME<br />
                        Model name.<br />
  --load_model LOAD_MODEL<br />
                        Load model path.<br />
  --optimizer {Adam,SGD}<br />
                        Optimizer type.<br />
  --load_all_train      Load all data as train flag.<br />
  --dataset {CIFAR,FMNIST,EMNIST}<br />
                        Dataset choice.<br />
  --model {CNN,NN,Regression}<br />
                        Model type.<br />
  --plot_freq PLOT_FREQ<br />
                        plot_freq<br />

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