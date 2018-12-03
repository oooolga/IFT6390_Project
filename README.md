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
<pre>
  usage: main.py [-h] [-lr LEARNING_RATE] [--batch_size BATCH_SIZE]
               [--test_batch_size TEST_BATCH_SIZE] [--epochs EPOCHS]
               [--seed SEED] [--weight_decay WEIGHT_DECAY] --model_name
               MODEL_NAME [--load_model LOAD_MODEL] [--optimizer {Adam,SGD}]
               [--dataset {CIFAR,FMNIST,EMNIST}] [--model {CNN,NN,Regression}]
               [--plot_freq PLOT_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate.
  --batch_size BATCH_SIZE
                        Mini-batch size for training.
  --test_batch_size TEST_BATCH_SIZE
                        Mini-batch size for testing.
  --epochs EPOCHS       Total number of epochs.
  --seed SEED           Random number seed.
  --weight_decay WEIGHT_DECAY
                        Weight decay.
  --model_name MODEL_NAME
                        Model name.
  --load_model LOAD_MODEL
                        Load model path.
  --optimizer {Adam,SGD}
                        Optimizer type.
  --dataset {CIFAR,FMNIST,EMNIST}
                        Dataset choice.
  --model {CNN,NN,Regression}
                        Model type.
  --plot_freq PLOT_FREQ
                        plot_freq
</pre>


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