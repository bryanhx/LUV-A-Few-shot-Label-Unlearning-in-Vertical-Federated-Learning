# This is an official implementation of the paper LUV:A Few-shot Label Unlearning in Vertical Federated Learning

## Getting Start
### Datasets
Download the following datasets from the link provided.
Place the datasets in the .\data directory.

ModelNet: https://drive.google.com/drive/folders/14WZ7oaobP4STJkhHDWLHo6LDd9U994FX?usp=sharing

Brain Tumor MRI: https://drive.google.com/drive/folders/1gFVOAGlUh-sCl-wbDzzrM9G_2UwtMCHB?usp=sharing

Yahoo Answer : https://drive.google.com/drive/folders/1Frwb-ozdsDCSwUbGKuXsj5bCbd3hIp8K?usp=sharing


### Commands to train VFL model:
#### CIFAR10 Resnet18
Train Full Model:
``` 
python main.py
```

Train a retrain model in 1 label unlearning scenario:
``` 
python main.py --mode=retrain
```

Train a retrain model in 2 labels unlearning scenario:
``` 
python main.py --mode=retrain --unlearn_class_num=2
```

Train a retrain model in 4 labels unlearning scenario:
```
python main.py --mode=retrain --unlearn_class_num=4
```

#### MNIST Resnet18:
Train Full Model:
```
python main.py --data=mnist
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --data=mnist --mode=retrain
```

Train a retrain model in 2 labels unlearning scenario:
```
python main.py --data=mnist --mode=retrain --unlearn_class_num=2
```

Train a retrain model in 4 labels unlearning scenario:
```
python main.py --data=mnist --mode=retrain --unlearn_class_num=4
```


#### CIFAR100 Resnet18:
Train Full Model:
```
python main.py --data=cifar100 --num_classes=100
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain
```

Train a retrain model in 2 labels unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --unlearn_class_num=2
```


Train a retrain model in 4 labels unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --unlearn_class_num=4
```


#### Yahoo Answer MixText
Train Full Model:
```
python main.py --data=yahoo --model_type=mixtext --epochs=30
```

Train a retrain model:
```
python main.py --data=yahoo --model_type=mixtext --epochs=20 --mode=retrain --unlearn_class=6
```


#### ModelNet Resnet18:
Train Full Model:
```
python main_modelnet.py --data=modelnet --num_classes=40
```

Train a retrain model in 1 label unlearning scenario:
```
python main_modelnet.py --data=modelnet --num_classes=40 --mode=retrain
```


#### Brain MRI Resnet18:
Train Full Model:
```
python main.py --data=mri --num_classes=4
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --data=mri --num_classes=4 --mode=retrain --unlearn_class=2
```

#### CIFAR10 VGG16
Train Full Model:
```
python main.py --model_type=vgg16
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --mode=retrain --model_type=vgg16
```

Train a retrain model in 2 labels unlearning scenario:
```
python main.py --mode=retrain --unlearn_class_num=2 --model_type=vgg16
```


Train a retrain model in 4 labels unlearning scenario:
```
python main.py --mode=retrain --unlearn_class_num=4 --model_type=vgg16
```


#### CIFAR100 VGG16:
Train Full Model:
```
python main.py --data=cifar100 --num_classes=100 --model_type=vgg16
```

Train a retrain model in 1 label unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --model_type=vgg16
```

Train a retrain model in 2 labels unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --unlearn_class_num=2 --model_type=vgg16
```

Train a retrain model in 4 labels unlearning scenario:
```
python main.py --data=cifar100 --num_classes=100 --mode=retrain --unlearn_class_num=4 --model_type=vgg16
```

### Before running the command for unlearning, change the saved model path directory in the torch.load() code from the unlearn python file.
### Command for unlearning


