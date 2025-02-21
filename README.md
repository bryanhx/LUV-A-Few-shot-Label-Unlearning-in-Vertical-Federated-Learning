# This is an official implementation of the paper LUV:A Few-shot Label Unlearning in Vertical Federated Learning

## Datasets
Download the following datasets from the link provided.
Place the datasets in the .\data directory.

ModelNet: https://drive.google.com/drive/folders/14WZ7oaobP4STJkhHDWLHo6LDd9U994FX?usp=sharing

Brain Tumor MRI: https://drive.google.com/drive/folders/1gFVOAGlUh-sCl-wbDzzrM9G_2UwtMCHB?usp=sharing

Yahoo Answer : https://drive.google.com/drive/folders/1Frwb-ozdsDCSwUbGKuXsj5bCbd3hIp8K?usp=sharing


## Commands to train VFL model:
### CIFAR10 Resnet18
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

