# FedHash
Federated learning for image hashing <br>

# Intro

Implement deep supervised image hashing under Federated learning settings, where data is split and cannot be shared during training. We adopt CNN network to generate hash code and FedAvg as aggregation method. We compare both independent and identically distribution (IID) as well as non independent and identically distribution (non-IID) data setting. On each setting, we calculate mean average percision (mAP) to evaluate. Currently, we use CIFAR10 only.


# Quick start

```shell
cd FedHash/code/
python fed_main.py
```
### Environment

see also ```requirement.txt```

```
python=3.7.3
pytorch=1.2.0
torchvision=0.4.0
numpy=1.15.4
tensorboardx=1.4
matplotlib=3.0.1
tqdm
scipy
```

### More hyper parameters
|hyper parameters|default value|remark|
|  ----  | ----  | ---- |
|--global_epochs| 2 | number of rounds of globally training |
|--number_of_clients| 5 | number of clients |
|--frac| 1.0 |the fraction of clients selected to globally training|
|--local_epochs| 2 |the number of local training epochs|
|--local_batch_size| 256 |local batch size|
|--lr| 0.001 |learning rate|
|--gpu| 0 |To use cuda, set to a specific GPU ID. Default is set to use CPU|
|--optimizer| adam|"type of optimizer|
|--iid| 1 |Default set to IID. Set to 0 for non-IID.|
|--num_binary|12|Length of binary hash code|
|--alpha|0.01|The alpha in hashing loss function.|