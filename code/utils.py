import torch
from torchvision import datasets, transforms
import numpy as np
import copy
from mAP import compute_result, compute_mAP
def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=test_transform)

    return train_dataset, test_dataset

def cifar_iid(dataset_train, dataset_test, num_clients):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset_train: train dataset
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset_train)/num_clients)
    num_items_test = int(len(dataset_test)/num_clients)
    dict_users_train = {}
    dict_users_test = {}
    all_index_train, all_index_test = [i for i in range(len(dataset_train))], [j for j in range(len(dataset_test))]
    for i in range(num_clients):
        dict_users_train[i] = set(np.random.choice(all_index_train, num_items, replace=False))
        dict_users_test[i] = set(np.random.choice(all_index_test, num_items_test, replace=False))
        all_index_train = list(set(all_index_train) - dict_users_train[i])
        all_index_test = list(set(all_index_test) - dict_users_test[i])
    return dict_users_train, dict_users_test


def cifar_noniid(dataset_train, dataset_test, num_clients):
    num_shards, num_imgs = 200, 250
    num_items_test = int(len(dataset_test) / num_clients)
    all_idxs_test = [i for i in range(len(dataset_test))]

    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_clients)}
    dict_users_test = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)  # 所有图片的索引
    idxs = idxs.astype(int)

    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset_train.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 对齐
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 排序
    idxs = idxs_labels[0, :]  # 图片索引
    idxs2 = idxs_labels[1, :]  # 对应图片的label

    dict_user_label = {i: np.array([]) for i in range(num_clients)}

    # divide and assign
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, int(num_shards / num_clients), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        dict_users_test[i] = set(np.random.choice(all_idxs_test, num_items_test, replace=False))
        all_idxs_test = list(set(all_idxs_test) - dict_users_test[i])
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            dict_user_label[i] = np.concatenate((dict_user_label[i], idxs2[rand * num_imgs:(rand + 1) * num_imgs]),
                                                axis=0)
    for i in range(num_clients):
        dict_users[i] = dict_users[i].astype(int)

    return dict_users, dict_users_test

def hashing_loss(output, label, m, alpha=0.01): # m = 2 * length of hash code
    y = (label.unsqueeze(0) != label.unsqueeze(1)).float().view(-1)
    dist = ((output.unsqueeze(0) - output.unsqueeze(1)) ** 2).sum(dim=2).view(-1)
    loss = (1 - y) / 2 * dist + y / 2 * (m - dist).clamp(min=0)
    loss = loss.mean() + alpha * (output.abs() - 1).abs().sum(dim=1).mean() * 2
    return loss

def fedavg(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def calculate_mAP(model, train_dataloader, test_dataloader):
    trn_binary, trn_label = compute_result(train_dataloader, model.cuda())
    tst_binary, tst_label = compute_result(test_dataloader, model.cuda())
    mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label)
    print(f'retrieval mAP: {mAP:.4f}')
    return mAP
