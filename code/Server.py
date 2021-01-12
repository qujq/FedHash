import torch
from utils import cifar_iid, cifar_noniid, fedavg
from Client import Client
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Server(object):
    def __init__(self, train_dataset, test_dataset, model, args, device):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.args = args
        self.device = device

        self.list_clients = []
        self.dict_train_data_for_each_client = {}
        self.dict_test_data_for_each_client = {}
        self.list_average_validation_loss = []
        self.list_average_test_loss = []

    def setup_data(self):
        if self.args.dataset == 'cifar':
            if self.args.iid == 1:
                self.dict_train_data_for_each_client, self.dict_test_data_for_each_client = cifar_iid(
                    dataset_train=self.train_dataset,
                    dataset_test=self.test_dataset,
                    num_clients=self.args.number_of_clients)
            elif self.args.iid == 0:
                self.dict_train_data_for_each_client, self.dict_test_data_for_each_client = cifar_noniid(
                    dataset_train=self.train_dataset,
                    dataset_test=self.test_dataset,
                    num_clients=self.args.number_of_clients)

    def setup_clients(self):
        for i in range(self.args.number_of_clients):
            train_dataloder_for_this_client, validate_dataloader_for_this_client, test_dataloader_for_this_client = \
                self.assign_dataloader_for_each_client(client_id=i)

            client = Client(client_id=i,
                            model=copy.deepcopy(self.model),
                            args=self.args,
                            dataloader_train=train_dataloder_for_this_client,
                            dataloader_validate=validate_dataloader_for_this_client,
                            dataloader_test=test_dataloader_for_this_client,
                            device=self.device,
                            loss_function='hashing loss'
                            )
            self.list_clients.append(client)

    def assign_dataloader_for_each_client(self, client_id):
        train_dataloder_for_this_client, validate_dataloader_for_this_client, test_dataloader_for_this_client = \
            self.setup_dataloader(list(self.dict_train_data_for_each_client[client_id]),
                                  list(self.dict_test_data_for_each_client[client_id]))

        return train_dataloder_for_this_client, validate_dataloader_for_this_client, test_dataloader_for_this_client

    def setup_dataloader(self, list_train_data_index_for_this_client, list_test_data_index_for_this_client):

        idxs_train = list_train_data_index_for_this_client[:int(1.0 * len(list_train_data_index_for_this_client))]
        idxs_val = list_train_data_index_for_this_client[int(0.9 * len(list_train_data_index_for_this_client)):
                                              int(1 * len(list_train_data_index_for_this_client))]

        idxs_test = list_test_data_index_for_this_client[:int(len(list_test_data_index_for_this_client))]

        trainloader = DataLoader(DatasetSplit(self.train_dataset, idxs_train),
                                 batch_size=self.args.local_batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(self.train_dataset, idxs_val),
                                 batch_size=self.args.local_batch_size, shuffle=False)

        testloader = DataLoader(DatasetSplit(self.test_dataset, idxs_test),
                                batch_size=self.args.local_batch_size, shuffle=False)

        return trainloader, validloader, testloader

    def train(self):
        num_of_clients_chosen_for_train = max(int(self.args.frac * self.args.number_of_clients), 1)

        for epoch in range(self.args.global_epochs):
            list_index_of_clients_for_train = np.random.choice(range(self.args.number_of_clients),
                                                               num_of_clients_chosen_for_train,
                                                               replace=False)
            list_index_of_clients_for_train.sort()

            list_all_clients_feedback_model_parameters = []
            for client_id in list_index_of_clients_for_train:
                self.list_clients[client_id].model = copy.deepcopy(self.model)
                dict_feedback_model_parameters = self.list_clients[client_id].train(epoch)
                list_all_clients_feedback_model_parameters.append(dict_feedback_model_parameters)

            dict_new_global_model_parameters = fedavg(list_all_clients_feedback_model_parameters)
            self.model.load_state_dict(dict_new_global_model_parameters)
            self.validate(list_index_of_clients_for_train)

        list_index_of_clients_for_test = np.random.choice(range(self.args.number_of_clients),
                                                          num_of_clients_chosen_for_train,
                                                          replace=False)
        list_index_of_clients_for_test.sort()
        self.test(list_index_of_clients_for_test)

    def validate(self, list_index_of_client_for_validate):
        list_mAP_of_each_client_in_validation = []
        for client_id in list_index_of_client_for_validate:
            mAP_of_this_client = self.list_clients[client_id].validate()
            list_mAP_of_each_client_in_validation.append(mAP_of_this_client)
        self.list_average_validation_loss.append(sum(list_mAP_of_each_client_in_validation) /
                                                 len(list_mAP_of_each_client_in_validation))

    def test(self, list_index_of_client_for_test):
        list_mAP_of_each_client_in_test = []
        for client_id in list_index_of_client_for_test:
            mAP_of_this_client = self.list_clients[client_id].test()
            list_mAP_of_each_client_in_test.append(mAP_of_this_client)
        self.list_average_test_loss.append(sum(list_mAP_of_each_client_in_test) /
                                           len(list_mAP_of_each_client_in_test))


