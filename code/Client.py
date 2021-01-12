import torch
from utils import hashing_loss, calculate_mAP
class Client(object):
    def __init__(self, client_id, model, args, dataloader_train, dataloader_validate, dataloader_test, device,
                 loss_function):
        self.id = client_id
        self.model = model
        self.args = args
        self.train_dataloader = dataloader_train
        self.validate_dataloader = dataloader_validate
        self.test_dataloader = dataloader_test
        self.device = device
        self.loss_function = loss_function
        self.list_loss_every_global_epoch = []

    def train(self, global_epoch_id):
        if self.args.optimizer == 'adam':
            print(self.model, '!!')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.004)

        self.model.train()
        self.model.to(self.device)

        list_local_epoch_loss = []
        for local_epoch in range(self.args.local_epochs):
            list_batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                output = self.model(images)
                if self.args.agg_method == 'fedavg':
                    if self.loss_function == 'hashing loss':
                        loss = hashing_loss(output, labels, 2 * self.args.num_binary, self.args.alpha)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(
                        '|User id : {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            self.id, global_epoch_id, local_epoch, batch_idx * len(images),
                            len(self.train_dataloader.dataset), 100. * batch_idx / len(self.train_dataloader), loss.item())
                            )
                list_batch_loss.append(loss)
            list_local_epoch_loss.append(sum(list_batch_loss)/len(list_batch_loss))
        self.list_loss_every_global_epoch.append(sum(list_local_epoch_loss) / len(list_local_epoch_loss))
        return self.model.state_dict()

    def validate(self):
        self.model.to(self.device)
        self.model.eval()
        list_validate_batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.validate_dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            with torch.no_grad():
                if self.args.agg_method == 'fedavg':
                    if self.loss_function == 'hashing loss':
                        loss = hashing_loss(output, labels, 2 * self.args.num_binary, self.args.alpha)
                list_validate_batch_loss.append(loss)
        mAP = calculate_mAP(self.model, self.train_dataloader, self.validate_dataloader)
        print('Validation loss: ', sum(list_validate_batch_loss)/len(list_validate_batch_loss))
        return mAP

    def test(self):
        self.model.to(self.device)
        self.model.eval()
        list_test_batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.test_dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            with torch.no_grad():
                if self.args.agg_method == 'fedavg':
                    if self.loss_function == 'hashing loss':
                        loss = hashing_loss(output, labels, 2 * self.args.num_binary, self.args.alpha)
                list_test_batch_loss.append(loss)
        mAP = calculate_mAP(self.model, self.train_dataloader, self.test_dataloader)
        print('Test loss: ', sum(list_test_batch_loss)/len(list_test_batch_loss))
        return mAP

