import torch
from Server import Server
from Args import args_parser
from model import CNN
from utils import get_dataset
import copy
if __name__ == "__main__":
    args = args_parser()

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    if args.model == 'cnn':
        global_model = CNN(args)
    global_model.to(device)

    train_dataset, test_dataset = get_dataset(args)

    server = Server(train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    model=copy.deepcopy(global_model),
                    args=args,
                    device=device)

    server.setup_data()
    server.setup_clients()
    server.train()
