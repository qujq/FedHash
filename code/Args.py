import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--global_epochs', type=int, default=2,
                        help="number of rounds of training")
    parser.add_argument('--model', type=str, default='cnn',
                        help="global model")
    parser.add_argument('--number_of_clients', type=int, default=5,
                        help="number of clients: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_epochs', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_batch_size', type=int, default=256,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--agg_method', type=str, default='fedavg', help='aggregation method')



    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name \
                        of dataset")

    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')

    # hashing
    parser.add_argument('--num_binary', type=int, default=12, help="Length of binary hash code")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='The alpha in hashing loss function.')

    args = parser.parse_args()
    return args
