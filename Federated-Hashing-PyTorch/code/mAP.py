import time
from functools import wraps
import numpy as np
import torch
from torch.autograd import Variable

def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        print(f'total time = {time.time() - time_start:.4f}')
        return ret

    return wrapper


def compute_result(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls in dataloader:
        clses.append(cls)
        with torch.no_grad():
            #bs.append(net(Variable(img.cuda(), volatile=True)).data.cpu())
            bs.append(net(Variable(img.cuda())).data)
    return torch.sign(torch.cat(bs)), torch.cat(clses)


@timing
def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    """
    for x in trn_binary:
        x = x.long()
    for x in tst_binary:
        x = x.long()
    for x in trn_label:
        x = x.long()
    for x in tst_label:
        x = x.long()
    """

    trn_binary, tst_binary, trn_label, tst_label = \
        trn_binary.long().cuda(), tst_binary.long().cuda(), trn_label.cuda(), tst_label.cuda()
    
    """
    print(trn_binary)  #tensor([9000,36])
    print(tst_binary)  #tensor([1000,36])
    print(trn_label)   #tensor([9000])
    print(tst_label)   #tensor([1000])
    """

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    Ns = Ns.cuda()
    #mAP = torch.tensor(0)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        query_label = query_label.cuda()
        query_binary = query_binary.cuda()
        _, query_result = torch.sum((query_binary != trn_binary), dim=1).sort()#第一个返回值是value，query_result 是查询到的图片的索引
        correct = (query_label == trn_label[query_result])
        correct = correct.float()
        tmp2 = torch.cumsum(correct, dim=0)#从第一列开始后面的每一列都是前面对应行元素的累加和
        Ns = Ns.float()
        P = tmp2 / Ns #平均准确率
        tmp = P * correct
        if torch.sum(correct) == 0:
            AP.append(0)
        else:
            AP.append((torch.sum(tmp) / torch.sum(correct)).cpu().numpy())
    mAP = np.mean(AP)

    return mAP

