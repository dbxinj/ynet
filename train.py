import numpy
import os
from argparse import ArgumentParser
from tqdm import tnrange

import model
from singa import optimizer

class Config():
    def __init__(self, data_dir):
        self.batchsize = 32
        self.max_epoch = 100
        self.lr = 0.001
        self.weight_decay = 1e-4


def update_perf(his, cur, a=0.8):
    '''Accumulate the performance by considering history and current values.'''
    return his * a + cur * (1 - a)


def train(cfg, net, train_data, val_data=None):
    sgd = opt.SGD(momentum=cfg.mom, weight_decay=cfg.weight_decay)

    best_loss = 1000
    for epoch in range(max_epoch):
        bar = tnrange(dat.num_batches, desc='Epoch %d' % epoch)
        loss = 0
        train_data.shuffle()
        for b in bar:
            qimg, pimg, nimg, ptag, ntag = train_data.next()
            grads, l = net.bprop(qimg, pimg, nimg, ptag, ntag)
            for pname, pval, pgrad in (net.param_names(), net.param_values(), grads):
                sgd.apply_with_lr(epoch, cfg.lr, pgrad, pval, str(pname))
            loss = update_perf(loss, l)
            bar.set_postfix(train_loss=loss)

        if val_data == None:
            continue

        bar = tnrange(val_data.num_batches, desc='Epoch %d' % epoch)
        loss = 0
        for b in bar:
            qimg, pimg, nimg, ptag, ntag = val_data.next()
            loss += net.evaluate_one_batch(qimg, pimg, nimg, ptag, ntag)
        print('Epoch %d, validation loss = %f' % (epoch, loss / val_data.num_batches)

        if loss < best_loss - cfg.gama:
            best_loss = loss
            nb_epoch_after_best = 0
            if best_loss < cfg.margin:
                net.save(os.path.join(cfg.param, 'model-%d' % epoch), 50, use_pickle=True)
        else:
            nb_epoch_after_best += 1
            if nb_epoch_after_best > 4:
                break
            elif nb_epoch_after_best % 2 == 0:
                cfg.lr /= 10
                print("Decay learning rate %f -> %f" % (cfg.lr * 10, cfg.lr))
                logging.info("Decay lr rate %f -> %f" % (cfg.lr * 10, cfg.lr))


if __name__ == '__main__':
    parser = ArgumentParser(description="Context-depedent attention modeling")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dataset", choices =['darn', 'fashion'], default='darn')
    parser.add_argument("--gama", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.1)
    args = parser.parse_args()
    print type(args)
    print(args)
    return

    # args = random_gen_args(args)
    train_pair = os.path.join(args.data_dir, 'train_pair.txt')
    train_shop = os.path.join(args.data_dir, 'train_shop.txt'))
    val_pair = os.path.join(args.data_dir, 'val_pair.txt')
    val_shop = os.path.join(args.data_dir, 'val_shop.txt')

    if args.dataset == 'darn':
        train_data = DARNDataIter(train_pair, train_shop)
        val_data = DARNDataIter(val_pair, val_shop)
    elif: args.dataset == 'fashion':
        train_data = FashionDataIter(train_pair, train_shop)
        val_data = FashionDataIter(val_pair, val_shop)
    else:
        print('Unknown dataset name')

    #net = model.CANet()
    #train(args, net, train_data, val_data)
