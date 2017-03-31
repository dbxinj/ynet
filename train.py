import numpy
import os
import logging
import time
import datetime
from argparse import ArgumentParser
from tqdm import tnrange

import model
from data import DARNDataIter, FashionDataIter
from singa import optimizer



def update_perf(his, cur, a=0.8):
    '''Accumulate the performance by considering history and current values.'''
    return his * a + cur * (1 - a)


def train(cfg, net, train_data, val_data=None):
    log_dir = os.path.join('log', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)


    sgd = optimizer.SGD(momentum=cfg.mom, weight_decay=cfg.weight_decay)

    best_loss = 1000
    for epoch in range(cfg.max_epoch):
        bar = tnrange(train_data.num_batches, desc='Epoch %d' % epoch)
        loss = 0
        train_data.shuffle()
        for b in bar:
            qimg, pimg, nimg, ptag, ntag = train_data.next()
            grads, l = net.bprop(qimg, pimg, nimg, ptag, ntag)
            for pname, pval, pgrad in zip(net.param_names(), net.param_values(), grads):
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
        print('Epoch %d, validation loss = %f' % (epoch, loss / val_data.num_batches))

        if loss < best_loss - cfg.gama:
            best_loss = loss
            nb_epoch_after_best = 0
            if best_loss < cfg.margin:
                net.save(os.path.join(cfg.param_dir, 'model-%d' % epoch), 50, use_pickle=True)
        else:
            nb_epoch_after_best += 1
            if nb_epoch_after_best > 4:
                break
            elif nb_epoch_after_best % 2 == 0:
                cfg.lr /= 10
                print("Decay learning rate %f -> %f" % (cfg.lr * 10, cfg.lr))
                logging.info("Decay lr rate %f -> %f" % (cfg.lr * 10, cfg.lr))
        net.save(os.path.join(cfg.param_dir, 'model'), 50, use_pickle=True)


if __name__ == '__main__':
    parser = ArgumentParser(description="Context-depedent attention modeling")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dataset", choices=['darn', 'fashion'], default='darn')
    parser.add_argument("--gama", type=float, default=0.01, help='delta theta threshold')
    parser.add_argument("--margin", type=float, default=0.1, help='margin for the triplet loss')
    parser.add_argument("--param_dir", default='../param')
    parser.add_argument("--data_dir", default='../data')
    args = parser.parse_args()
    print type(args)
    print(args)


    # args = random_gen_args(args)
    args.param_dir = os.path.join(args.param_dir, args.dataset)
    os.makedirs(args.param_dir)

    data_dir = os.path.join(args.data_dir, args.dataset)
    train_pair = os.path.join(data_dir, 'train_pair.txt')
    train_shop = os.path.join(data_dir, 'train_shop.txt')
    val_pair = os.path.join(data_dir, 'val_pair.txt')
    val_shop = os.path.join(data_dir, 'val_shop.txt')

    if args.dataset == 'darn':
        train_data = DARNDataIter(train_pair, train_shop)
        val_data = DARNDataIter(val_pair, val_shop)
    elif args.dataset == 'fashion':
        train_data = FashionDataIter(train_pair, train_shop)
        val_data = FashionDataIter(val_pair, val_shop)
    else:
        print('Unknown dataset name')
    net = model.CANet('canet')
    train(args, net, train_data, val_data)
