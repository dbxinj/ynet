import numpy as np
import os
import logging
import time
import datetime
from argparse import ArgumentParser
from tqdm import trange

import model
from data import DARNDataIter, FashionDataIter
from singa import optimizer
from singa import device


def update_perf(his, cur, a=0.8):
    '''Accumulate the performance by considering history and current values.'''
    return his * a + cur * (1 - a)


def train(cfg, net, meanstd, train_data, val_data=None):
    log_dir = os.path.join('log', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)

    sgd = optimizer.SGD(momentum=cfg.mom, weight_decay=cfg.weight_decay)

    best_loss = 1000
    for epoch in range(cfg.max_epoch):
        bar = trange(train_data.num_batches, desc='Epoch %d' % epoch)
        loss = 0
        train_data.do_shuffle()
        train_data.start(train_data.load_triples)
        for b in bar:
            qimg, pimg, nimg, ptag, ntag = train_data.next()
            qimg -= meanstd[0][np.newaxis, :, np.newaxis, np.newaxis]
            qimg /= meanstd[1][np.newaxis, :, np.newaxis, np.newaxis]
            pimg -= meanstd[2][np.newaxis, :, np.newaxis, np.newaxis]
            pimg /= meanstd[3][np.newaxis, :, np.newaxis, np.newaxis]
            nimg -= meanstd[2][np.newaxis, :, np.newaxis, np.newaxis]
            nimg /= meanstd[3][np.newaxis, :, np.newaxis, np.newaxis]
            grads, l = net.bprop(qimg, pimg, nimg, ptag, ntag)
            for pname, pval, pgrad in zip(net.param_names(), net.param_values(), grads):
                sgd.apply_with_lr(epoch, cfg.lr, pgrad, pval, str(pname))
            loss = update_perf(loss, l.l1())
            bar.set_postfix(train_loss=loss)

        if val_data == None:
            continue

        bar = trange(val_data.num_batches, desc='Epoch %d' % epoch)
        loss = 0
        val_data.start(val_data.load_triples)
        for b in bar:
            qimg, pimg, nimg, ptag, ntag = val_data.next()
            l= net.evaluate(qimg, pimg, nimg, ptag, ntag)
            loss += l.l1()
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
    parser.add_argument("--param_dir", default='param')
    parser.add_argument("--data_dir", default='data')
    parser.add_argument("--image_dir", default='/home/wangyan/darn_dataset')
    parser.add_argument("--param_path", help='param pickle file path')
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # args = random_gen_args(args)
    args.param_dir = os.path.join(args.param_dir, args.dataset)
    # os.makedirs(args.param_dir)

    data_dir = os.path.join(args.data_dir, args.dataset)
    meanstd = np.load(os.path.join(data_dir, 'meta.npy'))
    train_pair = os.path.join(data_dir, 'train_pair.txt')
    train_shop = os.path.join(data_dir, 'train_shop.txt')
    val_pair = os.path.join(data_dir, 'validation_pair.txt')
    val_shop = os.path.join(data_dir, 'validation_shop.txt')

    if args.dataset == 'darn':
        train_data = DARNDataIter(args.image_dir, train_pair, train_shop)
        val_data = DARNDataIter(args.image_dir, val_pair, val_shop)
    elif args.dataset == 'fashion':
        train_data = FashionDataIter(args.image_dir, train_pair, train_shop)
        val_data = FashionDataIter(args.image_dir, val_pair, val_shop)
    else:
        print('Unknown dataset name')
    dev = device.create_cuda_gpu_on(2)
    net = model.CANIN('canet', model.TripletLoss(args.margin), dev, batchsize=args.batchsize, debug=args.debug)
    net.init_params(args.param_path)
    train(args, net, meanstd, train_data, val_data)
