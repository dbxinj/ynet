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

    # sgd = optimizer.SGD(momentum=cfg.mom, weight_decay=cfg.weight_decay)
    sgd = optimizer.Adam(weight_decay=cfg.weight_decay)

    best_loss = 1000
    for epoch in range(cfg.max_epoch):
        bar = trange(train_data.num_batches, desc='Epoch %d' % epoch)
        loss, dap, dan = 0, 0, 0
        train_data.do_shuffle()
        train_data.start(train_data.load_triples)
        for b in bar:
            t1 = time.time()
            qimg, pimg, nimg, ptag, ntag = train_data.next()
            t2 = time.time()
            qimg -= meanstd[0][np.newaxis, :, np.newaxis, np.newaxis]
            qimg /= meanstd[1][np.newaxis, :, np.newaxis, np.newaxis]
            pimg -= meanstd[2][np.newaxis, :, np.newaxis, np.newaxis]
            pimg /= meanstd[3][np.newaxis, :, np.newaxis, np.newaxis]
            nimg -= meanstd[2][np.newaxis, :, np.newaxis, np.newaxis]
            nimg /= meanstd[3][np.newaxis, :, np.newaxis, np.newaxis]
            grads, l = net.bprop(qimg, pimg, nimg, ptag, ntag)
            if cfg.debug:
                print('-------------prams---------------')
            for pname, pval, pgrad in zip(net.param_names(), net.param_values(), grads):
                if cfg.debug:
                    print('%30s = %f, %f' % (pname, pval.l1(), pgrad.l1()))
#                print pname, pgrad.shape, pval.shape
                sgd.apply_with_lr(epoch, cfg.lr, pgrad, pval, str(pname), b)
            loss = update_perf(loss, l[0])
            dap = update_perf(dap, l[1])
            dan = update_perf(dan, l[2])
            t3 = time.time()
            bar.set_postfix(train_loss=loss, dap=dap, dan=dan, bptime=t3-t2, load_time=t2-t1)

        if val_data == None:
            continue

        bar = trange(val_data.num_batches, desc='Epoch %d' % epoch)
        loss, dap, dan = 0, 0, 0
        val_data.start(val_data.load_triples)
        for b in bar:
            qimg, pimg, nimg, ptag, ntag = val_data.next()
            qimg -= meanstd[0][np.newaxis, :, np.newaxis, np.newaxis]
            qimg /= meanstd[1][np.newaxis, :, np.newaxis, np.newaxis]
            pimg -= meanstd[2][np.newaxis, :, np.newaxis, np.newaxis]
            pimg /= meanstd[3][np.newaxis, :, np.newaxis, np.newaxis]
            nimg -= meanstd[2][np.newaxis, :, np.newaxis, np.newaxis]
            nimg /= meanstd[3][np.newaxis, :, np.newaxis, np.newaxis]
            l, ap, an= net.evaluate(qimg, pimg, nimg, ptag, ntag)
            loss += l
            dap += ap
            dan += an
        print('Epoch %d, validation loss = %f, pos dist = %f, neg dist = %f' %\
              (epoch, loss / val_data.num_batches, dap / val_data.num_batches,
               dan / val_data.num_batches))
        net.retrieval(val_data, os.path.join(cfg.param_dir, 'result-%d-' % epoch), meanstd, 100)

        if loss < best_loss - best_loss/10:
            if loss < best_loss - best_loss/5:
                net.save(os.path.join(cfg.param_dir, 'model-%d' % epoch))
            best_loss = loss
            nb_epoch_after_best = 0
        else:
            nb_epoch_after_best += 1
            if nb_epoch_after_best > 10:
                break
            elif nb_epoch_after_best % 5 == 0:
                cfg.lr /= 10
                print("Decay learning rate %f -> %f" % (cfg.lr * 10, cfg.lr))
                logging.info("Decay lr rate %f -> %f" % (cfg.lr * 10, cfg.lr))
        net.save(os.path.join(cfg.param_dir, 'model'))


if __name__ == '__main__':
    parser = ArgumentParser(description="Context-depedent attention modeling")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dataset", choices=['darn', 'deepfashion'], default='darn')
    parser.add_argument("--margin", type=float, default=0.2, help='margin for the triplet loss')
    parser.add_argument("--param_dir", default='param')
    parser.add_argument("--data_dir", default='data')
    parser.add_argument("--image_dir", default='/home/wangyan/darn_dataset')
    parser.add_argument("--param_path", help='param pickle file path')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nproc", type=int, default=2, help='num of data loading process')
    parser.add_argument("--gpu", type=int, default=0, help='gpu id')
    parser.add_argument("--img_size", type=int, default=224, help='image size')
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, args.dataset)
    meanstd = np.load(os.path.join(data_dir, 'meta.npy'))
    train_pair = os.path.join(data_dir, 'train_pair.txt')
    val_pair = os.path.join(data_dir, 'validation_pair.txt')
    if args.dataset == 'darn':
        train_shop = os.path.join(data_dir, 'train_shop.txt')
        val_shop = os.path.join(data_dir, 'validation_shop.txt')
        train_data = DARNDataIter(args.image_dir, train_pair, train_shop, img_size=args.img_size, nproc=args.nproc)
        val_data = DARNDataIter(args.image_dir, val_pair, val_shop, img_size=args.img_size, nproc=args.nproc)
    elif args.dataset == 'deepfashion':
        shop = os.path.join(data_dir, 'shop.txt')
        train_data = FashionDataIter(args.image_dir, train_pair, shop, img_size=args.img_size, nproc=args.nproc)
        val_data = FashionDataIter(args.image_dir, val_pair, shop, img_size=args.img_size, nproc=args.nproc)
    else:
        print('Unknown dataset name')
    dev = device.create_cuda_gpu_on(args.gpu)
    net = model.CANIN('canet', model.TripletLoss(args.margin), dev, img_size=args.img_size, batchsize=args.batchsize, debug=args.debug)
    net.init_params(args.param_path)
    # args = random_gen_args(args)
    args.param_dir = os.path.join(args.param_dir, args.dataset)
    os.makedirs(args.param_dir)
    train(args, net, meanstd, train_data, val_data)
