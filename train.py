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


def train(cfg, net, train_data, val_data=None):
    log_dir = os.path.join('log', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)
    logging.info('%s' % cfg.str())

    if cfg.opt == 'adam':
        opt = optimizer.Adam(weight_decay=cfg.weight_decay)
    elif cfg.opt == 'nesterov':
        opt = optimizer.Nesterov(momentum=cfg.mom, weight_decay=cfg.weight_decay)
    else:
        opt = optimizer.SGD(momentum=cfg.mom, weight_decay=cfg.weight_decay)

    best_loss = 1000
    precision = []
    for epoch in range(cfg.max_epoch):
        net.train_on_epoch(epoch, train_data, opt, cfg.lr, cfg.nuser, cfg.nshop)
        net.evaluate_on_epoch(epoch, val_data)
        if epoch % cfg.search_freq == 0:
            perf, _ = net.retrieval(val_data, '%s-%d-result' % (cfg.param_dir, epoch), cfg.topk)
            precision.append(perf)
            logging.info('Retrieval performance of epoch %d = %s' % (epoch, perf))
            print perf

        if loss < best_loss - best_loss/10:
            if loss < best_loss - best_loss/5:
                net.save(os.path.join(cfg.param_dir, 'model-%d' % epoch))
            best_loss = loss
            nb_epoch_after_best = 0
        else:
            nb_epoch_after_best += 1
            if nb_epoch_after_best > 20:
                break
            elif nb_epoch_after_best % 10 == 0:
                cfg.lr /= 10
                print("Decay learning rate %f -> %f" % (cfg.lr * 10, cfg.lr))
                logging.info("Decay lr rate %f -> %f" % (cfg.lr * 10, cfg.lr))
    net.save(os.path.join(cfg.param_dir, 'model'))
    print precision


def create_datasets(args):
    data_dir = os.path.join(args.data_dir, args.dataset)
    meanstd = np.load(os.path.join(data_dir, 'meta.npy'))
    image_list_file = os.path.join(data_dir, 'image.txt')
    product_list_file = os.path.join(data_dir, 'product.txt')
    products = data.read_products(product_list_file)
    num_products = len(products)
    num_train_products = int(num_products * args.train_split)
    num_val_products = (num_products - num_train_products) // 2
    train_products = data.filter_products(args.img_dir, image_list_file,
            products[0:num_train_products], nuser=args.nuser, nshop=args.nshop)
    train_data = DataIter(args.image_dir, image_list_file, train_products, meanstd, img_size=args.img_size, batchsize=args.batchsize, nproc=args.nproc)
    val_data = DataIter(args.image_dir, image_list_file,
            products[num_train_products: num_train_products + num_val_products], meanstd, img_size=args.img_size, batchsize=args.batchsize, nproc=args.nproc)

    return train_data, val_data


if __name__ == '__main__':
    parser = ArgumentParser(description="Product search with attention modeling")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--opt", choices= ['sgd', 'adam', 'nesterov'], default='sgd')
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
    parser.add_argument("--nproc", type=int, default=1, help='num of data loading process')
    parser.add_argument("--gpu", type=int, default=0, help='gpu id')
    parser.add_argument("--img_size", type=int, default=224, help='image size')
    parser.add_argument("--nuser", type=int, default=1, help='min num of user images per product for filtering training products')
    parser.add_argument("--nshop", type=int, default=1, help='min num of user images per product for filtering training products')
    parser.add_argument("--train_split", type=float, default=0.8, help='ratio of products for training')
    parser.add_argument("--search_freq", type=int, default=1, help='frequency of validation on retrieval')
    args = parser.parse_args()

    train_data, val_data = create_datasets(args)
    dev = device.create_cuda_gpu_on(args.gpu)
    net = model.YNIN('YNIN', model.TripletLoss(args.margin), dev, img_size=args.img_size,
            batchsize=args.batchsize, args.nuser, args.nshop, debug=args.debug)
    net.init_params(args.param_path)
    args.param_dir = os.path.join(args.param_dir, args.dataset)
    os.makedirs(args.param_dir)
    train(args, net, train_data, val_data)
