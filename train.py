import numpy as np
import random
import os
import sys
import logging
import datetime
import cPickle as pickle
from argparse import ArgumentParser
from tqdm import trange

log_dir = os.path.join('log', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
os.makedirs(log_dir)
# logging.basicConfig(stream=sys.stdout, format='%(message)s', level=logging.INFO)
logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'), format='%(message)s', level=logging.INFO)

import model
import data
from singa import optimizer

def early_stop():
    pass
    '''
    best_loss = 1000
    nb_epoch_after_best = 0

    if loss < best_loss - best_loss/10:
        if epoch > 5 and loss < best_loss - best_loss/5:
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
    '''


def train(cfg, net, train_data, val_data, test_data=None):
    logging.info(cfg)
    print cfg
    if cfg.opt == 'adam':
        opt = optimizer.Adam(weight_decay=cfg.weight_decay)
    elif cfg.opt == 'nesterov':
        opt = optimizer.Nesterov(momentum=cfg.mom, weight_decay=cfg.weight_decay)
    else:
        opt = optimizer.SGD(momentum=cfg.mom, weight_decay=cfg.weight_decay)

    precision = []
    for epoch in range(cfg.max_epoch):
        train_loss = net.train_on_epoch(epoch, train_data, opt, cfg.lr)
        logging.info('Training at epoch %d: %s' % (epoch, np.array_str(train_loss)))
        if np.any(np.isnan(train_loss)) or np.any(np.isinf(train_loss)):
            return

        val_loss = net.evaluate_on_epoch(epoch, val_data)
        logging.info('Validation at epoch %d: %s' % (epoch, np.array_str(val_loss, 150)))
        print('Validation at epoch %d: %s' % (epoch, np.array_str(val_loss, 150)))
        if np.any(np.isnan(val_loss)) or np.any(np.isinf(val_loss)):
            return

        if epoch % cfg.search_freq == 0 and test_data is not None:
            perf, _ = net.retrieval(test_data, cfg.topk, args.candidate_path)
            precision.append(perf)
            logging.info('Test at epoch %d: %s' % (epoch, np.array_str(perf, 150)))
            print('Test at epoch %d: %s' % (epoch, np.array_str(perf, 150)))
            # net.save(os.path.join(cfg.param_dir, 'model-%d' % epoch))
    net.save(os.path.join(cfg.param_dir, 'model'))
    for prec in precision:
        print precision


def create_datasets(args, with_train, with_val, with_test=False):
    data_dir = os.path.join(args.data_dir, args.dataset)
    meanstd = np.load(os.path.join(data_dir, 'mean-std.npy'))
    img_list_file = os.path.join(data_dir, 'image.txt')
    product_list_file = os.path.join(data_dir, 'product.txt')
    products = data.read_products(product_list_file) # [0:5000]
    num_products = len(products)
    num_train_products = int(num_products * args.train_split)
    num_val_products = (num_products - num_train_products) // 2
    train_data, val_data, test_data = None, None, None
    if with_train:
        train_products = data.filter_products(args.img_dir, img_list_file, products[0:num_train_products], args.nuser, args.nshop)
        train_data = data.DataIter(args.img_dir, img_list_file, train_products,
                img_size=args.img_size, batchsize=args.batchsize, nproc=args.nproc, meanstd=meanstd)
    if with_val:
        val_products = products[num_train_products: num_train_products + num_val_products]
        val_products = data.filter_products(args.img_dir, img_list_file, val_products, args.nuser, args.nshop)
        val_data = data.DataIter(args.img_dir, img_list_file, val_products,
                img_size=args.img_size, batchsize=args.batchsize, nproc=args.nproc, meanstd=meanstd)
    if with_test:
        test_products = products[num_train_products + num_val_products:]
        test_data = data.DataIter(args.img_dir, img_list_file, test_products,
                img_size=args.img_size, batchsize=args.batchsize, nproc=args.nproc, meanstd=meanstd)

    return train_data, val_data, test_data


def gen_cfg(cfg):
    cfg.lr=random.choice([0.1, 0.01, 0.001, 0.0001])
    cfg.mom=random.choice([0.5, 0.8, 0.9])
    cfg.weight_decay=random.choice([5e-4, 1e-4, 5e-5, 1e-5])
    cfg.opt=random.choice(['sgd', 'adam', 'nesterov'])
    if cfg.opt == 'adam':
        cfg.mom = 0
    cfg.margin=random.choice([1.5, 1.2, 1.0, 0.8, 0.5, 0.2, 0.1])
    return cfg


if __name__ == '__main__':
    parser = ArgumentParser(description="Product search with attention modeling")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--opt", choices= ['sgd', 'adam', 'nesterov'], default='sgd')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--mom", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dataset", choices=['darn', 'deepfashion'], default='darn')
    parser.add_argument("--margin", type=float, default=0.5, help='margin for the triplet loss')
    parser.add_argument("--param_dir", default='param')
    parser.add_argument("--data_dir", default='data')
    parser.add_argument("--img_dir", default='../darn/')
    parser.add_argument("--param_path", help='param pickle file path')
    parser.add_argument("--candidate_path", help='results from initial search')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nproc", type=int, default=1, help='num of data loading process')
    parser.add_argument("--gpu", type=int, default=0, help='gpu id')
    parser.add_argument("--img_size", type=int, default=224, help='img size')
    parser.add_argument("--nuser", type=int, default=1, help='min num of user imgs per product for filtering training products')
    parser.add_argument("--nshop", type=int, default=1, help='min num of shop imgs per product for filtering training products')
    parser.add_argument("--train_split", type=float, default=0.8, help='ratio of products for training')
    parser.add_argument("--search_freq", type=int, default=5, help='frequency of validation on retrieval')
    parser.add_argument("--topk", type=int, default=100, help='top results')
    parser.add_argument("--net", default='tagnin', choices=['tagnin', 'ctxnin', 'ynin'])
    parser.add_argument("--nshift", type=int, default=4)
    parser.add_argument("--ntrail", type=int, default=1)
    parser.add_argument("--freeze_shared", action="store_true")
    parser.add_argument("--freeze_user", action="store_true")
    parser.add_argument("--freeze_shop", action="store_true")
    args = parser.parse_args()

    train_data, val_data, test_data = create_datasets(args, True, True, True)
    net = model.create_net(args, test_data)
    for i in range(args.ntrail):
        if args.ntrail > 1:
            args = gen_cfg(args)
            logging.info('\n\n-----------------------%d trail----------------------------' % i)
            args.param_dir = os.path.join('param', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        else:
            args.param_dir = os.path.join('param', args.dataset)
        os.makedirs(args.param_dir)
        net.init_params(args.param_path)
        train(args, net, train_data, val_data, test_data)
