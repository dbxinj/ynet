from ynet import YNIN, YVGG, TripletLoss
from tagnet import TagNIN, TagVGG
from ctxnet import CtxNIN, QuadLoss, CtxVGG
from singa import device

import cPickle as pickle
import numpy as np
import logging
import os
import sys

logger = logging.getLogger(__name__)


def create_net(args, test_data=None):
    dev = device.create_cuda_gpu_on(args.gpu)
    if args.net == 'ynin':
        net = YNIN('YNIN', TripletLoss(args.margin, args.nshift), dev,
                args.img_size, args.batchsize,  nshift=args.nshift, debug=args.debug)
    elif args.net == 'yvgg':
        net = YVGG('YVGG', TripletLoss(args.margin, args.nshift), dev,
                args.img_size, args.batchsize,  nshift=args.nshift, debug=args.debug)
    elif args.net == 'tagnin':
        assert args.ncat > 0 or args.nattr > 0, 'Either num category or num tags should be set'
        net = TagNIN('TagNIN', TripletLoss(args.margin, args.nshift), dev, args.img_size, args.batchsize, args.ncat+args.nattr,
                args.freeze_shared, args.freeze_shop, args.freeze_user, nshift=args.nshift, debug=args.debug)
    elif args.net == 'tagvgg':
        assert args.ncat > 0 or args.nattr > 0, 'Either num category or num tags should be set'
        net = TagVGG('TagVGG', TripletLoss(args.margin, args.nshift), dev, args.img_size, args.batchsize, args.ncat+args.nattr,
                args.freeze_shared, args.freeze_shop, args.freeze_user, nshift=args.nshift, debug=args.debug)
    elif args.net == 'ctxnin':
        assert args.candidate_path is not None, 'must provide the candiate path'
        if not os.path.exists(args.candidate_path):
            net = TagNIN('TagNIN', None, dev, args.img_size, args.batchsize, ntag = args.ncat+args.nattr, debug=args.debug)
            net.init_params(args.param_path)
            perf, result = net.retrieval(test_data, topk=256)
            logging.info('Init retrieval: %s' % (np.array_str(perf, 150)))
            print('Init retrieval %s' % (np.array_str(perf, 150)))
            with open(args.candidate_path, 'w') as fd:
                pickle.dump(result, fd)
        net = CtxNIN('CtxNIN', QuadLoss(args.margin, args.nshift), dev, args.img_size, args.batchsize, args.ncat+args.nattr,
                args.freeze_shared, args.freeze_shop, args.freeze_user, nshift=args.nshift,debug=args.debug)
    elif args.net == 'ctxvgg':
        assert args.candidate_path is not None, 'must provide the candiate path'
        if not os.path.exists(args.candidate_path):
            net = TagVGG('TagVGG', None, dev, args.img_size, args.batchsize, ntag = args.ncat+args.nattr, debug=args.debug)
            net.init_params(args.param_path)
            perf, result = net.retrieval(test_data, topk=256)
            logging.info('Init retrieval: %s' % (np.array_str(perf, 150)))
            print('Init retrieval %s' % (np.array_str(perf, 150)))
            with open(args.candidate_path, 'w') as fd:
                pickle.dump(result, fd)
        net = CtxVGG('CtxVGG', QuadLoss(args.margin, args.nshift), dev, args.img_size, args.batchsize, args.ncat+args.nattr,
                args.freeze_shared, args.freeze_shop, args.freeze_user, nshift=args.nshift,debug=args.debug)
    else:
        print('Unknown net type %s' % args.net)
        sys.exit(1)
    return net
