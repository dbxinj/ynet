from ynet import YNIN, TripletLoss
from tagnet import TagNIN
from ctxnet import CtxNIN
from singa import device

import cPickle as pickle
import logging
import os
import sys

logger = logging.getLogger(__name__)


def create_net(args, test_data=None):
    dev = device.create_cuda_gpu_on(args.gpu)
    if args.net == 'ynin':
        net = YNIN('YNIN', TripletLoss(args.margin, args.nshift), dev, args.img_size, args.batchsize, debug=args.debug)
    elif args.net == 'tagnin':
        net = TagNIN('TagNIN', TripletLoss(args.margin, args.nshift), dev, args.img_size, args.batchsize,
                args.debug, args.freeze_shared, args.freeze_shop, args.freeze_user)
    elif args.net == 'ctxnin':
        assert args.candidate_path is not None, 'must provide the candiate path'
        if not os.path.exists(args.candidate_path):
            net = TagNIN('TagNIN', None, dev, args.img_size, args.batchsize, debug=args.debug)
            perf, result = net.retrieval(test_data, topk=256)
            logging.info('Init retrieval: %s' % (np.array_str(perf, 150)))
            print('Init retrieval %s' % (np.array_str(perf, 150)))
            with open(args.candiate_path, 'w') as fd:
                pickle.dump(result, fd)
        net = ContextNIN('CtxNIN', QuadLoss(args.margin, args.nshift), dev, args.img_size, args.batchsize, args.nshift,
                args.freeze_shared, args.freeze_shop, args.freeze_user, debug=args.debug)
    else:
        print('Unknown net type %s' % args.net)
        sys.exit(1)
    return net
