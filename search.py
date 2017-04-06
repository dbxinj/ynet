import model
from data import DARNDataIter, FashionDataIter

from singa import device
import os
import numpy as np
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description="Context-depedent attention modeling search")
    parser.add_argument("--param_path", default='param')
    parser.add_argument("--dataset", choices=['darn', 'deepfashion'], default='darn')
    parser.add_argument("--image_dir", default='/home/wangyan/darn_dataset')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0, help='gpu id')
    parser.add_argument("--img_size", type=int, default=227, help='image size')
    args = parser.parse_args()

    data_dir = os.path.join('data', args.dataset)
    meanstd = np.load(os.path.join(data_dir, 'meta.npy'))
    pair = os.path.join(data_dir, 'test_pair.txt')
    shop = os.path.join(data_dir, 'test_shop.txt')

    if args.dataset == 'darn':
        data = DARNDataIter(args.image_dir, pair, shop, img_size=args.img_size, nproc=1)
    elif args.dataset == 'deepfashion':
        data = FashionDataIter(args.image_dir, pair, shop, img_size=args.img_size, nproc=1)
    else:
        print('Unknown dataset name')

    dev = device.create_cuda_gpu_on(args.gpu)
    net = model.CANIN('canet', None, dev, img_size=args.img_size, batchsize=args.batchsize, debug=args.debug)
    net.init_params(args.param_path)

    result_path = os.path.join('result', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

    net.retrieval(data, result_path, topk=100)
