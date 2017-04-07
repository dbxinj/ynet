import caffe
import numpy as np
import cPickle as pickle
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description="extract param from caffe to pickel")
    parser.add_argument("--proto", help='proto file')
    parser.add_argument("--model", help='bin file')
    parser.add_argument("--output", help='pickle file path')
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--names", type=list, help="list of layer names")
    args = parser.parse_args()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    if args.show:
        print net.params.keys()
    else:
        params = {}
        names = ['conv1-3x3', 'conv1-1x1-1', 'conv1-1x1-2', 'conv2-3x3', 'conv2-1x1-1', 'conv2-1x1-2', 'conv3-3x3', 'conv3-1x1-1', 'conv3-1x1-2', 'street-conv4-3x3', 'street-conv4-1x1-1', 'street-conv4-1x1-2', 'shop-conv4-3x3', 'shop-conv4-1x1-1', 'shop-conv4-1x1-2']
        keys = net.params.keys()
        keys = keys + keys[-3:]
        print names
        print keys
        for out_layer, caffe_layer in zip(names, keys):
            weights=np.copy(net.params[caffe_layer][0].data)
            bias=np.copy(net.params[caffe_layer][1].data)
            params[out_layer+'_weight']=weights
            params[out_layer+'_bias']=bias
            print out_layer, weights.shape, bias.shape
        with open(args.output, 'wb') as fd:
            pickle.dump(params, fd)
