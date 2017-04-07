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
        for out_layer, caffe_layer in zip(args.names, net.params.keys()):
            weights=np.copy(net.params[caffe_layer][0].data)
            bias=np.copy(net.params[lcaffe_layer][1].data)
            params[out_layer+'_weight']=weights
            params[out_layer+'_bias']=bias
            print out_layer, weights.shape, bias.shape
        with open(args.output, 'wb') as fd:
            pickle.dump(params, fd)
