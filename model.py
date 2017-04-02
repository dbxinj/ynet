from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D, Flatten, Slice
from singa import initializer
from singa import layer
from singa import loss


class L2Norm(layer.Layer):
    def __init__(self, input_sample_shape=None):
        pass


class TripletLoss(loss.Loss):
    def __init__(self, margin=0.8):
        pass


class CANet():
    '''Context-depedent attention modeling'''
    def __init__(self, name):
        self.name = name

    def create_net(self, batchsize=32):
        pass

    def init_params(net, weight_path=None):
        if weight_path is None:
            for pname, pval in zip(net.param_names(), net.param_values()):
                print pname, pval.shape
                if 'conv' in pname and len(pval.shape) > 1:
                    initializer.gaussian(pval, 0, pval.shape[1])
                else:
                    pval.set_value(0)
        else:
            net.load(weight_path, use_pickle=True)

    def extract_query_feature(self, x):
        pass

    def extract_db_feature(self, x, tag):
        pass

    def bprop(self, qimg, pimg, nimg, ptag, ntag):
        pass

    def evaluate(self, qimg, pimg, nimg, ptag, ntag):
        pass

    def save(self, fpath):
        self.street_net.save(fpath, use_pickle=True)
        self.shop_net.save(fpath, use_pickle=True)

    def load(self, fpath):
        self.street_net.save(fpath, use_pickle=True)
        self.shop_net.save(fpath, use_pickle=True)


class CANIN(CANet):
    def add_conv(layers, name, nb_filter, size, stride, pad=0, sample_size=None):
        if sample_size != None:
            layers.append(Conv2D('%s-3x3' % name, nb_filter, size, stride, pad=pad,
                           input_sample_shape=sample_size))
        else:
            layers.append(Conv2D('%s-3x3' % name, nb_filter, size, stride, pad=pad,
                           input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-3x3-relu' % name, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Conv2D('%s-1x1-1' % name, nb_filter, 1, 1, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-1x1-1-relu', input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Conv2D('%s-1x1-2', nb_filter, 1, 1, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-1x1-2-relu' % name, input_sample_shape=layers[-1].get_output_sample_shape()))

    def create_net(self, batchsize=32):
        shared = []

        self.add_conv(shared, 'c1', 96, 7, 2, sample_size=(3, 277, 277))
        shared.append(MaxPooling2D('p1', 3, 2), input_sample_shape=shared[-1].get_output_sample_shape())

        self.add_conv(shared, 'c2', 256, 5, 2, 2)
        shared.append(MaxPooling2D('p2', 3, 2), input_sample_shape=shared[-1].get_output_sample_shape())

        self.add_conv(shared, 'c3', 384, 3, 1, 1)
        shared.append(MaxPooling2D('p3', 3, 2), input_sample_shape=shared[-1].get_output_sample_shape())
        slice_layer = Slice('slice', 0, [batchsize], input_sample_shape=shared[-1].get_output_sample_shape())
        shared.append(slice_layer)

        street = []
        self.add_conv(street, 'steet-c4', 512, 3, 1, 1, input_sample_shape=slice_layer.get_output_sample_shape()[0])
        street.append(AvgPooling2D('street-p4', 3, 2, input_sample_shape=street[-1].get_output_sample_shape()[1]))
        street.append(Flatten('street-flat', input_sample_shape=street[-1].get_output_sample_shape()))
        street.append(L2Norm('street-l2', input_sample_shape=street[-1].get_output_sample_shape()))

        shop = []
        self.add_conv(street, 'shop-c4', 512, 3, 1, 1, input_sample_shape=slice_layer.get_output_sample_shape()[1])
        street.append(AvgPooling2D('shop-p4', 3, 2, input_sample_shape=shop[-1].get_output_sample_shape()))
        street.append(Flatten('shop-flat', input_sample_shape=shop[-1].get_output_sample_shape()))
        street.append(L2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))
        street.append(Slice('shop-slice', 0, [batchsize], input_sample_shape=shop[-1].get_output_sample_shape()))

        return shared, street, shop
