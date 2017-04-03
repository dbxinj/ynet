from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D, Flatten, Slice
from singa import initializer
from singa import layer
from singa import loss
from singa import tensor

import numpy as np


class L2Norm(layer.Layer):
    def __init__(self, input_sample_shape=None):
        self.y = None
        self.norm = None

    def forward(self, is_train, x):
        squared_norm = tensor.sum_columns(x * x)
        norm = tensor.sqrt(squared_norm)
        if is_train:
            self.y = x.div_column(norm)
            self.norm = norm
            return self.y
        else:
            return x.div_column(norm)

    def backward(self, dy):
        # (b - b * k) /norm, k = sum(dy * y)
        k = tensor.sum_columns(dy * self.y)
        dx = dy
        dx -= self.y.mult_columns(-k)
        dx /= self.norm
        return dx


class TripletLoss(loss.Loss):
    def __init__(self, margin=0.8):
        self.margin = margin
        self.ga = None
        self.gp = None
        self.gn = None

    def forward(self, is_train, a, p, n):
        d_ap = a - p
        d1 = tensor.sum_columns(d_ap * d_ap)
        d_an = a - n
        d2 = tensor.sum_columns(d_an * d_an)
        d = d1 - d2
        d += self.margin
        sign = d > 0
        loss = d.sum() * sign
        batchsize = float(a.shape[0])
        if is_train:
            self.ga = (n - p) * (2 / batchsize)
            self.ga.mult_column(sign)
            self.gp = d_ap * (-2 / batchsize)
            self.gp.mult_column(sign)
            self.gn = d_an * (2 / batchsize)
            self.gn.mult_column(sign)
        return loss

    def backward(self):
        return self.ga, self.gp, self.gn


class CANet():
    '''Context-depedent attention modeling'''
    def __init__(self, name, loss, dev, batchsize=32):
        self.name = name
        self.loss = loss
        self.device = dev
        self.batchsize = batchsize
        self.layers = []
        self.create_net(batchsize)
        self.to_device(dev)

    def create_net(self, batchsize=32):
        pass

    def to_device(self, dev):
        for lyr in self.layers:
            lyr.to_device(dev)

    def param_names(self):
        pname = []
        for lyr in self.layers:
            pname.extends(lyr.param_names())
        return pname

    def param_values(self):
        pvals = []
        for lyr in self.layers:
            pvals.extends(lyr.param_values())
        return pvals

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
    def __init__(self, name, loss, dev, batchsize=32):
        super(CANet, self).__init__(name, loss, dev, batchsize)
        self.shared, self.street, self.shop = self.create_net(name, batchsize)
        self.layers.extends(self.shared)
        self.layers.extends(self.street)
        self.layers.extends(self.shop)

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
        #street.append(L2Norm('street-l2', input_sample_shape=street[-1].get_output_sample_shape()))

        shop = []
        self.add_conv(street, 'shop-c4', 512, 3, 1, 1, input_sample_shape=slice_layer.get_output_sample_shape()[1])
        street.append(AvgPooling2D('shop-p4', 3, 2, input_sample_shape=shop[-1].get_output_sample_shape()))
        street.append(Flatten('shop-flat', input_sample_shape=shop[-1].get_output_sample_shape()))
        #street.append(L2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))
        street.append(Slice('shop-slice', 0, [batchsize], input_sample_shape=shop[-1].get_output_sample_shape()))

        return shared, street, shop

    def forward(self, is_train, qimg, pimg, nimg, ptag, ntag):
        x = np.concatenate((qimg, pimg, nimg), axis=0)
        x = tensor.from_numpy(x)
        for lyr in self.shared:
            x = lyr.forward(is_train, x)
        a, b = x
        for lyr in self.street:
            a = lyr.forward(is_train, a)
        for lyr in self.shop:
            b = lyr.forward(is_train, b)
        p, n = b
        return a, p, n

    def backward(self, da, dp, dn):
        param_grads = []
        ds, _ = self.shop[-1].backward(True, [dp, dn])
        dshop = ds[0]
        for lyr in self.shop[0:-2].reverse():
            dshop, dp = lyr.backward(True, dshop)
            if dp is not None:
                param_grads.extends(reversed(dp))

        for lyr in self.street.reverse():
            da, dp = lyr.backward(True, da)
            if dp is not None:
                param_grads.extends(reversed(dp))

        ds, _ = self.shared[-1].backward(True, [da, dshop])
        d = ds[0]
        for lyr in self.shared.reverse():
            d, dp = lyr.backward(True, d)
            if dp is not None:
                param_grads.extends(reversed(dp))
        return param_grads

    def bprop(self, qimg, pimg, nimg, ptag, ntag):
        assert self.batchsize == qimg.shape[0], 'batchsize not correct %d vs %d' % (self.batchsize, qimg.shape[0])
        a, p, n = self.forward(True, qimg, pimg, nimg, ptag, ntag)
        loss = self.loss.forward(True, a, p, n)
        da, dp, dn = self.loss.backward()
        return loss.sum() / qimg.shape[0], self.backward(self, da, dp, dn)

    def evalute(self, qimg, pimg, nimg, ptag, ntag):
        assert self.batchsize == qimg.shape[0], 'batchsize not correct %d vs %d' % (self.batchsize, qimg.shape[0])
        a, p, n = self.forward(False, qimg, pimg, nimg, ptag, ntag)
        loss = self.loss.forward(False, a, p, n)
        return loss.sum() / qimg.shape[0]
