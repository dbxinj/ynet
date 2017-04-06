from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D, Flatten, Slice
from singa import initializer
from singa import layer
from singa import loss
from singa import tensor
import cPickle as pickle

import numpy as np
import scipy.spatial
from tqdm import trange
import time


def compute_precision(target):
    points = range(9, 101, 10)
    ret = target.cumsum(axis=1) > 0
    prec = np.average(ret[:, points], axis=0)
    return prec


class L2Norm(layer.Layer):
    def __init__(self, name, input_sample_shape, epsilon=1e-8):
        super(L2Norm, self).__init__(name)
        self.y = None
        self.norm = None
        self.name = name
        self.epsilon = epsilon
        self.out_sample_shape = input_sample_shape

    def get_output_sample_shape(self):
        return self.out_sample_shape

    def forward(self, is_train, x):
        norm = tensor.sum_columns(tensor.square(x))
        norm += self.epsilon
        norm = tensor.sqrt(norm)
        self.y = x.clone()
        self.y.div_column(norm)

        if is_train:
            self.norm = norm
        return self.y

    def backward(self, is_train, dy):
        # (b' - b * k) /norm, k = sum(dy * y)
        k = tensor.sum_columns(tensor.eltwise_mult(dy, self.y))
        self.y.mult_column(k)
        dx = dy - self.y
        dx.div_column(self.norm)
        return dx, []


class TripletLoss(loss.Loss):
    def __init__(self, margin=0.1):
        self.margin = margin
        self.ga = None
        self.gp = None
        self.gn = None

    def forward(self, is_train, a, p, n):
        d_ap = a - p
        d1 = tensor.sum_columns(tensor.square(d_ap))
        d_an = a - n
        d2 = tensor.sum_columns(tensor.square(d_an))
        d = d1 - d2
        d += self.margin
        sign = d > float(0)
        loss = tensor.eltwise_mult(d, sign)
        batchsize = float(a.shape[0])
        if is_train:
            self.ga = (n - p) * (2 / batchsize)
            self.ga.mult_column(sign)
            self.gp = d_ap * (-2 / batchsize)
            self.gp.mult_column(sign)
            self.gn = d_an * (2 / batchsize)
            self.gn.mult_column(sign)
        return (loss.l1(), d1.l1(), d2.l1())

    def backward(self):
        return self.ga, self.gp, self.gn


class CANet(object):
    '''Context-depedent attention modeling'''
    def __init__(self, name, loss, dev, img_size, batchsize=32, debug=True):
        self.debug = debug
        self.name = name
        self.loss = loss
        self.device = dev
        self.batchsize = batchsize
        self.img_size = img_size
        self.layers = []

    def create_net(self, name, img_size, batchsize=32):
        pass

    def to_device(self, dev):
        for lyr in self.layers:
            lyr.to_device(dev)

    def param_names(self):
        pname = []
        for lyr in self.layers:
            pname.extend(lyr.param_names())
        return pname

    def param_values(self):
        pvals = []
        for lyr in self.layers:
            pvals.extend(lyr.param_values())
        return pvals

    def extract_query_feature(self, x, y):
        '''x for street images, y for shop image features'''
        pass

    def extract_db_feature(self, x, tag):
        '''x for shop images'''
        pass

    def retrieval(self, data, result_path, topk=1000):
        bar = trange(data.num_batches, desc='Query Image')
        query_ids = []
        query = None
        data.start(data.load_street_images)
        for i in bar:
            img, item = data.next()
            x = np.squeeze(self.extract_query_feature(img, None))
            query_ids.extend(item)
            if query is None:
                query = np.empty((data.query_size, x.shape[1]), dtype=np.float32)
            query[i*x.shape[0]:(i+1)*x.shape[0]]=x
        print 'query shape', query.shape

        bar = trange(data.shop_batches, desc='Database Image')
        db_ids = []
        db = None
        data.start(data.load_shop_images)
        for i in bar:
            img, item, tag = data.next()
            x = np.squeeze(self.extract_db_feature(img, tag))
            db_ids.extend(item)
            if db is None:
                db = np.empty((data.db_size, x.shape[1]), dtype=np.float32)
            db[i*x.shape[0]:(i+1)*x.shape[0]]=x
        print 'db shape', db.shape

        data.stop()
        '''
        db_ids = [0] * 1000
        query_ids = [0] * 1000
        db = np.random.rand(1000, 10)
        query = np.random.rand(100, 10)
        '''

        t = time.time()
        dist=scipy.spatial.distance.cdist(query, db,'euclidean')
        print('distance computation time = %f' % (time.time() - t))
        sorted_idx=np.argsort(dist,axis=1)[:, 0:topk]
        topdist = np.empty(sorted_idx.shape, dtype=np.float32)
        for i in range(sorted_idx.shape[0]):
            topdist[i] = dist[i, sorted_idx[i]]
        np.save('%s-dist' % result_path, topdist)

        K=100
        target = np.empty((query.shape[0], K), dtype=np.bool)
        for i in range(query.shape[0]):
            for j in range(K):
                target[i,j] = db_ids[sorted_idx[i, j]] == query_ids[i]
        np.save('%s-target' % result_path, target)

        prec = compute_precision(target)
        print 'Precision ',  prec
        np.savetxt('%s-precision.txt' % result_path, prec)
        return query, db, sorted_idx

    def rerank(self, query, db, candidate):
        pass


    def bprop(self, qimg, pimg, nimg, ptag, ntag):
        pass

    def evaluate(self, qimg, pimg, nimg, ptag, ntag):
        pass

    def save(self, fpath):
        params = {}
        for (name, val) in zip(self.param_names(), self.param_values()):
            val.to_host()
            params[name] = tensor.to_numpy(val)
            with open(fpath, 'wb') as fd:
                pickle.dump(params, fd)

    def load(self, fpath):
        with open(fpath, 'rb') as fd:
                params = pickle.load(fd)
                for name, val in zip(self.param_names(), self.param_values()):
                    if name not in params:
                        print 'Param: %s missing in the checkpoint file' % name
                        continue
                    try:
                        val.copy_from_numpy(params[name])
                    except AssertionError as err:
                        print 'Error from copying values for param: %s' % name
                        print 'shape of param vs checkpoint', \
                                val.shape, params[name].shape
                        raise err

    def init_params(self, weight_path=None):
        if weight_path is None:
            for pname, pval in zip(self.param_names(), self.param_values()):
                if 'conv' in pname and len(pval.shape) > 1:
                    initializer.gaussian(pval, 0, pval.shape[1])
                else:
                    pval.set_value(0)
                if self.debug:
                    print pname, pval.shape, pval.l1()
        else:
            self.load(weight_path)


class CANIN(CANet):
    def __init__(self, name, loss, dev, img_size, batchsize=32, debug=True):
        super(CANIN, self).__init__(name, loss, dev, img_size, batchsize, debug)
        self.shared, self.street, self.shop = self.create_net(name, img_size, batchsize)
        self.layers.extend(self.shared)
        self.layers.extend(self.street)
        self.layers.extend(self.shop)
        if debug:
            for lyr in self.layers:
                print lyr.name, lyr.get_output_sample_shape()
        self.to_device(dev)

    def add_conv(self, layers, name, nb_filter, size, stride, pad=0, sample_shape=None):
        if sample_shape != None:
            layers.append(Conv2D('%s-3x3' % name, nb_filter, size, stride, pad=pad,
                           input_sample_shape=sample_shape))
        else:
            layers.append(Conv2D('%s-3x3' % name, nb_filter, size, stride, pad=pad,
                           input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-3x3-relu' % name, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Conv2D('%s-1x1-1' % name, nb_filter, 1, 1, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-1x1-1-relu' % name, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Conv2D('%s-1x1-2' % name, nb_filter, 1, 1, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-1x1-2-relu' % name, input_sample_shape=layers[-1].get_output_sample_shape()))


    def create_net(self, name, img_size, batchsize=32):
        shared = []

        self.add_conv(shared, 'conv1', 96, 5, 2, sample_shape=(3, img_size, img_size))
        shared.append(MaxPooling2D('p1', 3, 2, input_sample_shape=shared[-1].get_output_sample_shape()))

        self.add_conv(shared, 'conv2', 256, 5, 2, 2)
        shared.append(MaxPooling2D('p2', 3, 2, input_sample_shape=shared[-1].get_output_sample_shape()))

        self.add_conv(shared, 'conv3', 384, 3, 1, 1)
        shared.append(MaxPooling2D('p3', 3, 2, input_sample_shape=shared[-1].get_output_sample_shape()))
        slice_layer = Slice('slice', 0, [batchsize], input_sample_shape=shared[-1].get_output_sample_shape())
        shared.append(slice_layer)

        street = []
        self.add_conv(street, 'steet-conv4', 128, 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[0])
        street.append(AvgPooling2D('street-p4', 7, 1, pad=0, input_sample_shape=street[-1].get_output_sample_shape()))
        street.append(Flatten('street-flat', input_sample_shape=street[-1].get_output_sample_shape()))
        street.append(L2Norm('street-l2', input_sample_shape=street[-1].get_output_sample_shape()))

        shop = []
        self.add_conv(shop, 'shop-conv4', 128, 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[1])
        shop.append(AvgPooling2D('shop-p4', 7, 1, pad=0, input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(Flatten('shop-flat', input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(L2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(Slice('shop-slice', 0, [batchsize], input_sample_shape=shop[-1].get_output_sample_shape()))

        return shared, street, shop

    def forward(self, is_train, qimg, pimg, nimg, ptag, ntag):
        x = np.concatenate((qimg, pimg, nimg), axis=0)
        x = tensor.from_numpy(x)
        x.to_device(self.device)
        if self.debug:
            print '------------forward------------'
            print('%30s = %2.8f' % ('data', x.l1()))

        for lyr in self.shared:
            x = lyr.forward(is_train, x)
            if self.debug:
                if type(x) == tensor.Tensor:
                    print('%30s = %2.8f' % (lyr.name, x.l1()))
                else:
                    print('%30s = %2.8f, %2.8f' % (lyr.name, x[0].l1(), x[1].l1()))
        a, b = x
        for lyr in self.street:
            a = lyr.forward(is_train, a)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, a.l1()))
        for lyr in self.shop:
            b = lyr.forward(is_train, b)
            if self.debug:
                if type(b) == tensor.Tensor:
                    print('%30s = %2.8f' % (lyr.name, b.l1()))
                else:
                    print('%30s = %2.8f, %2.8f' % (lyr.name, b[0].l1(), b[1].l1()))
        p, n = b
        return a, p, n

    def backward(self, da, dp, dn):
        if self.debug:
            print '------------backward------------'
        param_grads = []
        dshop = [dp, dn]
        for lyr in self.shop[::-1]:
            dshop, dp = lyr.backward(True, dshop)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, dshop.l1()))
            if dp is not None:
                param_grads.extend(dp[::-1])

        for lyr in self.street[::-1]:
            da, dp = lyr.backward(True, da)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, da.l1()))
            if dp is not None:
                param_grads.extend(dp[::-1])

        d = [da, dshop]
        for lyr in self.shared[::-1]:
            d, dp = lyr.backward(True, d)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, d.l1()))
            if dp is not None:
                param_grads.extend(dp[::-1])
        return param_grads[::-1]

    def bprop(self, qimg, pimg, nimg, ptag, ntag):
        assert self.batchsize == qimg.shape[0], 'batchsize not correct %d vs %d' % (self.batchsize, qimg.shape[0])
        a, p, n = self.forward(True, qimg, pimg, nimg, ptag, ntag)
        loss = self.loss.forward(True, a, p, n)
        da, dp, dn = self.loss.backward()
        return self.backward(da, dp, dn), loss

    def evaluate(self, qimg, pimg, nimg, ptag, ntag):
        assert self.batchsize == qimg.shape[0], 'batchsize not correct %d vs %d' % (self.batchsize, qimg.shape[0])
        a, p, n = self.forward(False, qimg, pimg, nimg, ptag, ntag)
        loss = self.loss.forward(False, a, p, n)
        return loss

    def forward_layers(self, x, layers):
        x = tensor.from_numpy(x)
        x.to_device(self.device)
        if self.debug:
            print '------------forward------------'
            print('%30s = %2.8f' % ('data', x.l1()))

        for lyr in layers:
            x = lyr.forward(False, x)
            if self.debug:
                if type(x) == tensor.Tensor:
                    print('%30s = %2.8f' % (lyr.name, x.l1()))
        return tensor.to_numpy(x)


    def extract_query_feature(self, x, y):
        return self.forward_layers(x, self.shared[0:-2] + self.street)

    def extract_db_feature(self, x, tag):
        return self.forward_layers(x, self.shared[0:-2] + self.street[0:-2])
