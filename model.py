from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D, Flatten, Slice
from singa import initializer
from singa import layer
from singa import loss
from singa import tensor
import cPickle as pickle
import logging
import os

import numpy as np
from numpy.core.umath_tests import inner1d
import scipy.spatial
from tqdm import trange
import time

logger = logging.getLogger(__name__)


def update_perf(his, cur, a=0.8):
    '''Accumulate the performance by considering history and current values.'''
    return his * a + cur * (1 - a)


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


class Softmax(layer.Layer):
    def __init__(self, name, input_sample_shape):
        super(Softmax, self).__init__(name)
        self.a = None

    def forward(self, is_train, x):
        assert len(x.shape) == 2, 'softmax input should be 2d-array'
        a = x - np.max(x, axis=1)[:, np.newaxis]
        a = np.exp(a)
        a /= np.sum(a, axis=1)[:, np.newaxis]
        if is_train:
            self.a = a
        return a

    def backward(self, is_train, dy):
        c = np.einsum('ij, ij->i', dy, self.a)
        return self.a * (dy - c[:, np.newaxis]), []


class Aggregation(layer.Layer):
    def __init__(self, name, input_sample_shape):
        super(Aggregation, self).__init__(name)
        self.c, h, w = input_sample_shape[0]
        assert h * w == input_sample_shape[1][0], \
                '# locations not match: %d vs %d' % (h * w, input_sample_shape[1][0])
        self.x = None

    def forward(self, is_train, xs):
        x = xs[0].reshape((xs[0].shape[0], self.c, -1))
        w = xs[1]
        if is_train:
            self.x = x
            self.w = w
        return np.einsum('ijk, ik -> ij', x, w)

    def backward(self, is_train, dy):
        dw = np.einsum('ij, ijk -> ik', dy, self.x)
        dx = np.einsum('ij, ik -> ijk', dy, self.w)
        return [dx, dw], []


class TagEmbedding(layer.Layer):
    def __init__(self, name, num_output, input_sample_shape):
        super(TagEmbedding, self).__init__(name)
        self.W = tensor.Tensor((input_sample_shape[0], num_output))
        initializer.gaussian(self.W, input_sample_shape[0], num_output)

    def param_names(self):
        return ['%s_weight' % self.name]

    def param_values(self):
        return [self.W]

    def forward(self, is_train, x):
        if is_train:
            self.x = x
        W = tensor.to_numpy(self.W)
        # b = self.to_numpy(self.b)
        return np.dot(x, W)  # + b[np.newaxis, :]

    def backward(self, is_train, dy):
        dw = np.einsum('id, ij -> dj', self.x, dy)
        # db = np.sum(dt, axis=0)
        return [], [tensor.from_numpy(dw)]


class ProductAttention(layer.Layer):
    def __init__(self, name, input_sample_shape):
        super(ProductAttention, self).__init__(name)
        self.c, self.h, self.w = input_sample_shape[0]
        assert self.c == input_sample_shape[1][0], \
                '# channels != tag embed dim: %d vs %d' % (self.c, input_sample_shape[1][0])
        self.x = None
        self.t = None

    def forward(self, is_train, xs):
        x = xs[0].reshape((xs[0].shape[0], self.c, -1))
        t = xs[1]
        if is_train:
            self.x = x
            self.t = xs[1]
        return np.einsum('ijk, ij->ik', x, t)

    def backward(self, is_train, dy):
        dt = np.einsum('ik, ijk -> ij', dy, self.x)
        dx = np.einsum('ij, ik -> ijk', self.t, dy)
        return [dx, dt], []


class MLPAttention(layer.Layer):
    def __init__(self, name, dim=128, input_sample_shape=None):
        super(MLPAttention, self).__init__(name)
        self.c, self.h, self.w = input_sample_shape[0]
        assert self.c == input_sample_shape[1][0], \
                '# channels != tag embed dim: %d vs %d' % (self.c, input_sample_shape[1][0])
        self.h = tensor.Tensor((dim,))
        self.U = tensor.Tensor((self.c, dim))
        self.V = tensor.Tensor((self.c, dim))
        initializer.gaussian(self.U, self.c, dim)
        initializer.gaussian(self.V, self.c, dim)
        initializer.gaussian(self.h, dim, 0)
        self.x = None
        self.t = None

    def get_output_sample_shape(self):
        return (self.h * self.w, )

    def param_names(self):
        return [self.name+'_U', self.name+'_V', self.name+'_h']

    def param_values(self):
        return [self.U, self.V, self.h]

    def forward(self, is_train, xs):
        self.npU = tensor.to_numpy(self.U)
        self.npV = tensor.to_numpy(self.V)
        self.nph = tensor.to_numpy(self.h)

        x = xs[0].reshape((xs[0].shape[0], self.c, -1))
        ctx = xs[1]
        a1 = np.einsum('ncl, ck -> nkl', x, self.npU)
        a2 = np.einsum('nc, ck -> nk', ctx, self.npV)
        a = np.tanh(a1 + a2[:, : , np.newaxis])
        if is_train:
            self.x = x
            self.ctx = ctx
            self.a = a
        return np.einsum('nkl, k->nl', a, self.nph)

    def backward(self, is_train, dy):
        dh = np.einsum('nl, nkl -> k', dy, self.a)
        da = np.einsum('nl, k -> nkl', dy, self.nph)
        da *= 1 - (self.a**2)
        dU = np.einsum('nkl, ncl -> ck', da, self.x)
        dx = np.einsum('nkl, ck -> ncl', da, self.npU)
        da = np.einsum('nkl->nk', da)
        dctx = np.einsum('nk, ck ->nc', da, self.npV)
        dV = np.einsum('nk, nc ->ck', da, self.ctx)
        return [dx, dctx], [tensor.from_numpy(d) for d in [dU, dV, dh]]


class TagAttention(layer.Layer):
    def __init__(self, name, input_sample_shape, debug=False):
        super(TagAttention, self).__init__(name)
        self.c, self.h, self.w = input_sample_shape[0]
        l = self.h * self.w
        self.embed = TagEmbedding('%s_embed' % name, self.c, input_sample_shape[1])
        self.attention = ProductAttention('%s_attention' % name, [input_sample_shape[0], (self.c,)])
        self.softmax = Softmax('%s_softmax' % name, (l,))
        self.agg = Aggregation('%s_agg' % name, [input_sample_shape[0], (l,)])
        self.dev = None
        self.debug= debug

    def get_output_sample_shape(self):
        return  (self.c, )

    def param_names(self):
        return self.embed.param_names()

    def param_values(self):
        return self.embed.param_values()

    def display(self, name, val):
        if self.debug:
            print('%30s = %2.8f' % (name, np.average(np.abs(val))))

    def forward(self, is_train, x):
        if is_train:
            self.dev = x[0].device
        img = tensor.to_numpy(x[0])
        t = self.embed.forward(is_train, x[1])
        self.display(self.embed.name, t)
        w = self.attention.forward(is_train, [img, t])
        self.display(self.attention.name, w)
        w = self.softmax.forward(is_train, w)
        self.display(self.softmax.name, w)
        y = self.agg.forward(is_train, [img, w])
        self.display(self.agg.name, y)
        return tensor.from_numpy(y)

    def backward(self, is_train, dy):
        dy = tensor.to_numpy(dy)
        [dx1, dw], _ = self.agg.backward(is_train, dy)
        self.display(self.agg.name, dx1)
        dw, _ = self.softmax.backward(is_train, dw)
        self.display(self.softmax.name, dw)
        [dx2, dt], _ = self.attention.backward(is_train, dw)
        self.display(self.attention.name, dx2)
        _, dW = self.embed.backward(is_train, dt)
        dx = dx1 + dx2
        dx = tensor.from_numpy(dx.reshape((dx.shape[0], self.c, self.h, self.w)))
        dx.to_device(self.dev)
        return dx, dW

class ContextAttention(layer.Layer):
    def __init__(self, name, input_sample_shape, debug=False):
        super(ContextAttention, self).__init__(name)
        self.c, self.h, self.w = input_sample_shape[0]
        l = self.h * self.w
        assert self.c == input_sample_shape[1][0], \
            'channel mis-match. street vs shop: %d, %d' % (self.c, input_sample_shape[1][0])
        self.attention = ProductAttention('%s_attention' % name, [input_sample_shape[0], (self.c,)])
        self.softmax = Softmax('%s_softmax' % name, (l,))
        self.agg = Aggregation('%s_agg' % name, [input_sample_shape[0], (l,)])
        self.dev = None
        self.debug= debug

    def get_output_sample_shape(self):
        return  (self.c, )

    def param_names(self):
        return self.attention.param_names()

    def param_values(self):
        return self.attention.param_values()

    def display(self, name, val):
        if self.debug:
            if type(val) == tensor.Tensor:
                print('%30s = %2.8f' % (name, np.average(np.abs(val))))
            else:
                for v in val:
                    print('%30s = %2.8f' % (name, np.average(np.abs(v))))

    def forward(self, is_train, x):
        if is_train:
            self.dev = x[0].device
        img = tensor.to_numpy(x[0])
        w = self.attention.forward(is_train, [img, x[1]])
        self.display(self.attention.name, w)
        w = self.softmax.forward(is_train, w)
        self.display(self.softmax.name, w)
        y = self.agg.forward(is_train, [img, w])
        self.display(self.agg.name, y)
        return tensor.from_numpy(y)

    def backward(self, is_train, dy):
        dy = tensor.to_numpy(dy)
        [dx1, dw], _ = self.agg.backward(is_train, dy)
        self.display(self.agg.name, dx1)
        dw, _ = self.softmax.backward(is_train, dw)
        self.display(self.softmax.name, dw)
        [dx2, dctx], dp = self.attention.backward(is_train, dw)
        self.display(self.attention.name, [dx2, dctx])
        dx = dx1 + dx2
        dx = tensor.from_numpy(dx.reshape((dx.shape[0], self.c, self.h, self.w)))
        dx.to_device(self.dev)
        return [dx, dctx], dp


def loss_bp(is_train, a1, a2, p, n, margin):
    d_ap = a1 - p
    d1 = inner1d(d_ap, d_ap)
    d_an = a2 - n
    d2 = inner1d(d_an, d_an)
    d = d1 - d2
    d += margin
    sign = d > 0
    loss = d * sign
    batchsize = float(a1.shape[0])
    grads = None
    if is_train:
        gp = d_ap * sign[:, np.newaxis] * (-2 / batchsize)
        gn = d_an * sign[:, np.newaxis] * (2 / batchsize)
        grads = [-gp, -gn, gp, gn]
    return (np.array([loss.mean(), sign.mean(), d1.mean(), d2.mean()]), grads)


class TripletLoss(loss.Loss):
    def __init__(self, margin=0.1, nshift=1, nuser=1, nshop=1):
        self.margin = margin
        self.nshift = nshift
        self.nuser = nuser
        self.nshop = nshop
        self.guser = None
        self.gshop = None
        self.deva = None
        self.devb = None

    def forward(self, is_train, user_fea, shop_fea, pids):
        if type(user_fea) == tensor.Tensor:
            ufea = tensor.to_numpy(user_fea)
        if type(shop_fea) == tensor.Tensor:
            sfea = tensor.to_numpy(shop_fea)
        if is_train:
            self.deva = user_fea.device
            self.devb = shop_fea.device
            self.guser = np.zeros(ufea.shape, dtype=np.float32)
            self.gshop = np.zeros(sfea.shape, dtype=np.float32)
        ret = None
        for i in range(1, self.nshift+1):
            offset = i * self.nshop
            idx = range(offset, sfea.shape[0]) + range(0, offset)
            loss, grads = loss_bp(is_train, ufea, ufea, sfea, sfea[idx], self.margin)
            if is_train:
                self.guser += grads[0] + grads[1]
                self.gshop += grads[2]
                self.gshop[idx] += grads[3]
            if ret is None:
                ret = np.zeros(loss.shape)
            ret += loss
        return ret/self.nshift

    def backward(self):
        tguser = tensor.from_numpy(self.guser/self.nshift)
        tgshop = tensor.from_numpy(self.gshop/self.nshift)
        tguser.to_device(self.deva)
        tgshop.to_device(self.devb)
        return tguser, tgshop


class YNet(object):
    '''Context-depedent attention modeling'''
    def __init__(self, name, loss, dev, img_size, batchsize=32, nuser=1, nshop=1, debug=False):
        self.debug = debug
        self.name = name
        self.loss = loss
        self.device = dev
        self.batchsize = batchsize
        self.img_size = img_size
        self.nuser = nuser
        self.nshop = nshop

        self.shared, self.user, self.shop = self.create_net(name, img_size, batchsize)
        self.layers = self.shared + self.user + self.shop
        if debug:
            for lyr in self.layers:
                print lyr.name, lyr.get_output_sample_shape()
        self.to_device(dev)

    def create_net(self, name, img_size, batchsize):
        pass

    def forward(self, is_train, data, to_loss):
        pass

    def backward(self):
        pass

    def bprop(self, data):
        pass

    def extract_query_feature_on_batch(self, data):
        pass

    def extract_db_feature_on_batch(self, data):
        pass

    def to_device(self, dev):
        for lyr in self.layers:
            lyr.to_device(dev)

    def param_names(self, all=False):
        pname = []
        for lyr in self.layers:
            pname.extend(lyr.param_names())
        return pname

    def param_values(self, all=False):
        pvals = []
        for lyr in self.layers:
            pvals.extend(lyr.param_values())
        return pvals

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

    def save(self, fpath):
        params = {}
        for (name, val) in zip(self.param_names(True), self.param_values(True)):
            val.to_host()
            params[name] = tensor.to_numpy(val)
            with open(fpath, 'wb') as fd:
                pickle.dump(params, fd)

    def load(self, fpath):
        with open(fpath, 'rb') as fd:
            params = pickle.load(fd)
            for name, val in zip(self.param_names(True), self.param_values(True)):
                if name not in params:
                    print 'Param: %s missing in the checkpoint file' % name
                    continue
                try:
                    if name == 'conv1-3x3_weight' and len(params[name].shape) == 4:
                        oc, ic, h, w = params[name].shape
                        assert ic == 3, 'input channel should be 3'
                        w = np.reshape(params[name], (oc, ic, -1))
                        w[:, [0,1,2], :] = w[:, [2,1,0], :]
                        params[name] = np.reshape(w, (oc,-1))
                    val.copy_from_numpy(params[name])
                    if self.debug:
                        print name, params[name].shape, val.l1()

                except AssertionError as err:
                    print 'Error from copying values for param: %s' % name
                    print 'shape of param vs checkpoint', \
                            val.shape, params[name].shape
                    raise err

    def train_on_epoch(self, epoch, data, opt, lr, nuser, nshop):
        loss = None
        data.start(nuser, nshop)
        bar = trange(data.num_batches, desc='Epoch %d' % epoch)
        for b in bar:
            grads, l, t = self.bprop(data)
            if self.debug:
                print('-------------prams---------------')
            for pname, pval, pgrad in zip(self.param_names(), self.param_values(), grads):
                if self.debug:
                    print('%30s = %f, %f' % (pname, pval.l1(), pgrad.l1()))
                opt.apply_with_lr(epoch, lr, pgrad, pval, str(pname), b)
            if loss is None:
                loss = np.zeros(l.shape)
            loss = update_perf(loss, l)
            bar.set_postfix(train_loss=np.array_str(loss), load_time=t[0], bptime=t[1])
            if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
                break
        data.stop()
        return loss

    def extract_query_feature(self, data):
        '''x for user images, y for shop image features'''
        data.start(1, 0)
        bar = trange(data.num_batches, desc='Query Image')
        query_fea, query_pid = None, []
        for i in bar:
            fea, pid = self.extract_query_feature_on_batch(data)
            query_pid.extend(pid)
            if query_fea is None:
                query_fea = np.empty((data.num_batches * self.batchsize, fea.shape[1]), dtype=np.float32)  #data.query_size,
            query_fea[i*fea.shape[0]:(i+1)*fea.shape[0]] = fea
        data.stop()
        if self.debug:
            print 'query shape', query_fea.shape
        return query_fea, query_pid

    def extract_db_feature(self, data):
        '''x for shop images'''
        data.start(0, 1)
        bar = trange(data.num_batches, desc='Database Image')
        db_fea, db_pid = None, []
        for i in bar:
            fea, pid = self.extract_db_feature_on_batch(data)
            db_pid.extend(pid)
            if db_fea is None:
                db_fea = np.empty((data.num_batches * self.batchsize,  fea.shape[1]), dtype=np.float32) #data.db_size,
            db_fea[i*fea.shape[0]:(i+1)*fea.shape[0]] = fea
        data.stop()
        if self.debug:
            print 'db shape', db_fea.shape
        return db_fea, db_pid

    def evaluate_on_epoch(self, epoch, data, nuser, nshop):
        loss = None
        data.start(nuser, nshop)
        bar = trange(data.num_batches, desc='Epoch %d' % epoch)
        for b in bar:
            l, _, _ = self.forward(False, data)
            if loss is None:
                loss = np.zeros(l.shape)
            loss += l
        loss /= data.num_batches
        data.stop()
        return loss

    def retrieval(self, data, result_path, topk=100):
        query_fea, query_id = self.extract_query_feature(data)
        db_fea, db_id = self.extract_db_feature(data)
        dist=scipy.spatial.distance.cdist(query_fea, db_fea,'euclidean')
        target, sorted_idx = self.match(dist, query_id, db_id, topk)
        prec = compute_precision(target)
        # np.save('%s-dist' % result_path, topdist)
        # np.save('%s-target' % result_path, target)
        # np.savetxt('%s-precision.txt' % result_path, prec)
        return prec, sorted_idx

    def match(self, dist, query_id, db_id, topk=100):
        # logger.info('distance computation time = %f' % (time.time() - t))
        sorted_idx=np.argsort(dist,axis=1)[:, 0:topk]
        topdist = np.empty(sorted_idx.shape, dtype=np.float32)
        for i in range(sorted_idx.shape[0]):
            topdist[i] = dist[i, sorted_idx[i]]

        target = np.empty((len(query_id), topk), dtype=np.bool)
        for i in range(len(query_id)):
            for j in range(topk):
                target[i,j] = db_id[sorted_idx[i, j]] == query_id[i]
        return target, sorted_idx  # , topdist

    def rerank(self, data, result_path, candidate_size=1000, topk=100):
        pass
        '''
        db_ids = [0] * 1000
        query_ids = [0] * 1000
        db = np.random.rand(1000, 10)
        query = np.random.rand(100, 10)
        '''

class YNIN(YNet):
    def add_conv(self, layers, name, nb_filter, size, stride, pad=0, sample_shape=None):
        if sample_shape != None:
            layers.append(Conv2D('%s-3x3' % name, nb_filter[0], size, stride, pad=pad,
                           input_sample_shape=sample_shape))
        else:
            layers.append(Conv2D('%s-3x3' % name, nb_filter[0], size, stride, pad=pad,
                           input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-3x3-relu' % name, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Conv2D('%s-1x1-1' % name, nb_filter[1], 1, 1, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-1x1-1-relu' % name, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Conv2D('%s-1x1-2' % name, nb_filter[2], 1, 1, input_sample_shape=layers[-1].get_output_sample_shape()))
        layers.append(Activation('%s-1x1-2-relu' % name, input_sample_shape=layers[-1].get_output_sample_shape()))


    def create_net(self, name, img_size, batchsize=32):
        shared = []

        self.add_conv(shared, 'conv1', [96, 96, 96], 11, 4, sample_shape=(3, img_size, img_size))
        shared.append(MaxPooling2D('p1', 3, 2, pad=1, input_sample_shape=shared[-1].get_output_sample_shape()))

        self.add_conv(shared, 'conv2', [256, 256, 256], 5, 1, 2)
        shared.append(MaxPooling2D('p2', 3, 2, pad=0, input_sample_shape=shared[-1].get_output_sample_shape()))

        self.add_conv(shared, 'conv3', [384, 384, 384], 3, 1, 1)
        shared.append(MaxPooling2D('p3', 3, 2, pad=0, input_sample_shape=shared[-1].get_output_sample_shape()))
        slice_layer = Slice('slice', 0, [batchsize*self.nuser], input_sample_shape=shared[-1].get_output_sample_shape())
        shared.append(slice_layer)

        user = []
        self.add_conv(user, 'street-conv4', [1024, 1024, 1000] , 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[0])
        user.append(AvgPooling2D('street-p4', 6, 1, pad=0, input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(Flatten('street-flat', input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(L2Norm('street-l2', input_sample_shape=user[-1].get_output_sample_shape()))

        shop = []
        self.add_conv(shop, 'shop-conv4', [1024, 1024, 1000], 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[1])
        shop.append(AvgPooling2D('shop-p4', 6, 1, pad=0, input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(Flatten('shop-flat', input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(L2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))

        return shared, user, shop

    def put_input_to_gpu(self, imgs):
        x = tensor.from_numpy(imgs)
        x.to_device(self.device)
        if self.debug:
            print '------------forward------------'
            print('%30s = %2.8f' % ('data', x.l1()))
        return x

    def forward_layers(self, is_train, x, layers):
        for lyr in layers:
            x = lyr.forward(is_train, x)
            if self.debug:
                if type(x) == tensor.Tensor:
                    print('%30s = %2.8f' % (lyr.name, x.l1()))
                else:
                    print('%30s = %2.8f, %2.8f' % (lyr.name, x[0].l1(), x[1].l1()))
        return x

    def forward(self, is_train, data, to_loss=True):
        t1 = time.time()
        imgs, pids = data.next()
        t2 = time.time()
        x = self.put_input_to_gpu(imgs)
        a, b = self.forward_layers(is_train, x, self.shared)
        a = self.forward_layers(is_train, a, self.user)
        b = self.forward_layers(is_train, b, self.shop)
        loss = self.loss.forward(is_train, a, b, pids)
        return loss, t2 - t1, time.time() -t2

    def backward_layers(self, dy, layers, dparam):
        for lyr in layers:
            dx, dp = lyr.backward(True, dy)
            if self.debug:
                if type(dx) == tensor.Tensor:
                    print('%30s = %2.8f' % (lyr.name, dx.l1()))
                else:
                    for x in dx:
                        print('%30s = %2.8f' % (lyr.name, x.l1()))
            if dp is not None:
                dparam.extend(dp[::-1])

    def backward(self):
        if self.debug:
            print '------------backward------------'
        param_grads = []
        duser, dshop = self.loss.backward()
        dshop = self.backward_layers(dshop, self.shop[::-1], param_grads)
        duser = self.backward_layers(duser, self.user[::-1], param_grads)
        self.backward_layers([duser, dshop], self.shared[::-1], param_grads)
        return param_grads[::-1]

    def bprop(self, data):
        loss, t1, t2 = self.forward(True, data)
        tic = time.time()
        ret = self.backward()
        return ret, loss, [t1, t2 + time.time() - tic]

    def extract_query_feature_on_batch(self, data):
        img, pid = data.next()
        img = self.put_input_to_gpu(img)
        fea = self.forward_layers(False, img, self.shared[0:-1] + self.user)
        return tensor.to_numpy(fea), pid

    def extract_db_feature_on_batch(self, data):
        img, pid = data.next()
        img = self.put_input_to_gpu(img)
        fea = self.forward_layers(False, img, self.shared[0:-1] + self.shop)
        return tensor.to_numpy(fea), pid


class TagNIN(YNIN):
    def create_net(self, name, img_size, batchsize=32, ntags=20):
        shared = []

        self.add_conv(shared, 'conv1', [96, 96, 96], 11, 4, sample_shape=(3, img_size, img_size))
        shared.append(MaxPooling2D('p1', 3, 2, pad=1, input_sample_shape=shared[-1].get_output_sample_shape()))

        self.add_conv(shared, 'conv2', [256, 256, 256], 5, 1, 2)
        shared.append(MaxPooling2D('p2', 3, 2, pad=0, input_sample_shape=shared[-1].get_output_sample_shape()))

        self.add_conv(shared, 'conv3', [384, 384, 384], 3, 1, 1)
        shared.append(MaxPooling2D('p3', 3, 2, pad=0, input_sample_shape=shared[-1].get_output_sample_shape()))
        slice_layer = Slice('slice', 0, [batchsize*self.nuser], input_sample_shape=shared[-1].get_output_sample_shape())
        shared.append(slice_layer)

        user = []
        self.add_conv(user, 'street-conv4', [1024, 1024, 1000] , 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[0])
        user.append(AvgPooling2D('street-p4', 6, 1, pad=0, input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(Flatten('street-flat', input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(L2Norm('street-l2', input_sample_shape=user[-1].get_output_sample_shape()))

        shop = []
        self.add_conv(shop, 'shop-conv4', [1024, 1024, 1000], 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[1])
        # shop.append(AvgPooling2D('shop-p4', 6, 1, pad=0, input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(TagAttention('shop-tag', input_sample_shape=[shop[-1].get_output_sample_shape(), (ntags, )], debug=self.debug))
        shop.append(L2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))
        return shared, user, shop

    def forward(self, is_train, data, to_loss=True):
        t1 = time.time()
        imgs, pids = data.next()
        t2 = time.time()
        imgs = self.put_input_to_gpu(imgs)
        a, b = self.forward_layers(is_train, imgs, self.shared)
        a = self.forward_layers(is_train, a, self.user)
        b = self.forward_layers(is_train, b, self.shop[0:-2])
        b = self.shop[-2].forward(is_train, [b, data.tag2vec(pids[a.shape[0]:])])
        b = self.forward_layers(is_train, b, self.shop[-1:])
        loss = self.loss.forward(is_train, a, b, pids)
        return loss, t2 - t1, time.time() - t2

    def extract_db_feature_on_batch(self, data):
        img, pid = data.next()
        img = self.put_input_to_gpu(img)
        fea = self.forward_layers(False, img, self.shared[0:-1] + self.shop[0:-2])
        fea = self.shop[-2].forward(False, [fea, data.tag2vec(pid)])
        fea = self.forward_layers(False, fea, self.shop[-1:])
        return tensor.to_numpy(fea), pid


class ContextNIN(TagNIN):
    def param_names(self, flag=False):
        pname = []
        layers = self.user
        if flag:
            layers = self.layers
        for lyr in layers:
            pname.extend(lyr.param_names())
        return pname

    def param_values(self, flag=False):
        pvals = []
        layers = self.user
        if flag:
            layers = self.layers
        for lyr in layers:
            pvals.extend(lyr.param_values())
        return pvals

    def create_net(self, name, img_size, batchsize=32, ntags=20):
        shared = []

        self.add_conv(shared, 'conv1', [96, 96, 96], 11, 4, sample_shape=(3, img_size, img_size))
        shared.append(MaxPooling2D('p1', 3, 2, pad=1, input_sample_shape=shared[-1].get_output_sample_shape()))

        self.add_conv(shared, 'conv2', [256, 256, 256], 5, 1, 2)
        shared.append(MaxPooling2D('p2', 3, 2, pad=0, input_sample_shape=shared[-1].get_output_sample_shape()))

        self.add_conv(shared, 'conv3', [384, 384, 384], 3, 1, 1)
        shared.append(MaxPooling2D('p3', 3, 2, pad=0, input_sample_shape=shared[-1].get_output_sample_shape()))
        slice_layer = Slice('slice', 0, [batchsize*self.nuser], input_sample_shape=shared[-1].get_output_sample_shape())
        shared.append(slice_layer)

        shop = []
        self.add_conv(shop, 'shop-conv4', [1024, 1024, 1000], 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[1])
        # shop.append(AvgPooling2D('shop-p4', 6, 1, pad=0, input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(TagAttention('shop-tag', input_sample_shape=[shop[-1].get_output_sample_shape(), (ntags, )], debug=self.debug))
        shop.append(L2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))

        user = []
        self.add_conv(user, 'street-conv4', [1024, 1024, 1000] , 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[0])
        # user.append(AvgPooling2D('street-p4', 6, 1, pad=0, input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(ContextAttention('street-cxt', input_sample_shape=[user[-1].get_output_sample_shape(), shop[-1].get_output_sample_shape()], debug=self.debug))
        user.append(L2Norm('street-l2', input_sample_shape=user[-1].get_output_sample_shape()))

        return shared, user, shop

    def forward(self, is_train, data, to_loss=True):
        t1 = time.time()
        imgs, pids = data.next()
        t2 = time.time()
        imgs = self.put_input_to_gpu(imgs)

        a, b = self.forward_layers(is_train, imgs, self.shared)

        b = self.forward_layers(is_train, b, self.shop[0:-2])
        b = self.shop[-2].forward(is_train, [b, data.tag2vec(pids[a.shape[0]:])])

        a = self.forward_layers(is_train, a, self.user[0:-2])
        a = self.forward_layers(is_train, [a, b], self.user[-2:])

        b = self.forward_layers(is_train, b, self.shop[-1:])
        loss = self.loss.forward(is_train, a, b, pids)
        return loss, t2 - t1, time.time() - t2

    def backward(self):
        if self.debug:
            print '------------backward------------'
        # dp_shop, dp_user = [], []
        dp_user = []
        duser, _ = self.loss.backward()
        # dshop1 = self.backward_layers(dshop, self.shop[-1:-2:-1], dp_shop)
        duser, _ = self.backward_layers(duser, self.user[-1:-3:-1], dp_user)
        # dshop = dshop1 + dshop2
        # dshop = self.backward_layers(dshop, self.shop[-2::-1], dp_shop)
        duser = self.backward_layers(duser, self.user[-3::-1], dp_user)
        # param_grads = dp_shop + dp_user
        # self.backward_layers([duser, dshop], self.shared[::-1], param_grads)
        return dp_user[::-1]

    def search_on_batch(self, data, db_fea, db_pid, topk):
        img, pid = data.next()
        dist =np.zeros((img.shape[0], db_fea.shape[0]))
        for i in range(img.shape[0]):
            fea = np.repeat(img[0], db_fea.shape[1], axis=0)
            fea = self.put_input_to_gpu(fea)
            fea = self.forward_layers(False, fea, self.shared[0:-1] + self.user[0:-2])
            fea = self.forward_layers(False, [fea, db_fea[i]], self.user[-2:])
            dist[i, :] = np.sum((fea - db_fea[i]) ** 2, axis=1)
        return self.match(dist, pid, db_pid, topk)


    def retrieval(self, data, result_path, topk=100):
        sorted_idx, db_fea, db_pid = np.load(os.path.join(result_path, 'init_search.npz'))
        data.start(1, 0)
        bar = trange(data.num_batches, desc='Query Image')
        target = np.empty(sorted_idx.shape, dtype=np.bool)
        new_sorted_idx = np.empty(sorted_idx.shape, dtype=np.int)
        for i in bar:
            db = np.empty((self.batchsize, sorted_idx.shape[1], db_fea.shape[1]))
            pid = [None] * self.batchsize
            for j in range(self.batchsize):
                db[j,:, :] = db_fea[sorted_idx[i * self.batchsize + j]]
                pid[j] = db_pid[sorted_idx[i * self.batchsize + j]]
            t, idx = self.search_on_batch(data, db, pid, topk)
            target[i * self.batchsize: (i + 1) * self.batchsize] = t
            new_sorted_idx[i * self.batchsize: (i + 1) * self.batchsize] = idx
        prec = compute_precision(target)
        return prec, new_sorted_idx
