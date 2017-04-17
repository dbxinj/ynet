from tagnet import Softmax, Aggregation, ProductAttention, TagNIN

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
        self.debug= debug

    def get_output_sample_shape(self):
        return (self.c, )

    def param_names(self):
        return self.attention.param_names()

    def param_values(self):
        return self.attention.param_values()

    def forward(self, is_train, x):
        img, ctx = x[0], x[1]
        w = self.attention.forward(is_train, [img, ctx])
        if self.debug:
            show_debuginfo(self.attention.name, w)
        w = self.softmax.forward(is_train, w)
        if self.debug:
            show_debuginfo(self.softmax.name, w)
        y = self.agg.forward(is_train, [img, w])
        if self.debug:
            show_debuginfo(self.agg.name, y)
        return tensor.from_numpy(y)

    def backward(self, is_train, dy):
        [dx1, dw], _ = self.agg.backward(is_train, dy)
        if self.debug:
            show_debuginfo(self.agg.name, dx1)
        dw, _ = self.softmax.backward(is_train, dw)
        if self.debug:
            show_debuginfo(self.softmax.name, dw)
        [dx2, dctx], dp = self.attention.backward(is_train, dw)
        if self.debug:
            show_debuginfo(self.attention.name, [dx2, dctx])
        dx = dx1 + dx2
        dx = dx.reshape((dx.shape[0], self.c, self.h, self.w))
        return [dx, dctx], dp


class QuadLoss(object):
    def __init__(self, margin=0.1, nshift=1):
        self.margin = margin
        self.nshift = nshift
        self.guser = None
        self.gshop = None

    def forward(self, is_train, ufea, sfea, pids):
        if is_train:
            self.guser = np.zeros(ufea.shape, dtype=np.float32)
            self.gshop = np.zeros(ufea.shape, dtype=np.float32)
        ret = None
        bs = ufea.shape[0] / (self.nshift + 1)
        for i in range(1, self.nshift+1):
            s, e = i*bs, (i+1)*bs
            loss, grads = loss_bp(is_train, ufea[0:bs], ufea[s: e], sfea[0:bs], sfea[s:e], self.margin)
            if is_train:
                self.guser[0:bs] += grads[0]
                self.guser[s:e] += grads[1]
                self.gshop[0:bs] += grads[2]
                self.gshop[s:e] += grads[3]
            if ret is None:
                ret = np.zeros(loss.shape)
            ret += loss
        return ret/self.nshift

    def backward(self):
        return self.guser/self.nshift, self.gshop/self.nshift


class CtxNIN(TagNIN):
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
        shop.append(NpL2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))

        user = []
        self.add_conv(user, 'street-conv4', [1024, 1024, 1000] , 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[0])
        # user.append(AvgPooling2D('street-p4', 6, 1, pad=0, input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(ContextAttention('street-cxt', input_sample_shape=[user[-1].get_output_sample_shape(), shop[-1].get_output_sample_shape()], debug=self.debug))
        user.append(NpL2Norm('street-l2', input_sample_shape=user[-1].get_output_sample_shape()))

        return shared, user, shop

    def forward(self, is_train, data):
        t1 = time.time()
        imgs, pids = data.next()
        t2 = time.time()
        imgs = self.put_input_to_gpu(imgs)

        a, b = self.forward_layers(is_train, imgs, self.shared)

        b = self.forward_layers(is_train, b, self.shop[0:-2])
        ctx = self.shop[-2].forward(is_train, [b, data.tag2vec(pids[a.shape[0]:])])
        normb = self.forward_layers(is_train, b, self.shop[-1:])

        a = self.forward_layers(is_train, a, self.user[0:-2])
        a = tensor.to_numpy(a)
        a = np.tile(a, [self.nshift, 1, 1, 1])
        shifted_ctx = np.empty((a.shape[0], ctx.shape[1]))
        shifted_normb = np.empty(shifted_ctx.shape)
        s = ctx.shape[0]
        for i in range(self.nshift + 1):
            idx = range(i, s) + range(0, i)
            shifted_ctx[i * s: i * s + s] = ctx[idx]
            shifted_normb[i * s: i * s + s] = normb[idx]

        a = self.forward_layers(is_train, [a, shifted_ctx], self.user[-2:])

        loss = self.loss.forward(is_train, a, shifted_normb, pids)
        return loss, t2 - t1, time.time() - t2

    def backward(self):
        if self.debug:
            print '------------backward------------'
        # dp_shop, dp_user = [], []
        dp_user = []
        duser, _ = self.loss.backward()
        # dshop1 = self.backward_layers(dshop, self.shop[-1:-2:-1], dp_shop)
        dshifted_user, _ = self.backward_layers(duser, self.user[-1:-3:-1], dp_user)
        s = dshifted_user.shape
        s[0] /= self.nshift + 1
        duser = np.zeros(s)
        for i in range(self.nshift + 1):
            duser += dshifted_user[i * s[0]: (i+1) * s[0]]
        duser = tensor.to_numpy(duser)
        duser.to_device(self.device)
        # dshop = dshop1 + dshop2
        # dshop = self.backward_layers(dshop, self.shop[-2::-1], dp_shop)
        duser = self.backward_layers(duser, self.user[-3::-1], dp_user)
        # param_grads = dp_shop + dp_user
        # self.backward_layers([duser, dshop], self.shared[::-1], param_grads)
        return dp_user[::-1]

    def rerank(self, query_fea, query_pid, db_fea, db_pid, topk):
        fea = query_fea.reshape((1, query_fea.shape[0], query_fea.shape[1], query_fea.shape[2]))
        fea = self.put_input_to_gpu(fea)
        fea = self.forward_layers(False, fea, self.shared[0:-1] + self.user[0:-2])
        fea = np.tile(tensor.to_numpy(fea), [db_fea.shape[0], 1])
        fea = self.forward_layers(False, [fea, db_fea], self.user[-2:])
        dist = np.sum((fea - db_fea) ** 2, axis=1)
        return self.match(dist.reshape((1, -1)), [query_pid], db_pid, topk)

    def retrieval(self, data, topk, candidate_path):
        with open(os.path.join(candidate_path), 'r') as fd:
            sorted_idx, db_fea, db_pid = pickle.load(fd)
        data.start(1, 0)
        bar = trange(data.num_batches, desc='Query Image')
        new_target = np.empty(sorted_idx.shape, dtype=np.bool)
        new_sorted_idx = np.empty(sorted_idx.shape, dtype=np.int)
        for i in bar:
            img, pid = data.next()
            for j in range(self.batchsize):
                qid = i * self.batchsize + j
                candidate = sorted_idx[qid]
                t, s = self.rerank(img[j], pid[j], db_fea[candidate],
                                   db_pid[candidate], topk)
                new_target[qid] = t[0]
                new_sorted_idx[qid] = sorted_idx[qid][s[0]]
        data.stop()
        prec = compute_precision(new_target)
        return prec, None
