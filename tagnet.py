import ynet
from util import *

from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D, Flatten, Slice, LRN
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


class L2Norm(ynet.L2Norm):
    def forward(self, is_train, x):
        norm = np.sqrt(np.sum(x**2, axis=1) + self.epsilon)
        self.y = x / norm[:, np.newaxis]
        if is_train:
            self.norm = norm
        return self.y

    def backward(self, is_train, dy):
        # (b' - b * k) /norm, k = sum(dy * y)
        k = np.sum(dy * self.y, axis=1)
        dx = dy - self.y * k[:, np.newaxis]
        dx /= self.norm[:, np.newaxis]
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
        #initializer.gaussian(self.W, input_sample_shape[0], num_output)
        self.W.gaussian(0, 0.008)

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


class TagAttention(layer.Layer):
    def __init__(self, name, input_sample_shape):
        super(TagAttention, self).__init__(name)
        self.c, self.h, self.w = input_sample_shape[0]
        l = self.h * self.w
        self.embed = TagEmbedding('%s_embed' % name, self.c, input_sample_shape[1])
        self.attention = ProductAttention('%s_attention' % name, [input_sample_shape[0], (self.c,)])
        self.softmax = Softmax('%s_softmax' % name, (l,))
        self.agg = Aggregation('%s_agg' % name, [input_sample_shape[0], (l,)])
        self.dev = None

    def get_output_sample_shape(self):
        return (self.c, )

    def param_names(self):
        return self.embed.param_names()

    def param_values(self):
        return self.embed.param_values()

    def display(self, name, val):
        if ynet.debug:
            print('%30s = %2.8f' % (name, np.average(np.abs(val))))

    def forward(self, is_train, x, output_weight=False):
        if type(x[0]) == tensor.Tensor:
            self.dev = x[0].device
            img = tensor.to_numpy(x[0])
        else:
            img = x[0]
        t = self.embed.forward(is_train, x[1])
        if ynet.debug:
            show_debuginfo(self.embed.name, t)
        w = self.attention.forward(is_train, [img, t])
        if ynet.debug:
            show_debuginfo(self.attention.name, w)
        w = self.softmax.forward(is_train, w)
        if ynet.debug:
            show_debuginfo(self.softmax.name, w)
        y = self.agg.forward(is_train, [img, w])
        if ynet.debug:
            show_debuginfo(self.agg.name, y)
        if output_weight:
            return y, w
        else:
            return y

    def backward(self, is_train, dy):
        [dx1, dw], _ = self.agg.backward(is_train, dy)
        if ynet.debug:
            show_debuginfo(self.agg.name, dx1)
        dw, _ = self.softmax.backward(is_train, dw)
        if ynet.debug:
            show_debuginfo(self.softmax.name, dw)
        [dx2, dt], _ = self.attention.backward(is_train, dw)
        if ynet.debug:
            show_debuginfo(self.attention.name, dx2)
        _, dW = self.embed.backward(is_train, dt)
        dx = np.reshape(dx1 + dx2, (dx1.shape[0], self.c, self.h, self.w))
        if self.dev is not None:
            dx = tensor.from_numpy(dx)
            dx.to_device(self.dev)
        return dx, dW


class TagNIN(ynet.YNIN):
    def create_net(self, name, img_size, batchsize=32):
        assert self.ntag > 0, 'no tags for tag nin'
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
        user.append(ynet.L2Norm('street-l2', input_sample_shape=user[-1].get_output_sample_shape()))

        shop = []
        self.add_conv(shop, 'shop-conv4', [1024, 1024, 1000], 3, 1, 1, sample_shape=slice_layer.get_output_sample_shape()[1])
        shop.append(TagAttention('shop-tag',
            input_sample_shape=[shop[-1].get_output_sample_shape(), (self.ntag, )]))
        shop.append(L2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))
        return shared, user, shop

    def forward(self, is_train, data):
        t1 = time.time()
        imgs, pids = data.next()
        t2 = time.time()
        imgs = self.put_input_to_gpu(imgs)
        a, b = self.forward_layers(is_train and (not self.freeze_shared), imgs, self.shared)
        a = self.forward_layers(is_train and (not self.freeze_user), a, self.user)
        b = self.forward_layers(is_train and (not self.freeze_shop), b, self.shop[0:-2])
        b = self.shop[-2].forward(is_train, [b, data.tag2vec(pids[a.shape[0]:])])
        b = self.forward_layers(is_train and (not self.freeze_shop), b, self.shop[-1:])
        loss = self.loss.forward(is_train, a, b, pids)
        return loss, t2 - t1, time.time() - t2

    def extract_db_feature_on_batch(self, data):
        img, pid = data.next()
        img = self.put_input_to_gpu(img)
        fea = self.forward_layers(False, img, self.shared[0:-1] + self.shop[0:-2])
        fea = self.shop[-2].forward(False, [fea, data.tag2vec(pid)])
        return fea, pid

class TagVGG(TagNIN):
    def create_net(self, name, img_size, batchsize=32):
        assert self.ntag > 0, 'no tags for tag nin'
        shared = []

        shared.append(Conv2D('conv1-3x3', 96, 7, 2, pad=1,  input_sample_shape=(3, img_size, img_size)))
        shared.append(Activation('conv1-relu', input_sample_shape=shared[-1].get_output_sample_shape()))
        shared.append(LRN('conv1-norm', size=5, alpha=5e-4, beta=0.75, k=2, input_sample_shape=shared[-1].get_output_sample_shape()))
        shared.append(MaxPooling2D('pool1', 3, 3, pad=0, input_sample_shape=shared[-1].get_output_sample_shape()))

        shared.append(Conv2D('conv2', 256, 5, 1, cudnn_prefer='limited_workspace', workspace_byte_limit=1000, pad=1, input_sample_shape=shared[-1].get_output_sample_shape()))
        shared.append(Activation('conv2-relu', input_sample_shape=shared[-1].get_output_sample_shape()))
        shared.append(MaxPooling2D('pool2', 2, 2, pad=0, input_sample_shape=shared[-1].get_output_sample_shape()))

        shared.append(Conv2D('conv3', 512, 3, 1, cudnn_prefer='limited_workspace', workspace_byte_limit=1000, pad=1, input_sample_shape=shared[-1].get_output_sample_shape()))
        shared.append(Activation('conv3-relu', input_sample_shape=shared[-1].get_output_sample_shape()))

        shared.append(Conv2D('conv4', 512, 3, 1, cudnn_prefer='limited_workspace', workspace_byte_limit=1500, pad=1, input_sample_shape=shared[-1].get_output_sample_shape()))
        shared.append(Activation('conv4-relu', input_sample_shape=shared[-1].get_output_sample_shape()))

        slice_layer = Slice('slice', 0, [batchsize*self.nuser], input_sample_shape=shared[-1].get_output_sample_shape())
        shared.append(slice_layer)

        user = []
        user.append(Conv2D('street-conv5', 512, 3, 1, cudnn_prefer='limited_workspace', workspace_byte_limit=1500, pad=1, input_sample_shape=shared[-1].get_output_sample_shape()[1]))
        user.append(Activation('street-conv5-relu', input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(Conv2D('street-conv6', 128, 3, 2, cudnn_prefer='limited_workspace', workspace_byte_limit=1500, pad=0, input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(Activation('street-conv6-relu', input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(AvgPooling2D('street-pool6', 8, 1, pad=0, input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(Flatten('street-flat', input_sample_shape=user[-1].get_output_sample_shape()))
        user.append(ynet.L2Norm('street-l2', input_sample_shape=user[-1].get_output_sample_shape()))

        shop = []
        shop.append(Conv2D('shop-conv5', 512, 3, 1, cudnn_prefer='limited_workspace', workspace_byte_limit=1500, pad=1, input_sample_shape=shared[-1].get_output_sample_shape()[1]))
        shop.append(Activation('shop-conv5-relu', input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(Conv2D('shop-conv6', 128, 3, 2, cudnn_prefer='limited_workspace', workspace_byte_limit=1500, pad=0, input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(Activation('shop-conv6-relu', input_sample_shape=shop[-1].get_output_sample_shape()))
        shop.append(TagAttention('shop-tag',
            input_sample_shape=[shop[-1].get_output_sample_shape(), (self.ntag, )]))
        shop.append(L2Norm('shop-l2', input_sample_shape=shop[-1].get_output_sample_shape()))

        return shared, user, shop
