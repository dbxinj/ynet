from util import *

from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D, Flatten, Slice
from singa import initializer
from singa import layer
from singa import loss
from singa import tensor

import logging
import os
import cPickle as pickle
import numpy as np
import scipy.spatial
from tqdm import trange
import time

logger = logging.getLogger(__name__)


class YNet(object):
    '''Base network'''
    def __init__(self, name, loss, dev, img_size, batchsize=32, ntag=0,
            freeze_shared=False, freeze_shop=False, freeze_user=False,
            nshift=1, debug=False, nuser=1, nshop=1):
        self.debug = debug
        self.name = name
        self.loss = loss
        self.device = dev
        self.batchsize = batchsize
        self.img_size = img_size
        self.nuser = nuser
        self.nshop = nshop
        self.freeze_shared = freeze_shared
        self.freeze_user = freeze_user
        self.freeze_shop = freeze_shop
        self.nshift = nshift
        self.ntags = ntags

        self.shared, self.user, self.shop = self.create_net(name, img_size, batchsize)
        self.layers = self.shared + self.user + self.shop
        if debug:
            for lyr in self.layers:
                print lyr.name, lyr.get_output_sample_shape()
        self.to_device(dev)

    def create_net(self, name, img_size, batchsize):
        pass

    def forward(self, is_train, data):
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

    def collect_layers(self, flag):
        layers = []
        if (not self.freeze_shared) or flag:
            layers.extend(self.shared)
        if (not self.freeze_user) or flag:
            layers.extend(self.user)
        if (not self.freeze_shop) or flag:
            layers.extend(self.shop)
        return layers

    def param_names(self, flag=False):
        pname = []
        for lyr in self.collect_layers(flag):
            pname.extend(lyr.param_names())
        return pname

    def param_values(self, flag=False):
        pvals = []
        for lyr in self.collect_layers(flag):
            pvals.extend(lyr.param_values())
        return pvals

    def init_params(self, weight_path=None):
        if weight_path is None:
            assert not (self.freeze_shared or self.freeze_shop or self.freeze_user), \
                    'no checkpoint file is give, must train all branches'
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

    def train_on_epoch(self, epoch, data, opt, lr):
        loss = None
        data.start(self.nuser, self.nshop)
        bar = trange(data.num_batches, desc='Epoch %d' % epoch)
        for b in bar:
            grads, l, t = self.bprop(data)
            if self.debug:
                print('-------------params---------------')
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
        data.start(self.nuser, 0)
        bar = trange(data.num_batches, desc='Query Image')
        query_fea, query_pid = None, []
        for i in bar:
            fea, pid = self.extract_query_feature_on_batch(data)
            query_pid.extend(pid)
            if query_fea is None:
                query_fea = np.empty((data.num_batches * self.batchsize, fea.shape[1]), dtype=np.float32)
            query_fea[i*fea.shape[0]:(i+1)*fea.shape[0]] = fea
        data.stop()
        if self.debug:
            print 'query shape', query_fea.shape
        return query_fea, query_pid

    def extract_db_feature(self, data):
        '''x for shop images'''
        data.start(0, self.nshop)
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

    def evaluate_on_epoch(self, epoch, data):
        loss = None
        data.start(self.nuser, self.nshop)
        bar = trange(data.num_batches, desc='Epoch %d' % epoch)
        for b in bar:
            l, _, _ = self.forward(False, data)
            if loss is None:
                loss = np.zeros(l.shape)
            loss += l
            if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
                break

        loss /= data.num_batches
        data.stop()
        return loss

    def retrieval(self, data, topk=100, candiate_path=None):
        query_fea, query_id = self.extract_query_feature(data)
        db_fea, db_id = self.extract_db_feature(data)
        dist = scipy.spatial.distance.cdist(query_fea, db_fea,'euclidean')
        target, sorted_idx = self.match(dist, query_id, db_id, topk)
        prec = compute_precision(target)
        return prec, (sorted_idx, db_fea, db_id)

    def match(self, dist, query_id, db_id, topk=100):
        sorted_idx=np.argsort(dist,axis=1)[:, 0:topk]
        '''
        topdist = np.empty(sorted_idx.shape, dtype=np.float32)
        for i in range(sorted_idx.shape[0]):
            topdist[i] = dist[i, sorted_idx[i]]
        '''
        target = np.empty((len(query_id), topk), dtype=np.bool)
        for i in range(len(query_id)):
            for j in range(topk):
                target[i,j] = db_id[sorted_idx[i, j]] == query_id[i]
        return target, sorted_idx


class L2Norm(layer.Layer):
    '''for singa Tensor'''
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
    def __init__(self, margin=0.1, nshift=1):
        self.margin = margin
        self.nshift = nshift
        self.guser = None
        self.gshop = None
        self.deva = None
        self.devb = None

    def forward(self, is_train, ufea, sfea, pids):
        if type(ufea) == tensor.Tensor:
            self.deva = ufea.device
            ufea = tensor.to_numpy(ufea)
        if type(sfea) == tensor.Tensor:
            self.devb = sfea.device
            sfea = tensor.to_numpy(sfea)
        if is_train:
            self.guser = np.zeros(ufea.shape, dtype=np.float32)
            self.gshop = np.zeros(sfea.shape, dtype=np.float32)
        ret = None
        for i in range(1, self.nshift+1):
            offset = i
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
        if self.deva is not None:
            guser = tensor.from_numpy(self.guser)
            guser.to_device(self.deva)
        else:
            guser = self.guser

        if self.devb is not None:
            gshop = tensor.from_numpy(self.gshop)
            gshop.to_device(self.devb)
        else:
            gshop = self.gshop
        guser /= self.nshift
        gshop /= self.nshift
        return guser, gshop


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
                show_debuginfo(lyr.name, x)
        return x

    def forward(self, is_train, data):
        t1 = time.time()
        imgs, pids = data.next()
        t2 = time.time()
        x = self.put_input_to_gpu(imgs)
        a, b = self.forward_layers(is_train and (not self.freeze_shared), x, self.shared)
        a = self.forward_layers(is_train and (not self.freeze_user), a, self.user)
        b = self.forward_layers(is_train and (not self.freeze_shop), b, self.shop)
        loss = self.loss.forward(is_train, a, b, pids)
        return loss, t2 - t1, time.time() -t2

    def backward_layers(self, dy, layers, dparam):
        for lyr in layers:
            dy, dp = lyr.backward(True, dy)
            if self.debug:
                show_debuginfo(lyr.name, dy)
            if dp is not None:
                dparam.extend(dp[::-1])
        return dy

    def backward(self):
        if self.debug:
            print '------------backward------------'
        param_grads = []
        duser, dshop = self.loss.backward()
        dshop = self.backward_layers(dshop, self.shop[::-1], param_grads)
        duser = self.backward_layers(duser, self.user[::-1], param_grads)
        if not self.freeze_shared:
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
