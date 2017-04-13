from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D, Flatten, Slice
from singa import initializer
from singa import layer
from singa import loss
from singa import tensor
import cPickle as pickle
import logging

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


def l1(ary):
    return np.average(ary)


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
    return (np.array([l1(loss), l1(d1), l1(d2)]), grads)


class TripletLoss(loss.Loss):
    def __init__(self, margin=0.1, nshift=1, nuser=1, nshop=1):
        self.margin = margin
        self.nshift = nshift
        self.nuser = nuser
        self.nshop = nshop
        self.guser = None
        self.gshop = None
        self.dev = None

    def forward(self, is_train, user_fea, shop_fea, pids):
        ufea = tensor.to_numpy(user_fea)
        sfea = tensor.to_numpy(shop_fea)
        if is_train:
            self.dev = user_fea.device
            self.guser = np.zeros(ufea.shape, dtype=np.float32)
            self.gshop = np.zeros(sfea.shape, dtype=np.float32)
        ret = np.zeros((3,), dtype=float)
        for i in range(1, self.nshift+1):
            offset = i * self.nshop
            idx = range(offset, sfea.shape[0]) + range(0, offset)
            loss, grads = loss_bp(is_train, ufea, ufea, sfea, sfea[idx], self.margin)
            if is_train:
                self.guser += grads[0] + grads[1]
                self.gshop += grads[2]
                self.gshop[idx] += grads[3]
            ret += loss
        return ret/self.nshift

    def backward(self):
        tguser = tensor.from_numpy(self.guser/self.nshift)
        tgshop = tensor.from_numpy(self.gshop/self.nshift)
        tguser.to_device(self.dev)
        tgshop.to_device(self.dev)
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
        loss, dap, dan = 0, 0, 0
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
            loss = update_perf(loss, l[0])
            dap = update_perf(dap, l[1])
            dan = update_perf(dan, l[2])
            bar.set_postfix(train_loss=loss, dap=dap, dan=dan, load_time=t[0], bptime=t[1])
        data.stop()
        logger.info('Epoch %d, training loss = %f,  pos dist = %f, neg dist = %f' % (epoch, loss, dap, dan))

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
        loss, dap, dan = 0, 0, 0
        data.start(nuser, nshop)
        bar = trange(data.num_batches, desc='Epoch %d' % epoch)
        for b in bar:
            l, ap, an = self.forward(False, data.next())
            loss += l
            dap += ap
            dan += an
        data.stop()
        msg = 'Epoch %d, validation loss = %f, pos dist = %f, neg dist = %f' %\
              (epoch, loss / data.num_batches, dap / data.num_batches, dan / data.num_batches)
        print(msg)
        logger.info(msg)
        return loss / data.num_batches

    def retrieval(self, data, result_path, topk=100):
        query_fea, query_id = self.extract_query_feature(data)
        db_fea, db_id = self.extract_db_feature(data)
        prec, sorted_idx, target, topdist = self.match(query_fea, query_id, db_fea, db_id, topk)
        np.save('%s-dist' % result_path, topdist)
        np.save('%s-target' % result_path, target)
        np.savetxt('%s-precision.txt' % result_path, prec)
        return prec, sorted_idx

    def match(self, query_fea, query_id, db_fea, db_id, topk=100):
        t = time.time()
        dist=scipy.spatial.distance.cdist(query_fea, db_fea,'euclidean')
        print('distance computation time = %f' % (time.time() - t))
        sorted_idx=np.argsort(dist,axis=1)[:, 0:topk]
        topdist = np.empty(sorted_idx.shape, dtype=np.float32)
        for i in range(sorted_idx.shape[0]):
            topdist[i] = dist[i, sorted_idx[i]]

        target = np.empty((len(query_id), topk), dtype=np.bool)
        for i in range(len(query_id)):
            for j in range(topk):
                target[i,j] = db_id[sorted_idx[i, j]] == query_id[i]
        prec = compute_precision(target)
        return prec, sorted_idx, target, topdist

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
#        shop.append(Slice('shop-slice', 0, [batchsize], input_sample_shape=shop[-1].get_output_sample_shape()))

        return shared, user, shop

    def forward(self, is_train, data, to_loss=True):
        imgs, pids = data
        x = tensor.from_numpy(imgs)
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
        for lyr in self.user:
            a = lyr.forward(is_train, a)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, a.l1()))
        for lyr in self.shop:
            b = lyr.forward(is_train, b)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, b.l1()))
        if to_loss:
            return self.loss.forward(is_train, a, b, pids)
        else:
            return a, b, pids

    def backward(self):
        if self.debug:
            print '------------backward------------'
        param_grads = []
        duser, dshop = self.loss.backward()
        for lyr in self.shop[::-1]:
            dshop, dp = lyr.backward(True, dshop)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, dshop.l1()))
            if dp is not None:
                param_grads.extend(dp[::-1])

        for lyr in self.user[::-1]:
            duser, dp = lyr.backward(True, duser)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, duser.l1()))
            if dp is not None:
                param_grads.extend(dp[::-1])

        d = [duser, dshop]
        for lyr in self.shared[::-1]:
            d, dp = lyr.backward(True, d)
            if self.debug:
                print('%30s = %2.8f' % (lyr.name, d.l1()))
            if dp is not None:
                param_grads.extend(dp[::-1])
        return param_grads[::-1]

    def bprop(self, data):
        t1 = time.time()
        dat = data.next()
        t2 = time.time()
        loss = self.forward(True, dat)
        ret = self.backward()
        t3 = time.time()
        return ret, loss, [t2-t1, t3-t2]

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

    def extract_query_feature_on_batch(self, data):
        img, pid = data.next()
        fea = self.forward_layers(img, self.shared[0:-1] + self.user)
        return fea, pid

    def extract_db_feature_on_batch(self, data):
        img, pid = data.next()
        fea = self.forward_layers(img, self.shared[0:-1] + self.shop)
        return fea, pid
