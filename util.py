import numpy as np
from singa import tensor
from numpy.core.umath_tests import inner1d

def update_perf(his, cur, a=0.8):
    '''Accumulate the performance by considering history and current values.'''
    return his * a + cur * (1 - a)


def compute_precision(target):
    assert target.shape[1] >= 100, 'the db has less than 100 records'
    points = range(9, 101, 10)
    ret = target.cumsum(axis=1) > 0
    prec = np.average(ret[:, points], axis=0)
    return prec


def tuple2str(t):
    return ' '.join([str(x) for x in t])

def show_debuginfo(name, x):
    if type(x) == tensor.Tensor:
        print('%30s = %2.8f, (%s)' % (name, x.l1(), tuple2str(x.shape)))
    elif type(x) == np.ndarray:
        print('%30s = %2.8f, (%s)' % (name, np.average(np.abs(x)), tuple2str(x.shape)))
    elif type(x[0]) == tensor.Tensor:
        print('%30s = %2.8f, %2.8f, (%s)' % (name, x[0].l1(), x[1].l1(), tuple2str(x[0].shape)))
    else:
        print('%30s = %2.8f, %2.8f, (%s)' % (name, np.average(np.abs(x[0])), np.average(np.abs(x[1])), tuple2str(x[0].shape)))


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
