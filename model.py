from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D,\
        Split, Merge, Flatten, Dense, BatchNormalization, Softmax
from singa import net as ffnet
from singa import initializer
from singa import layer


class CANet():
    '''Context-depedent attention modeling'''
    def __init__(self, name):
        self.name = name

    def create_net(self):

    def init_params(weight_path=None):
    if weight_path == None:
        for pname, pval in zip(net.param_names(), net.param_values()):
            print pname, pval.shape
            if 'conv' in pname and len(pval.shape) > 1:
                initializer.gaussian(pval, 0, pval.shape[1])
            elif 'dense' in pname:
                if len(pval.shape) > 1:
                    initializer.gaussian(pval, 0, pval.shape[0])
                else:
                    pval.set_value(0)
            # init params from batch norm layer
            elif 'mean' in pname or 'beta' in pname:
                pval.set_value(0)
            elif 'var' in pname:
                pval.set_value(1)
            elif 'gamma' in pname:
                initializer.uniform(pval, 0, 1)
    else:
        net.load(weight_path, use_pickle = 'pickle' in weight_path)


    def extract_query_feature(self, x):
        pass

    def extract_db_feature(self, x, tag):
        pass

    def bprop(self, qimg, pimg, nimg, ptag, ntag):
        pass

    def evaluate(self, qimg, pimg, nimg, ptag, ntag):
        pass

    def save(self, fpath):
        pass

    def load(self, fpath):
        pass

class CANIN(CANet):
    def add_conv(net, name, nb_filter, size, stride, pad, sample_size):
        if sample_size != None:
            net.add(Conv2D('%s-3x3' % name, nb_filter, size, stride, input_sample_shape=sample_size))

            net.add(Conv2D('%s-3x3' % name, nb_filter, size, stride, input_sample_shape=sample_size))
        net.add(Activation('c1-3x3-relu'))
        net.add(Conv2D('c1-1x1-1', 96, 1, 1))
        net.add(Activation('c1-1x1-1-relu'))
        net.add(Conv2D('c1-1x1-2', 96, 1, 1))
        net.add(Activation('c1-1x1-2-relu'))
        net.add(MaxPooling2D('pool1', 3, 2))


    def create_net(self):
        net = ffnet.FeedForwardNet()

        net.add(Conv2D('c1-3x3', 96, 7, 2, input_sample_shape=(3, 227, 227)))
        net.add(Activation('c1-3x3-relu'))
        net.add(Conv2D('c1-1x1-1', 96, 1, 1))
        net.add(Activation('c1-1x1-1-relu'))
        net.add(Conv2D('c1-1x1-2', 96, 1, 1))
        net.add(Activation('c1-1x1-2-relu'))
        net.add(MaxPooling2D('pool1', 3, 2))






        net.add(BatchNormalization('input-bn'))

