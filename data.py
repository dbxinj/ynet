import random
import numpy as np
import time
import os
import sys
from multiprocessing import Process, Queue # , Value
from multiprocessing.sharedctypes import RawArray
from ctypes import c_float
import logging
from tqdm import trange

from singa import image_tool

logger = logging.getLogger(__name__)

FLOAT_WIDTH = 4

def read_products(fpath, delimiter=' ', seed=-1):
    '''read the product.txt file to get a list of products: product ID, attributes'''
    products = []
    with open(fpath, 'r') as fd:
        for line in fd.readlines():
            rec = line.strip('\n').split(delimiter)
            products.append(rec)
    if seed > -1:
        random.seed(seed)
        random.shuffle(products)
    return products

def filter_products(img_dir, img_file, products, nuser=1, nshop=1, delimiter=' '):
    user = np.zeros((len(products),), dtype=int)
    shop = np.zeros((len(products),), dtype=int)
    pname2id = {}  # product name to id (index)
    for id, rec in enumerate(products):
        pname2id[rec[0]] = id
    logger.info('Doing filtering with nuser=%d, nshop=%d' % (nuser, nshop))

    with open(img_file, 'r') as fd:
        for line in fd.readlines():
            rec = line.strip('\n').split(delimiter)
            if os.path.exists(os.path.join(img_dir, rec[0])):
                if rec[1] in pname2id:
                    pid = pname2id[rec[1]]
                    if rec[2] == '0':  # user
                        user[pid] += 1
                    else:
                        shop[pid] += 1
    idx1 = user >= nuser
    idx2 = shop >= nshop
    idx = np.squeeze(np.argwhere(idx1 * idx2 > 0))
    logger.info('Num of products before and after filtering: %d vs %d' % (len(products), len(idx)))
    return [products[i] for i in idx]


def stat_list(lists):
    min_len = 1000
    total_len = 0
    for l in lists:
        if min_len > len(l):
            min_len = len(l)
        total_len += len(l)
    return min_len, total_len, total_len/len(lists)


class DataIter(object):
    def __init__(self, img_dir, image_file, products, img_size=224, batchsize=32,
            capacity=10, delimiter=' ', nproc=1, meanstd=None, ncategory=20, nattribute=0):
        self.ncategory = ncategory
        self.nattribute = nattribute
        self.batchsize = batchsize  # num of products to process for training, num of images for val/test
        self.capacity = capacity
        self.proc = []
        self.nproc = nproc
        self.task = Queue(capacity) # tasks for the worker <offset, size>
        self.result = Queue(capacity) # results from the worker <offset, size, product ids>
        self.use_shared_mem = False
        self.products = products

        # self.is_stop = Value(c_bool, False)
        self.img_buf = None  # shared mem
        self.img_buf_size = 0

        self.idx2pid = range(len(products))  # for shuffle
        self.pname2id = {}  # product name to id (index)
        for id, rec in enumerate(products):
            self.pname2id[rec[0]] = id

        logger.info('Total num of products = %d' % len(products))
        self.img_dir = img_dir
        self.img_size = img_size
        self.img_path = []
        self.pid2cat = [-1] * len(products)
        self.pid2userids = [[] for _ in range(len(products))]
        self.pid2shopids = [[] for _ in range(len(products))]
        error_img = 0
        with open(image_file, 'r') as fd:
            for line in fd.readlines():
                rec = line.strip('\n').split(delimiter)
                if os.path.exists(os.path.join(img_dir, rec[0])):
                    if rec[1] in self.pname2id:
                        self.img_path.append(rec[0])
                        pid = self.pname2id[rec[1]]
                        if rec[2] == '0':  # user
                            self.pid2userids[pid].append(len(self.img_path) - 1)
                        else:
                            self.pid2shopids[pid].append(len(self.img_path) - 1)
                        cat = int(rec[3])
                        if self.pid2cat[pid] != -1 and self.pid2cat[pid] != cat:
                            error_img += 1
                        # assert self.pid2cat[pid] == -1 or self.pid2cat[pid] == cat, \
                        #    'img %s, category is not consistent; was set to %d' % (rec[0], self.pid2cat[pid])
                        self.pid2cat[pid] = cat
        logger.info('# of error imgs (whose product have multiple categories) = %d' % error_img)

        self.min_user_per_prod, self.num_user, avg_num = stat_list(self.pid2userids)
        logger.info('min street imgs per product = %d, avg = %d, total imgs = %d'
                % (self.min_user_per_prod, avg_num, self.num_user))
        self.min_shop_per_prod, self.num_shop, avg_num = stat_list(self.pid2shopids)
        logger.info('min shop imgs per product = %d, avg = %d, total imgs = %d'
                % (self.min_shop_per_prod, avg_num, self.num_shop))

        self.userid_pid = None
        self.shopid_pid = None
        if meanstd is None:
            self.user_meanstd = None
            self.shop_meanstd = None
        else:
            self.user_meanstd = meanstd[0:2]
            self.shop_meanstd = meanstd[2:]

    def clear_queue(self, q):
        while not q.empty():
            q.get()

    def prepare_list(self, pid2imgids):
        imgid_pid = []
        for pid, imgids in enumerate(pid2imgids):
            for imgid in imgids:
                imgid_pid.append((imgid, pid))
        return imgid_pid

    def start(self, nuser, nshop, shuffle=True):
        ''' func to load_triples, load_pid2useridss, load_pid2shopidss
            nuser and nshop >= 0
        '''
        assert nuser >=0 and nshop >= 0, 'must >= 0'
        self.clear_queue(self.result)
        if self.use_shared_mem:
            self.clear_queue(self.task)
            size = self.batchsize * 3 * self.img_size * self.img_size * (nuser + nshop)
            if self.img_buf_size < self.capacity * size:
                self.img_buf_size = self.capacity * size
                self.img_buf = RawArray(c_float, self.img_buf_size)
            for i in range(self.capacity):
                self.task.put((i * size * FLOAT_WIDTH, size))
        if shuffle:
            random.shuffle(self.idx2pid)
        if nuser * nshop > 0:
            self.num_batches = len(self.pname2id) // self.batchsize
            for i in range(self.nproc):
                self.proc.append(Process(target=self.load_pair, args=(i, nuser, nshop)))
                self.proc[-1].start()

        elif nuser == 0:
            if self.shopid_pid is None:
                self.shopid_pid = self.prepare_list(self.pid2shopids)
            self.num_batches = len(self.shopid_pid) // self.batchsize
            for i in range(self.nproc):
                self.proc.append(Process(target=self.load_shop, args=(i,)))
                self.proc[-1].start()
        else:
            if self.userid_pid is None:
                self.userid_pid = self.prepare_list(self.pid2userids)
            self.num_batches = len(self.userid_pid) // self.batchsize
            for i in range(self.nproc):
                self.proc.append(Process(target=self.load_user, args=(i,)))
                self.proc[-1].start()

    def next(self):
        try:
            if self.use_shared_mem:
                offset, count, pids = self.result.get()  # dequeue one mini-batch
                imgs = np.copy(np.frombuffer(self.img_buf, count=count, offset=offset)).reshape((-1, 3, self.img_size, self.img_size))
                self.task.put((offset, count))
            else:
                imgs, pids = self.result.get()
            return imgs, pids
        except Exception as e:
            print(e)
        except:
            self.stop()
            sys.exit(1)

    def stop(self):
        # self.is_stop.value = True
        for proc in self.proc:
            proc.terminate()

    def read_images(self, id_list, ret):
        for i, id in enumerate(id_list):
            # print self.img_path[id]
            ret[i, :]=self.read_image(os.path.join(self.img_dir, self.img_path[id]))

    def read_image(self, path, is_train=True):
        img = image_tool.load_img(path).resize((self.img_size, self.img_size))
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        return ary.transpose(2, 0, 1)

    def tag2vec(self, pids):
        l = self.ncategory + self.nattribute
        vec = np.zeros((len(pids), l), dtype=np.float32)
        for i, pid in enumerate(pids):
            if self.ncategory > 0:
                vec[i, self.pid2cat[pid]] = 1
            if self.nattribute > 0:
                for t in self.products[pid][1:]:
                    vec[i, self.ncategory + int(t)] = 1
        return vec

    def get_batch_range(self, nbatches, proc):
        nbatch_per_proc = nbatches // self.nproc
        start = proc * nbatch_per_proc
        end = start + nbatch_per_proc
        if proc == self.nproc - 1:
            end += nbatches % self.nproc
        return start, end

    def load_pair(self, proc, nuser, nshop):
        bstart, bend = self.get_batch_range(len(self.pname2id)//self.batchsize, proc)
        for b in range(bstart, bend):
            user_img, shop_img = [], []
            user_pid, shop_pid = [], []
            if self.use_shared_mem:
                offset, count = self.task.get()
                ary = np.frombuffer(self.img_buf, count=count, offset=offset).reshape((-1, 3, self.img_size, self.img_size))
                batch_vol = self.batchsize * (nuser + nshop) * 3 * self.img_size * self.img_size
                assert batch_vol == count, 'buf size mis-match batch vol = %d, buf count = %d' % (batch_vol, count)
            else:
                ary = np.empty((self.batchsize * (nuser + nshop), 3, self.img_size, self.img_size), dtype=np.float32)
            for idx in range(b * self.batchsize, (b + 1) * self.batchsize):
                pid = self.idx2pid[idx]
                user_img.extend(random.sample(self.pid2userids[pid], nuser))
                user_pid.extend([pid] * nuser)
                shop_img.extend(random.sample(self.pid2shopids[pid], nshop))
                shop_pid.extend([pid] * nshop)
            self.read_images(user_img + shop_img, ary)
            # normalize
            if self.user_meanstd is not None:
                ary[0:self.batchsize*nuser] -= self.user_meanstd[0][np.newaxis, :, np.newaxis, np.newaxis]
                ary[0:self.batchsize*nuser] /= self.user_meanstd[1][np.newaxis, :, np.newaxis, np.newaxis]
                ary[self.batchsize*nuser:] -= self.shop_meanstd[0][np.newaxis, :, np.newaxis, np.newaxis]
                ary[self.batchsize*nuser:] /= self.shop_meanstd[1][np.newaxis, :, np.newaxis, np.newaxis]
            if self.use_shared_mem:
                self.result.put((offset, count, user_pid + shop_pid))
            else:
                self.result.put((ary, user_pid + shop_pid))
        # logger.info('finish load triples by proc = %d' % proc)

    def load_user(self, proc):
        self.load_single(proc, self.userid_pid, self.user_meanstd)
        # logger.info('Finish loading user images')

    def load_shop(self, proc):
        self.load_single(proc, self.shopid_pid, self.shop_meanstd)
        # logger.info('Finish loading shop images')

    def load_single(self, proc, imgid_pid, meanstd):
        bstart, bend = self.get_batch_range(len(imgid_pid) // self.batchsize, proc)
        for b in range(bstart, bend):
            if self.use_shared_mem:
                offset, count = self.task.get()
                batch_vol = self.batchsize * 3 * self.img_size * self.img_size
                assert  batch_vol == count, 'buffer size mismatch batchsize %d vs task queue size %d' % (batch_vol, count)
                ary = np.frombuffer(self.img_buf, count=count, offset=offset).reshape((-1, 3, self.img_size, self.img_size))
            else:
                ary = np.empty((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
            imgs = [x[0] for x in imgid_pid[b * self.batchsize: (b + 1) * self.batchsize]]
            pids = [x[1] for x in imgid_pid[b * self.batchsize: (b + 1) * self.batchsize]]
            self.read_images(imgs, ary)
            # normalize
            if meanstd is not None:
                # print 'normalize user/db image'
                ary -= meanstd[0][np.newaxis, :, np.newaxis, np.newaxis]
                ary /= meanstd[1][np.newaxis, :, np.newaxis, np.newaxis]
            if self.use_shared_mem:
                self.result.put((offset, count, pids))
            else:
                self.result.put((ary, pids))


def calc_mean_std_for_single(data, nuser, nshop):
    count = 0
    rgb = np.zeros((3,), dtype=np.float32)
    data.start(nuser, nshop)
    for i in trange(data.num_batches):
        imgs, _ = data.next()
        for i in range(3):
            rgb[i] += np.average(imgs[:, i, :, :])
        count += imgs.shape[0]
    rgb /= count
    data.stop()

    std = np.zeros((3,), dtype=np.float32)
    data.start(nuser, nshop)
    for i in trange(data.num_batches):
        img, _ = data.next()
        for i in range(3):
            d =  rgb[i] - np.average(img[:, i, :, :])
            std[i] += d * d
    std = np.sqrt(std / count)
    return rgb, std


def calc_mean_std(img_dir, data_dir, ratio=0.5):
    products = read_products(os.path.join(data_dir, 'product.txt'))
    data = DataIter(img_dir, os.path.join(data_dir, 'image.txt'), products[0: int(ratio * len(products))])
    qrgb, qstd = calc_mean_std_for_single(data, 1, 0)
    nrgb, nstd = calc_mean_std_for_single(data, 0, 1)
    ary = np.array([qrgb, qstd, nrgb, nstd], dtype=np.float32)
    return ary


def benchmark(img_dir, data_dir):
    t = time.time()
    with open(os.path.join(data_dir, 'image.txt'), 'r') as fd:
        img_path = fd.readlines()[0:50]
    for i in trange(50):
        img = image_tool.load_img(os.path.join(img_dir, img_path[i].split(' ')[0])).resize((224, 224))
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        ary.transpose(2, 0, 1)
    print('time per image = %f' % ((time.time() - t)/50))

    products = read_products(os.path.join(data_dir, 'product.txt'))[0:1000]
    products = filter_products(img_dir, os.path.join(data_dir, 'image.txt'), products)
    meanstd = np.load(os.path.join(data_dir, 'mean-std.npy'))
    data = DataIter(img_dir, os.path.join(data_dir, 'image.txt'), products, meanstd=meanstd)
    data.start(1, 1)
    t = time.time()
    for i in range(data.num_batches):
        data.next()
    print('time per batch = %f, num batch = %d' % ((time.time() - t)/data.num_batches, data.num_batches))


def save_list(fpath, imgid_pid, imglist):
    with open(fpath, 'w') as fd:
        for imgid, pid in imgid_pid:
            fd.write('%s %s\n' % (imglist[imgid], pid))
        fd.flush()


def extract_testdata(data_dir, img_dir):
    img_list_file = os.path.join(data_dir, 'image.txt')
    product_list_file = os.path.join(data_dir, 'product.txt')
    products = read_products(product_list_file) # [0:5000]
    num_products = len(products)
    num_train_products = int(num_products * 0.8)
    num_val_products = (num_products - num_train_products) // 2
    train_data, val_data, test_data = None, None, None
    test_products = products[num_train_products + num_val_products:]
    test_data = DataIter(img_dir, img_list_file, test_products)

    imgid_pid = test_data.prepare_list(test_data.pid2userids)
    save_list(os.path.join(data_dir, 'test_list.txt'), imgid_pid, test_data.img_path)
    imgid_pid = test_data.prepare_list(test_data.pid2shopids)
    save_list(os.path.join(data_dir, 'db_list.txt'), imgid_pid, test_data.img_path)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format='%(message)s', level=logging.INFO)
    extract_testdata('./data/darn', '../darn/')
    # img_dir = '/home/wangyan/deepfashion/img'
    # data_dir = 'data/deepfashion/'
    # benchmark(img_dir, data_dir)
    # ary = calc_mean_std(img_dir, data_dir, 0.5)
    # np.save(os.path.join(data_dir, 'mean-std'), ary)
