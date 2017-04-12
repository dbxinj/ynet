import random
import numpy as np
import time
import os
import sys
from multiprocessing import Pool, Queue
# , Value
from multiprocessing.sharedctypes import RawArray
from ctypes import c_float

from tqdm import trange

from singa import image_tool


def read_products(fpath, delimiter=',', seed=-1):
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


def stat_list(lists):
    min_len = 1000
    total_len = 0
    for l in range(lists):
        if min_len > len(l):
            min_len = len(l)
        total_len += len(l)
    return min_len, total_len, total_len/len(lists)


class DataIter(object):
    def __init__(self, img_dir, image_file, products, img_size=224, batchsize=32,
            capacity=10, delimiter=' ', nproc=1):
        self.batchsize = batchsize  # num of products to process for training, num of images for val/test
        self.capacity = capacity
        self.pool = None
        self.nproc = nproc
        self.task = Queue(capacity) # tasks for the worker <offset, size>
        self.result = Queue(capacity) # results from the worker <offset, size, product ids>

        # self.is_stop = Value(c_bool, False)
        self.img_buf = None  # shared mem

        self.pname2id = {}  # product name to id (index)
        self.pid2streetids = [[]] * len(products)
        self.pid2shopids = [[]] * len(products)
        self.idx2pid = range(len(products))  # for shuffle
        self.img_path = []
        for id, rec in enumerate(products):
            pname2id[rec[0]] = id

        logging.info('Total num of products = %d' % len(products)
        self.img_dir = img_dir
        self.img_size = img_size
        with open(image_file, 'r') as fd:
            for line in fd.readlines():
                rec = line.strip('\n').split(delimiter)
                if os.path.exists(os.path.join(img_dir, rec[0])):
                    self.img_path.append(rec[0])
                    if rec[1] in self.pname2id:
                        pid = self.pname2id[rec[1]]
                        if rec[2] == '0':  # street
                            self.pid2streetids[pid].append(len(self.img_path) - 1)
                        else:
                            self.pid2shopids[pid].append(len(self.img_path) - 1)
        self.min_street_per_prod, self.num_street, avg_len = stat_list(self.pid2streetids)
        logging.info('Min stree imgs per product = %d, avg num per product = %d'
                % (self.min_street_per_prod, self.min_shop_per_prod))
        self.min_shop_per_prod, self.num_shop, avg_len = stat_list(self.pid2shopids)
        logging.info('Min shop imgs per product = %d, avg num per product = %d'
                % (self.min_shop_per_prod, avg_len))

    def clear_queue(self, q):
        while not q.empty():
            q.get()

    def start(self, nstreet, nshop, shuffle=True):
        ''' func to load_triples, load_pid2streetidss, load_pid2shopidss
            nstreet and nshop >= 0
        '''
        assert nstreet >=0 and nshop >= 0, 'must >= 0'
        self.clear_queue(self.task)
        self.clear_queue(self.result)
        if self.img_buf is None:
            size = self.batchsize * 3 * self.img_size * self.img_size * (nstreet + nshop)
            self.img_buf = RawArray(c_float, self.capacity * size)
            for i in range(self.capacity):
                self.task.put((i * size, size))
        if shuffle:
            random.shuffle(self.idx2pid)
        assert self.pool is None, 'pool exists'
        self.pool = Pool(self.nproc)
        if nstreet * nshop > 0:
            count = 0
            for pid in range(len(self.products)):
                if len(self.pid2street[pid]) > nstreet and self.pid2shopids[pid] > nshop:
                    count += 1
            self.num_batches = count // self.batchsize - self.nproc * 2
            self.pool.map_async(load_pair, zip(range(self.nproc), [nstreet] * self.nproc, [nshop] * self.nproc))
        elif nstreet == 0:
            self.num_batches = self.num_shop // self.batchsize - self.nproc * 2
            self.pool.map_async(load_shop, range(self.nproc))
        else:
            self.num_batches = self.num_street // self.batchsize - self.nproc * 2
            self.pool.map_async(load_street, range(self.nproc))

    def next(self):
        try:
            offset, count, pids = self.result.get()  # dequeue one mini-batch
            imgs = np.copy(np.frombuffer(self.img_buf, count=cout, offset=offset)).reshape((-1, 3, self.img_size, self.img_size))
            self.task.put((offset, count))
            return imgs, pids
        except Exception as e:
            print(e)
        except:
            self.stop()
            sys.exit(1)

    def stop(self):
        # self.is_stop.value = True
        self.pool.terminate()
        self.pool.join()
        self.pool = None

    def read_images(self, id_list, ret):
        for i, id in enumerate(id_list):
            ret[i, :]=self.read_image(os.path.join(self.img_dir, self.img_path[id]))

    def read_image(self, path, is_train=True):
        img = image_tool.load_img(path).resize((self.img_size, self.img_size))
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        return ary.transpose(2, 0, 1)

    def tag2vec(self, tags):
        vec = np.zeros((self.tag_dim,), dtype=np.float32)
        vec[tags] = 1.0
        return vec

    def get_product_range(self, proc):
        prod_per_proc = (len(self.products) // (self.nproc * self.batchsize)) * self.batchsize
        idx = proc * prod_per_proc
        if proc == self.nproc - 1:
            end = len(self.products)
        else:
            end = idx + prod_per_proc
        return idx, end

    def load_pair(self, meta):
        proc, nstreet, nshop = meta
        pidx, end = self.get_product_range(proc)
        nprod = 0
        street_img, shop_img = [], []
        street_pid, shop_pid = [], []
        offset, count = self.task.get()
        ary = np.frombuffer(self.img_buf, count=cout, offset=offset)
        while pidx < end:
            pid = self.idx2pid[pidx]
            if len(self.pid2streetids[pid]) >= nstreet and len(self.pid2shopids[pid]) >= nshop:
                street_img.extends(random.sample(self.pid2streetids[pid], nstreet))
                street_pid.extends([pid] * nstreet)
                shop_img.extends(random.sample(self.pid2shopids[pid], nshop))
                shop_pid.extends([pid] * nshop)
                nprod += 1
                pidx += 1
                if nprod % self.batchsize == 0:
                    self.read_images(street_img + shop_img, ary)
                    self.result.put((offset, count, street_pid + shop_pid))
                    street_img, shop_img = [], []
                    street_pid, shop_pid = [], []
                    offset, count = self.task.get()
                    ary = np.frombuffer(self.img_buf, count=cout, offset=offset)
        logging.info('finish load triples by proc = %d' % proc)

    def load_street(self, proc):
        self.load_single(self, proc, self.pid2streetids)

    def load_shop(self, proc):
        self.load_single(self, proc, self.pid2shopids)

    def load_single(self, proc, pid2imgids):
        pid, end = self.get_product_range(proc)

        imgs, pids = [], []
        offset, count = self.task.get()
        ary = np.frombuffer(self.img_buf, count=cout, offset=offset)
        batchsize = count / 3 / self.img_size / self.img_size

        cur_offset = 0
        while pid < end:
            n = math.min(batchsize - len(imgs), len(pid2imgids[pid]) - cur_offset)
            imgs.extends(pid2imgids[cur_offset:cur_offset+n])
            pids.extends([pid] * n)
            if cur_offset + n < len(pid2imgids[pid]):
                m += n
            else:
                m = 0
                pid += 1
            if len(imgs) == batchsize:
                self.read_images(imgs, ary)
                self.result.put((offset, count, pids))
                imgs, pids = [], []
                offset, count = self.task.get()
                ary = np.frombuffer(self.img_buf, count=cout, offset=offset)
                cur_offset = 0
        logging.info('Finish loading street/shop images')


def calc_mean_std_for_single(data, nstreet, nshop):
    rgb = np.zeros((3,), dtype=np.float32)
    count = 0
    data.start(nstreet, nshop)
    for i in trange(data.num_batches):
        imgs, _ = data.next()
        for i in range(3):
            rgb[i] += np.average(imgs[:, i, :, :])
        count += imgs.shape[0]
    rgb /= count
    data.stop()

    std = np.zeros((3,), dtype=np.float32)
    data.start(nstreet, nshop)
    for i in trange(data.num_batches):
        img, _ = data.next()
        for i in range(3):
            d =  rgb[i] - np.average(img[:, i, :, :])
            std += d * d
    std = np.sqrt(qstd / count)
    return mean, std


def calc_mean_std(img_dir, data_dir):
    products = read_products(os.path.join(data_dir, 'product.txt'))
    data = DataIter(img_dir, os.path.join(data_dir, 'image.txt'), products[0: int(0.1 * len(products))])
    qrgb, qstd = calc_mean_std_for_single(data, 1, 0)
    nrgb, nstd = calc_mean_std_for_single(data, 0, 1)
    ary = np.array([qrgb, qstd, nrgb, nstd], dtype=np.float32)
    np.save(os.path.join(data_dir, 'mean-std'), ary)


def benchmark(img_dir, data_dir):
    t = time.time()
    with open(os.path.join(data_dir, 'image.txt'), 'r') as fd:
        img_path = fd.readlines()[0:50]
    for i in trange(50):
        data.read_image(os.path.join(img_dir, img_path[i]))
    print('time per image = %f' % ((time.time() - t)/50))

    products = read_products(os.path.join(data_dir, 'product.txt'))[0:200]
    data = DataIter(img_dir, os.path.join(data_dir, 'image.txt'), products)
    data.start(1, 1)
    t = time.time()
    for i in range(data.num_batches):
        data.next()
    print('time per batch = %f' % ((time.time() - t)/data.num_batches))



if __name__ == '__main__':
    img_dir = '/data/jixin/darn_dataset'
    benchmark(img_dir, 'data/darn')
    '''
    sample('./data/deepfashion/')
    meta = calc_mean_std(img_dir, './data/deepfashion')
    np.save('./data/deepfashion/meta', meta)
    '''
