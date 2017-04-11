import random
import numpy as np
import time
import os
import sys
from multiprocessing import Process, Queue
from tqdm import trange

from singa import image_tool

train_tool = image_tool.ImageTool()
validate_tool = image_tool.ImageTool()


def read_product_info(fpath, delimiter=',', seed=0):
    '''return the product name to index mapping and product records'''
    products = []
    with open(fpath, 'r') as fd:
        for line in fd.readlines():
            rec = line.strip('\n').split(delimiter)
            products.append(rec)
    random.seed(seed)
    random.shuffle(products)
    return products


class DataIter(object):
    def __init__(self, image_dir, image_file, products, img_size,
                 batchsize=32, capacity=50, delimiter=' ', nproc=2):
        self.batchsize = batchsize  # num of products to process
        self.image_dir = image_dir
        self.img_size = img_size
        self.capacity = capacity
        self.nproc = nproc
        self.proc = []  # for loading triplet
        self.queue = Queue(capacity)
        self.product2id = {}
        self.street_images = [[]] * len(products)
        self.shop_images = [[]] * len(products)
        for idx, rec in enumerate(products):
            product2id[rec[0]] = idx

        with open(image_file, 'r') as fd:
            for line in fd.readlines():
                rec = line.strip('\n').split(delimiter)
                if rec[1] in self.product2id:
                    pid = self.product2id[rec[1]]
                    if rec[2] == '0':  # street
                        self.street_image[pid].append(rec[0])
                    else:
                        self.shop_image[pid].append(rec[0])

        self.idx = range(len(self.products))

    def next(self):
        assert self.proc is not None, 'call start before next'
        try:
            while self.queue.empty():
                time.sleep(0.1)
            imgs, pids = self.queue.get()  # dequeue one mini-batch
            return imgs, self.products[pids]
        except Exception as e:
            print(e)
        except:
            self.stop()
            sys.exit(1)

    def start(self, func):
        # func to be load_triples, load_street_images, load_shop_images
        while not self.queue.empty():
            self.queue.get()
        for i in range(self.nproc):
            self.proc.append(Process(target=func, args=(i, self.nproc)))
            time.sleep(0.4)
            self.proc[-1].start()

    def stop(self):
        for proc in self.proc:
            time.sleep(0.1)
            proc.terminate()
        self.proc = []

    def read_image(self, path, is_train=True):
        img = image_tool.load_img(path).resize((self.img_size, self.img_size))
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        return ary.transpose(2, 0, 1)

    def tag2vec(self, tags):
        vec = np.zeros((self.tag_dim,), dtype=np.float32)
        vec[tags] = 1.0
        return vec

    def do_shuffle(self):
        random.shuffle(self.idx)

    def do_load(self, n_street=1, n_shop=1):
        count = 0
        if n_streep * n_shop == 0:
            img_list = []
            pid_list = []
            offset = 0
        else:
            street_img_list = []
            shop_img_list = []
            street_pid_list = []
            shop_pid_list = []
        idx = 0
        while not stop:
            if len(self.street_image[idx]) > n_street and len(self.shop_image[idx]) > n_shop:
                if n_street == -1 and n_shop == 0:  # scan street images
                    offset, idx = self.load(self.street_image[idx], img_list, pid_list, idx, offset)
                elif n_street == 0 and n_shop == -1:  # scan shop images
                    offset, idx = self.load(self.shop_image[idx], img_list, pid_list, idx, offset)
                else:
                    samples = random.shuffle(range(len(self.street_image[idx])))[0:n_street]
                    street_img_list.extends(self.street_image[idx][samples])
                    street_pid_list.extends([idx] * n_street)
                    samples = random.shuffle(range(len(self.shop_image[idx])))[0:n_shop]
                    shop_img_list.extends(self.shop_image[idx][samples])
                    shop_pid_list.extends([idx] * n_shop)
                    idx += 1
            if len(img_list) == self.batchsize:
                self.queue.put((self.read_images(img_list),pid_list))
            if len(street_pid_list) == self.batchsize * n_street:
                self.queue.put((self.read_images(street_img_list + shop_img_list), street_pid_list + shop_pid_list))
            if idx == len(self.products):
                break
       print('finish load triples')

    def load(self, from_list, to_list, pid_list, idx, offset):
        n = math.min(self.batchsize - len(to_list), len(from_list) - offset)
        to_list.extends(from_list[offset:offset+n])
        pid_list.extends([idx] * n)
        if offset + n < len(from_list):
            ofset += n
        else:
            offset = 0
            idx += 1
        return idx, offset

    def read_images(self, img_list):
        pass

class DARNDataIter(DataIter):
    def __init__(self, image_dir, pair_file, shop_file, img_size, batchsize=32, nproc=1):
        self.ntags_per_attr = [20, 56, 10, 25, 27, 16, 7, 12, 6]
        super(DARNDataIter, self).__init__(image_dir, pair_file, shop_file, self.ntags_per_attr, img_size=img_size, batchsize=32, nproc=nproc)


class FashionDataIter(DataIter):
    def __init__(self, image_dir, pair_file, shop_file, img_size, batchsize=32, nproc=1):
        self.ntags_per_attr =  350
        super(FashionDataIter, self).__init__(image_dir, pair_file, shop_file, self.ntags_per_attr, img_size=img_size, batchsize=32, nproc=nproc)


def calc_mean_std(image_dir, data_dir):
    data = FashionDataIter(image_dir, os.path.join(data_dir, 'sample_pair.txt'),
        os.path.join(data_dir, 'train_shop.txt'))
    qrgb = np.zeros((3,), dtype=np.float32)
    nrgb = np.zeros((3,), dtype=np.float32)
    count = 0
    data.start(data.load_triples)
    for i in trange(data.num_batches):
        qimg, _, nimg, _,  _ = data.next()
        for (rgb, img) in zip([qrgb, nrgb], [qimg, nimg]):
            for i in range(3):
                rgb[i] += np.average(img[:, i, :, :])
        count += img.shape[0]

    qrgb /= count
    nrgb /= count

    qstd = np.zeros((3,), dtype=np.float32)
    nstd = np.zeros((3,), dtype=np.float32)

    data.start(data.load_triples)
    for i in trange(data.num_batches):
        qimg, _, nimg, _,  _ = data.next()
        for (mean, std, img) in zip([qrgb, nrgb], [qstd, nstd], [qimg, nimg]):
            for i in range(3):
                d =  mean[i] - np.average(img[:, i, :, :])
                std += d * d

    qstd = np.sqrt(qstd / count)
    nstd = np.sqrt(nstd / count)

    return np.array([qrgb, qstd, nrgb, nstd], dtype=np.float32)


def sample(data_dir, ratio=0.2):
    with open(os.path.join(data_dir, 'train_pair.txt'), 'r') as fd:
        lines = fd.readlines()
    np.random.shuffle(lines)
    with open(os.path.join(data_dir, 'sample_pair.txt'), 'w') as fd:
        for line in lines[0: int(len(lines) * ratio)]:
            fd.write(line)
        fd.flush()


def benchmark1(image_dir):
    # print 'cv2 optimized ', cv2.useOptimized()
    data = DARNDataIter(image_dir, './data/darn/train_pair.txt', './data/darn/train_shop.txt', img_size=224, batchsize=50)
    t = time.time()
    for i in trange(50):
        fpath, _, _ = data.shop_image[i]
        data.read_image(fpath)
    print('time per image = %f' % ((time.time() - t)/50))

    t = time.time()
    data.load_triples(0, data.num_batches)
    print('time per triple = %f' % ((time.time() - t)/50))


def benchmark2(image_dir):
    data = DARNDataIter(image_dir, './data/deepfashion/train_pair.txt', './data/deepfashion/train_shop.txt', batchsize=32, nproc=1)
    data.start(data.load_triples)
    for i in trange(50):
        data.next()
    data.stop()


if __name__ == '__main__':
    image_dir = '/data/jixin/darn_dataset'
    benchmark1(image_dir)
    '''
    sample('./data/deepfashion/')
    meta = calc_mean_std(image_dir, './data/deepfashion')
    np.save('./data/deepfashion/meta', meta)

    benchmark2()
        val_dat = DARNDataIter(image_dir, './data/darn/validation_pair.txt', './data/darn/validation_shop.txt')
    val_dat.start(val_dat.load_street_images)
    for i in trange(val_dat.num_batches):
        val_dat.next()
    time.sleep(1)

    val_dat.start(val_dat.load_shop_images)
    for i in trange(len(val_dat.shop_image) / val_dat.batchsize):
        val_dat.next()
    '''
