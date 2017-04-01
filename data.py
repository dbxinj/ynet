import random
import numpy as np
import time
import os
from multiprocessing import Process, Queue
from tqdm import trange

from singa import image_tool

train_tool = image_tool.ImageTool()
validate_tool = image_tool.ImageTool()

class DataIter(object):
    def __init__(self, image_dir, pair_file, shop_file, ntags_per_attr,
                 batchsize=32, capacity=50, shuffle=True, delimiter=' ',
                 image_size=224):
        self.batchsize = batchsize
        self.image_folder = image_dir
        self.image_size = image_size
        self.capacity = capacity
        self.shuffle = shuffle
        self.proc = None  # for loading triplet
        self.queue = Queue(capacity)
        self.image_pair = []
        self.shop_image = []

        tag_offset = []
        self.tag_dim = 0
        for ntags in ntags_per_attr:
            tag_offset.append(self.tag_dim)
            self.tag_dim += ntags

        with open(pair_file, 'r') as fd:
            for line in fd.readlines():
                vals = line.strip('\n').split(delimiter)
                tags = [int(idx) + off for (idx, off) in zip(vals[3:], tag_offset) if int(idx) != -1]
                record = (os.path.join(image_dir, vals[0]), os.path.join(image_dir, vals[1]), vals[2], tags)
                self.image_pair.append(record)
            self.num_batches = len(self.image_pair) / batchsize
            self.idx = range(len(self.image_pair))

        with open(shop_file, 'r') as fd:
            for line in fd.readlines():
                vals = line.strip('\n').split(delimiter)
                tags = [int(idx) + off for (idx, off) in zip(vals[2:], tag_offset) if int(idx) != -1]
                record = (os.path.join(image_dir, vals[0]), vals[1], tags)
                self.shop_image.append(record)

    def next(self):
        assert self.proc is not None, 'call start before next'
        while self.queue.empty():
            time.sleep(0.1)
        return self.queue.get()  # dequeue one mini-batch

    def start(self, func):
        # func to be load_triples, load_street_images, load_shop_images
        while not self.queue.empty():
            self.queue.get()
        self.proc = Process(target=func)
        self.proc.start()

    def terminate(self):
        if self.proc is not None:
            time.sleep(0.1)
            self.proc.terminate()

    def read_image(self, path, is_train=True):
        img = image_tool.load_img(path).resize((self.image_size, self.image_size))
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        return ary.transpose(2, 0, 1)

    def tag2vec(self, tags):
        vec = np.zeros((self.tag_dim,), dtype=np.float32)
        vec[tags] = 1.0
        return vec

    def load_triples(self):
        # self.queue = Queue(self.capacity)
        if self.shuffle:
            random.shuffle(self.idx)
        for b in range(len(self.image_pair) / self.batchsize):
            if not self.queue.full():
                qimgs = np.empty((self.batchsize, 3, self.image_size, self.image_size), dtype=np.float32)
                pimgs = np.empty((self.batchsize, 3, self.image_size, self.image_size), dtype=np.float32)
                nimgs = np.empty((self.batchsize, 3, self.image_size, self.image_size), dtype=np.float32)
                ptags = np.empty((self.batchsize, self.tag_dim), dtype=np.float32)
                ntags = np.empty((self.batchsize, self.tag_dim), dtype=np.float32)
                for i, k in enumerate(self.idx[b * self.batchsize: (b + 1) * self.batchsize]):
                    record = self.image_pair[k]
                    qimgs[i, :] = self.read_image(record[0])
                    pimgs[i, :] = self.read_image(record[1])
                    item = record[2]
                    ptags[i] = self.tag2vec(record[3])
                    nitem = item
                    while nitem == item:  # until find a negative sample
                        nidx = random.randint(0, len(self.shop_image)-1)
                        nitem = self.shop_image[nidx][1]
                    nimgs[i] = self.read_image(self.shop_image[nidx][0])
                    ntags[i] = self.tag2vec(self.shop_image[nidx][2])
                # enqueue one mini-batch
                self.queue.put((qimgs, pimgs, ptags, nimgs, ntags))
            else:
                time.sleep(0.1)

    def load_street_images(self):
        for b in range(len(self.image_pair) / self.batchsize):
            if not self.queue.full():
                qimgs = np.empty((self.batchsize, 3, self.image_size, self.image_size), dtype=np.float32)
                for i, rec in enumerate(self.image_pair[b * self.batchsize: (b + 1) * self.batchsize]):
                    qimgs[i, :] = self.read_image(rec[0])
                    item = rec[2]
                # enqueue one mini-batch
                self.queue.put((qimgs, item))
            else:
                time.sleep(0.1)

    def load_shop_images(self):
        # self.queue = Queue(self.capacity)
        for b in range(len(self.shop_image) / self.batchsize):
            if not self.queue.full():
                imgs = np.empty((self.batchsize, 3, self.image_size, self.image_size), dtype=np.float32)
                tags = np.empty((self.batchsize, self.tag_dim), dtype=np.float32)
                items = []
                for i, rec in enumerate(self.shop_image[b * self.batchsize: (b + 1) * self.batchsize]):
                    imgs[i, :] = self.read_image(rec[0])
                    items.append(rec[1])
                    tags[i] = self.tag2vec(rec[2])
                # enqueue one mini-batch
                self.queue.put((imgs, items, tags))
            else:
                time.sleep(0.1)

class DARNDataIter(DataIter):
    def __init__(self, image_dir, pair_file, shop_file):
        self.ntags_per_attr = [20, 56, 10, 25, 27, 16, 7, 12, 6]
        super(DARNDataIter, self).__init__(image_dir, pair_file, shop_file, self.ntags_per_attr)


class FashionDataIter(DataIter):
    def __init__(self, image_dir, pair_file, shop_file):
        self.ntags_per_attr = []
        super(FashionDataIter, self).__init__(image_dir, pair_file, shop_file, self.ntags_per_attr)


if __name__ == '__main__':
    image_dir = '/data/jixin/darn_dataset'
    train_dat = DARNDataIter(image_dir, './data/darn/train_pair.txt', './data/darn/train_shop.txt')
    train_dat.start(train_dat.load_triples)
    for i in trange(train_dat.num_batches):
        train_dat.next()

    val_dat = DARNDataIter(image_dir, './data/darn/validation_pair.txt', './data/darn/validation_shop.txt')
    val_dat.start(val_dat.load_street_images)
    for i in trange(val_dat.num_batches):
        val_dat.next()
    time.sleep(1)

    val_dat.start(val_dat.load_shop_images)
    for i in trange(len(val_dat.shop_image) / val_dat.batchsize):
        val_dat.next()
