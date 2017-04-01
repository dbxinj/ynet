import random
import numpy as np
import time
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
        self.stop = False
        self.proc = None  # for loading triplet
        self.queue = Queue(capacity)

        tag_offset = []
        self.tag_dim = 0
        for ntags in ntags_per_attr:
            tag_offset.append(ntags)
            self.tag_dim += ntags

        with open(pair_file, 'r') as fd:
            for line in fd.readlines():
                vals = line.strip('\n').split(delimiter)
                tags = [int(idx) + off if int(idx) != -1 for (idx, off) in zip(vals[3:], tag_offset)]
                record = (vals[0], vals[1], vals[2], tags)
                self.image_pair.append(record)
            self.num_batches = len(self.image_pair) / batchsize
            self.idx = range(len(self.image_pair))

        with open(shop_file, 'r') as fd:
            for line in fd.readlines():
                vals = line.strip('\n').split(delimiter)
                tags = [int(idx) + off if int(idx) != -1 for (idx, off) in zip(vals[3:], tag_offset)]
                record = (vals[0], vals[1], tags)
                self.shop_image.append(record)

    def next(self):
        assert self.proc is not None, 'call start before next'
        while self.queue.empty():
            time.sleep(0.1)
        return self.queue.get()  # dequeue one mini-batch

    def start(self, func):
        # func to be load_triples, load_street_images, load_shop_images
        self.stop = False
        self.proc = Process(target=func)
        self.proc.start()

    def stop(self):
        if self.proc is not None:
            self.stop = True
            time.sleep(0.1)
            self.proc.terminate()

    def read_image(self, path, is_train=True):
        img = image_tool.load_img(path).resize((self.image_size, self.image_size))
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        return ary

    def tag2vec(self, tags):
        vec = np.zeros((self.tag_dim), dtype=np.float32)
        vec[tags] = 1.0
        return vec

    def load_triples(self):
        self.queue.clear()
        if self.shuffle:
            random.shuffle(self.idx)
        for b in range(len(self.image_pair) / self.batchsize):
            if self.stop:
                return
            if not self.queue.full():
                qimgs = np.array((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
                pimgs = np.array((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
                nimgs = np.array((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
                ptags = np.array((self.batchsize, self.tag_dim), dtype=np.float32)
                ntags = np.array((self.batchsize, self.tag_dim), dtype=np.float32)
                for i, record in enumerate(self.image_pair[self.idx[b * self.batchsize: (b + 1) * self.batchsize]]):
                    qimgs[i] = self.read_image(record[0])
                    pimgs[i] = self.read_image(record[1])
                    item = record[2]
                    ptags[i] = self.tag2vec(record[3])
                    nitem = item
                    while nitem == item:  # until find a negative sample
                        nidx = random.randint(0, len(self.shop_image))
                        nitem = self.shop_image[nidx][1]
                    nimgs[i] = self.read_image(self.shop_image[nidx][0])
                    ntags[i] = self.tag2vec(self.shop_image[nidx][2])
                # enqueue one mini-batch
                self.queue.put((qimgs, pimgs, ptags, nimgs, ntags))
            else:
                time.sleep(0.1)
        self.stop = True

    def load_street_images(self):
        self.queue.clear()
        for b in range(len(self.image_pair) / self.batchsize):
            if self.stop:
                return
            if not self.query_queue.full():
                qimgs = np.array((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
                for i, rec in enumerate(self.image_pair[b * self.batchsize: (b + 1) * self.batchsize]):
                    qimgs[i] = self.read_image(rec[0])
                    item = rec[2]
                # enqueue one mini-batch
                self.queue.put((qimgs, item))
            else:
                time.sleep(0.1)
        self.stop = True

    def load_shop_images(self):
        self.queue.clear()
        for b in range(len(self.shop_image) / self.batchsize):
            if self.stop:
                return
            if not self.query_queue.full():
                imgs = np.array((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
                tags = np.array((self.batchsize, self.total_val), dtype=np.float32)
                items = []
                for i, rec in enumerate(self.shop_image[b * self.batchsize: (b + 1) * self.batchsize]):
                    imgs[i] = self.read_image(rec[0])
                    items.append(rec[1])
                    tags[i] = self.tag2vec(rec[2])
                # enqueue one mini-batch
                self.queue.put((imgs, items, tags))
            else:
                time.sleep(0.1)
        self.stop = True

class DARNDataIter(DataIter):
    def __init__(self, image_dir, pair_file, shop_file):
        self.ntags_per_attr = [20, 56, 10, 25, 27, 16, 7, 12, 6]
        super(DARNDataIter, self).__init__(image_dir, pair_file, shop_file, self.ntags_per_attr)


class FashionDataIter(DataIter):
    def __init__(self, image_dir, pair_file, shop_file):
        self.ntags_per_attr = []
        super(FashionDataIter, self).__init__(image_dir, pair_file, shop_file, self.ntags_per_attr)


if __name__ == '__main__':
    train_dat = DARNDataIter('../data/darn/image', '../data/darn/train_pair.txt', '../data/darn/train_shop.txt')
    train_dat.start(train_dat.load_triples)
    for i in trange(train_dat.num_batches):
        train_dat.next()
    assert train_dat.stop == True, 'load pairs should stop!'

    val_dat = DARNDataIter('../data/darn/image', '../data/darn/train_pair.txt', '../data/darn/train_shop.txt')
    val_dat.start(val_dat.load_street_images)
    for i in trange(val_dat.num_batches):
        val_dat.next()
    assert val_dat.stop == True, 'load street image should stop!'

    val_dat.start(val_dat.load_shop_images)
    for i in trange(len(val_dat.shop_image) / val_dat.batchsize):
        val_dat.next()
    assert val_dat.stop == True, 'load shop image should stop!'
