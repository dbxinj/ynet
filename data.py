import random
import numpy as np
import time
from multiprocessing import Process, Queue

from singa import image_tool

train_tool = image_tool.ImageTool()
validate_tool = image_tool.ImageTool()

class DataIter(object):
    def __init__(self, image_dir, pair_file, shop_file, ntags_per_attr,
                 batchsize=32, capacity=50, shuffle=True, delimiter=' ',
                 small_size=256, large_size=288, crop_size=224):
        self.batchsize = batchsize
        self.image_folder = image_dir
        self.small_size = small_size
        self.large_size = large_size
        self.crop_size = crop_size
        self.stop = False
        self.proc = None  # for loading triplet
        self.queue = Queue(capacity)
        self.pos = 0

        tag_offset = []
        self.tag_dim = 0
        for ntags in ntags_per_attr:
            tag_offset.append(ntags)
            self.tag_dim += ntags

        with open(pair_file, 'r') as fd:
            for line in fd.readlines():
                vals = line.strip('\n').split(delimiter)
                tags = [int(idx) + off for (idx, off) in zip(vals[3:], tag_offset)]
                record = (vals[0], vals[1], vals[2], tags)
                self.image_pair.append(record)
            self.num_batches = len(self.image_pair) / batchsize
            self.idx = range(len(self.image_pair))

        with open(shop_file, 'r') as fd:
            for line in fd.readlines():
                vals = line.strip('\n').split(delimiter)
                tags = [int(idx) + off for (idx, off) in zip(vals[2:], tag_offset)]
                record = (vals[0], vals[1], tags)
                self.shop_image.append(record)

    def next(self):
        assert self.proc is not None, 'call start before next'
        while self.queue.empty():
            time.sleep(0.1)
        return self.queue.get()  # dequeue one mini-batch

    def start(self, func):
        # func to be load_triples, load_street_images, load_shop_images
        self.proc = Process(target=func)
        self.proc.start()

    def stop(self):
        if self.proc is not None:
            self.stop = True
            time.sleep(0.1)
            self.proc.terminate()

    def read_image(self, path, is_train=True):
        if is_train:
            img = train_tool.load(path).resize_by_range(
                (self.small_size, self.large_size)).random_crop(
                (self.crop_size, self.crop_size)).get()
        else:
            img = image_tool.crop(validate_tool.load(path).resize_by_list(
                [(self.small_size + self.large_size) / 2]).get(), 'center')
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        return ary

    def tag2vec(self, tags):
        vec = np.zeros((self.tag_dim), dtype=np.float32)
        vec[tags] = 1.0
        return vec

    def load_triples(self):
        while not self.stop:
            if not self.queue.full():
                qimgs = np.array((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
                pimgs = np.array((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
                nimgs = np.array((self.batchsize, 3, self.img_size, self.img_size), dtype=np.float32)
                ptags = np.array((self.batchsize, self.tag_dim), dtype=np.float32)
                ntags = np.array((self.batchsize, self.tag_dim), dtype=np.float32)
                for i, record in enumerate(self.image_pair[self.idx[self.pos: self.pos+self.bathcsize]]):
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
                self.pos += self.batchsize
                if self.pos + self.batchsize > len(self.image_pair):
                    self.pos = 0  # seek to the first record
                    if self.shuffle:
                        random.shuffle(self.idx)
            else:
                time.sleep(0.1)

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

    def load_shop_images(self):
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


class DARNDataIter(DataIter):
    def __init__(self, pair_file, shop_file):
        super(DARNDataIter, self).__init__(pair_file, shop_file)
        self.num_values = []


class FashionDataIter(DataIter):
    def __init__(self, pair_file, shop_file):
        super(FashionDataIter, self).__init__(pair_file, shop_file)
        self.num_values = []
