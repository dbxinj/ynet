import random
import numpy as np
import time
import math
import os
import sys
from multiprocessing import Process, Queue
from tqdm import trange

from singa import image_tool

train_tool = image_tool.ImageTool()
validate_tool = image_tool.ImageTool()

class DataIter(object):
    def __init__(self, image_dir, pair_file, shop_file, ntags_per_attr,
                 batchsize=32, capacity=50, delimiter=' ',
                 image_size=224, nproc=2):
        self.batchsize = batchsize
        self.image_folder = image_dir
        self.image_size = image_size
        self.capacity = capacity
        self.nproc = nproc
        self.proc = []  # for loading triplet
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
        try:
            while self.queue.empty():
                time.sleep(0.1)
            return self.queue.get()  # dequeue one mini-batch
        except Exception as e:
            print(e)
        except:
            self.terminate()
            sys.exit(1)

    def start(self, func):
        # func to be load_triples, load_street_images, load_shop_images
        while not self.queue.empty():
            self.queue.get()
        for i in range(self.nproc):
            self.proc.append(Process(target=func, args=(i, self.nproc)))
            time.sleep(0.4)
            self.proc[-1].start()

    def terminate(self):
        for proc in self.proc:
            time.sleep(0.1)
            proc.terminate()
        self.proc = []

    def read_image(self, path, is_train=True):
        img = image_tool.load_img(path).resize((self.image_size, self.image_size))
        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
        return ary.transpose(2, 0, 1)

    def tag2vec(self, tags):
        vec = np.zeros((self.tag_dim,), dtype=np.float32)
        vec[tags] = 1.0
        return vec

    def do_shuffle(self):
        random.shuffle(self.idx)

    def load_triples(self, proc_id, nproc):
        nbatch = len(self.image_pair) // self.batchsize
        nbatch_per_proc = nbatch // nproc
        batch_start = nbatch_per_proc * proc_id
        if proc_id == nproc - 1:
            nbatch_per_proc += nbatch % nproc
        b = batch_start
        while b < batch_start + nbatch_per_proc:
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
                self.queue.put((qimgs, pimgs, nimgs, ptags, ntags))
                b += 1
            else:
                time.sleep(0.1)
        print('finish load triples')

    def load_street_images(self, proc_id, nproc):
        nbatch = len(self.image_pair) // self.batchsize
        nbatch_per_proc = nbatch // nproc
        batch_start = nbatch_per_proc * proc_id
        if proc_id == nproc - 1:
            nbatch_per_proc += nbatch % nproc
        # self.queue = Queue(self.capacity)
        b = batch_start
        while b < batch_start + nbatch_per_proc:
            if not self.queue.full():
                qimgs = np.empty((self.batchsize, 3, self.image_size, self.image_size), dtype=np.float32)
                items = []
                for i, rec in enumerate(self.image_pair[b * self.batchsize: (b + 1) * self.batchsize]):
                    qimgs[i, :] = self.read_image(rec[0])
                    items.append(rec[2])
                # enqueue one mini-batch
                self.queue.put((qimgs, items))
                b += 1
            else:
                time.sleep(0.1)
        print('finish load street')

    def load_shop_images(self, proc_id, nproc):
        nbatch = len(self.shop_image) // self.batchsize
        nbatch_per_proc = nbatch // nproc
        batch_start = nbatch_per_proc * proc_id
        if proc_id == nproc - 1:
            nbatch_per_proc += nbatch % nproc
        # self.queue = Queue(self.capacity)
        b = batch_start
        while b < batch_start + nbatch_per_proc:
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
                b += 1
            else:
                time.sleep(0.1)
        print('finish load shop images')


class DARNDataIter(DataIter):
    def __init__(self, image_dir, pair_file, shop_file):
        self.ntags_per_attr = [20, 56, 10, 25, 27, 16, 7, 12, 6]
        super(DARNDataIter, self).__init__(image_dir, pair_file, shop_file, self.ntags_per_attr, image_size=227)


class FashionDataIter(DataIter):
    def __init__(self, image_dir, pair_file, shop_file):
        self.ntags_per_attr = []
        super(FashionDataIter, self).__init__(image_dir, pair_file, shop_file, self.ntags_per_attr)


def calc_mean_std(image_dir, data_dir):
    data = DARNDataIter(image_dir, os.path.join(data_dir, 'sample_pair.txt'),
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


if __name__ == '__main__':
    image_dir = '/home/wangyan/darn_dataset' #'/data/jixin/darn_dataset'
    # sample('./data/darn/')
    # train_dat.start(train_dat.load_triples)
    meta = calc_mean_std(image_dir, './data/darn')
    np.save('./data/darn/meta', meta)
    '''
    val_dat = DARNDataIter(image_dir, './data/darn/validation_pair.txt', './data/darn/validation_shop.txt')
    val_dat.start(val_dat.load_street_images)
    for i in trange(val_dat.num_batches):
        val_dat.next()
    time.sleep(1)

    val_dat.start(val_dat.load_shop_images)
    for i in trange(len(val_dat.shop_image) / val_dat.batchsize):
        val_dat.next()
    '''
