import random
from multiprocessing import Process, Queue

class DataIter(object):
    def __init__(self, image_dir, pair_file, shop_file, num_values, batchsize=32, capacity=50):
        self.queue = Queue(capacity)
        self.batchsize = batchsize
        self.image_folder = image_dir
        self.stop = False
        self.p = None

        offset = []
        total_val = 0
        for num_val in num_values:
            offset.append(total_val)
            total_val += num_val

        with open(pair_file, 'r') as fd:
            for line in fd.readlines():
                vals = line.split(' ')
                tags = [int(idx) + off for (idx, off) in zip(vals[3:], offset)]
                record = (vals[0], vals[1], vals[2], tags)
                self.image_pair.append(record)
            self.num_batches = len(self.image_pair) / batchsize
            self.idx = range(len(self.image_pair))

        with open(shop_file, 'r') as fd:
            for line in fd.readlines():
                vals = line.split(' ')
                tags = [int(idx) + off for (idx, off) in zip(vals[2:], offset)]
                record = (vals[0], vals[1], tags)
                self.shop_image.append(record)


    def shuffle(self):
        random.shuffle(self.idx)

    def next_triple(self):
        pass

    def next_query(self):
        pass

    def next_db(self):
        pass

    def start(self):
        self.p = Process(target=self.run)
        self.p.start()

    def stop(self):
        self.stop = True

    def read_image(self, path):
        pass

    def run(self):
        while not self.stop:
            if not self.queue.full():
                x = []
                y = np.empty(self.batchsize, dtype=np.int32)
                i = 0
                while i < self.batchsize:
                    img_label, img_path = img_list[index]
                    aug_images = self.image_transform(
                            os.path.join(self.image_folder, img_path))
                    assert i + len(aug_images) <= self.batch_size, \
                        'too many images (%d) in a batch (%d)' % \
                        (i + len(aug_images), self.batch_size)
                    for img in aug_images:
                        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
                        x.append(ary.transpose(2, 0, 1))
                        y[i] = img_label
                        i += 1
                    index += 1
                    if index == self.num_samples:
                        index = 0  # reset to the first image
                        if self.shuffle:
                            random.shuffle(img_list)
                # enqueue one mini-batch
                self.queue.put((np.asarray(x), y))
            else:
                time.sleep(0.1)
        return


class DARNDataIter(DataIter):
    def __init__(self, pair_file, shop_file):
        super(DARNDataIter, self).__init__(pair_file, shop_file)
        self.num_values = []


class FashionDataIter(DataIter):
    def __init__(self, pair_file, shop_file):
        super(FashionDataIter, self).__init__(pair_file, shop_file)
        self.num_values = []
