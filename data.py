class DataIter():
    def __init__(self, pair_file, shop_file, batchsize=32):
        pass


    def shuffle(self):
        pass

    def next_triple(self):
        pass

    def next_query(self):
        pass

    def next_db(self):
        pass

    def stop(self):
        pass


class DARNDataIter(DataIter):
    def __init__(self, pair_file, shop_file):
        self.num_values = []


class FashionDataIter(DataIter):
    def __init__(self, pair_file, shop_file):
        super().__init__(pair_file, shop_file)
        self.num_values = []
