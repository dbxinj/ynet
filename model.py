class CANet():
    '''Context-depedent attention modeling'''
    def __init__(self, name, dev):
        pass

    def create_net(self):
        pass

    def init_params(self):
        pass

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
