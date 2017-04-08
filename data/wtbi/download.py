import urllib
import os
from tqdm import trange
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description="Context-depedent attention modeling")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--out_dir", default='../../../wtbi/', help='image folder')
    parser.add_argument("--photo", default='photos.txt')
    args = parser.parse_args()

    with  open(args.photo,'r') as fd:
        lines = fd.readlines()

    nlines = len(lines) // args.n
    s = nlines * args.id
    e = s + nlines + len(lines) % args.n

    for i in trange(e-s):
        id, url = lines[i+s].strip('\n').split(',')
        out = os.path.join(args.out_dir, '%d.jpg' % int(id))
        if not os.path.exists(out):
            try:
                urllib.urlretrieve(url, out)
            except Exception as e:
                print('bad url %s' % url)
