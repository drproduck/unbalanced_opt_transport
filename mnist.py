import random
import pdb
import os 
import numpy as np
import random

import gzip
import pickle as pkl
from urllib import request


def _download_mnist(root_dir):
    data_loc = os.path.join(root_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_loc):
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print("Downloading data from:", url)
        data_loc = os.path.join(root_dir, 'mnist.pkl.gz')
        data_loc, _ = request.urlretrieve(url, data_loc)
    else: print('WARNING: data might already exist')
    return data_loc

def _load_mnist(root_dir,  split_type, download):
    if download:
        data_loc = _download_mnist(root_dir)
    else:
        data_loc = os.path.join(root_dir, 'mnist.pkl.gz')
    f = gzip.open(data_loc, 'rb')

    train, valid, test = pkl.load(f, encoding='bytes')
    f.close()
    if split_type == 'train':
        x, y = train[0], train[1]
    if split_type == 'valid':
        x, y = valid[0], valid[1]
    if split_type == 'test':
        x, y = test[0], test[1]
    return x, y

