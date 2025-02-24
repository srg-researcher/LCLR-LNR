import anytree
import codecs
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index, get_datasets


class TABULAR(Dataset):
    data_name = 'TABULAR'
    file = [('', ''),
            ('', ''),
            ('', ''),
            ('', '')]

    def __init__(self, root, split, subset, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        # self.process()
        if not check_exists(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target = self.target[self.subset]
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.classes_to_labels, self.classes_size = self.classes_to_labels[self.subset], self.classes_size[self.subset]

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        input = {'img': img, self.subset: target}
        # if self.transform is not None:
        #     input = self.transform(input)
        return input

    def __len__(self):
        return len(self.img)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        # train_img = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
        # test_img = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
        # train_label = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        # test_label = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))

        train_file_path = r'C:\Users\panyi969\OneDrive - University of Otago\research\FedMD\fix-keras-bug\FedMD_clean\dataset\glass-0-1-2-3_vs_4-5-6\cross_validation_5\glass-0-1-2-3_vs_4-5-6-5-5tra.csv'
        test_file_path = r'C:\Users\panyi969\OneDrive - University of Otago\research\FedMD\fix-keras-bug\FedMD_clean\dataset\glass-0-1-2-3_vs_4-5-6\cross_validation_5\glass-0-1-2-3_vs_4-5-6-5-5tst.csv'
        # train_img, test_img, train_label, test_label = get_datasets(train_file_path=train_file_path,
        #                                                             test_file_path=test_file_path,
        #                                                             get_two_class=False,
        #                                                             class_one_index=0,
        #                                                             class_two_index=2)

        import pandas as pd
        df = pd.read_csv(train_file_path, header=0, delimiter=',').replace(to_replace=[' negative', ' positive'], value=[0, 1]).values
        # df = pd.read_csv(train_file_path, header=0, delimiter=',').values
        inputs = torch.from_numpy(df[:, 1:-1]).float()
        inputs = (inputs - inputs.min(dim=0)[0]) / (inputs.max(dim=0)[0] - inputs.min(dim=0)[0])
        inputs = inputs.nan_to_num(0)
        labels = torch.from_numpy(df[:, -1]).long()
        train_img = inputs
        train_label = labels

        df = pd.read_csv(test_file_path, header=0, delimiter=',').values
        inputs = torch.from_numpy(df[:, 1:-1]).float()
        inputs = (inputs - inputs.min(dim=0)[0]) / (inputs.max(dim=0)[0] - inputs.min(dim=0)[0])
        inputs = inputs.nan_to_num(0)
        labels = torch.from_numpy(df[:, -1]).long()
        test_img = inputs
        test_label = labels 

        # train_label = train_label.type(torch.LongTensor)
        # test_label = test_label.type(torch.LongTensor)
        train_target, test_target = {'label': train_label}, {'label': test_label}
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        classes = list(map(str, list(range(2))))
        for c in classes:
            make_tree(classes_to_labels['label'], [c])
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        print("type train_img: ", type(train_img))
        print("type train_label: ", type(train_label))
        print("shape train_img: ", train_img.shape)
        print("shape train_label: ", train_label.shape)

        return (train_img, train_target), (test_img, test_target), (classes_to_labels, classes_size)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape((length, num_rows, num_cols))
        return parsed


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(length).astype(np.int64)
        return parsed