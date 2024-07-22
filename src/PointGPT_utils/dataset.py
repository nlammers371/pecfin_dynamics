import sys
sys.path.append('/home/nick/projects')
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PointGPT.segmentation.pointnet_util import farthest_point_sample, pc_normalize
import pandas as pd
import glob2 as glob
from src.utilities.functions import path_leaf
import json


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(
            os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(
            os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class PartNormalDataset(Dataset):
    def __init__(self, root='/data/cgy/ShapenetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        train_ids = set(['02691156-10155655850468db78d106ce0a280f87', '02691156-1021a0914a7207aff927ed529ad90a11'])
        val_ids = set([2, 3])
        test_ids = set([4, 5])
        # with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
        #     train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        # with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        #     val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        # with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
        #     test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = fns #[fn for fn in fns if (
                    # (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.npy'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.load(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)

# NL: custom point dataset class
class FinDataset(Dataset):
    def __init__(self, root, cls_label=1, npoints=4096, split='train', outpath=None, overwrite=False, split_file=None,
                            train_test_val_split=None, seed=723):
        
        if split_file is None and train_test_val_split is None:
            train_test_val_split = [.8, .1, .1]
            
        self.npoints = npoints
        self.root = root
        self.cls_label = cls_label
        self.split = split

        # self.meta = {}
        if split_file is None:
            np.random.seed(seed)
            point_file_list = sorted(glob.glob(os.path.join(root, "data", '*.csv')))
            files_to_analyze = point_file_list
            if (outpath is not None) and overwrite == False:
                existing_feature_files = glob.glob(os.path.join(outpath, '*.csv'))
                feature_stubs = [path_leaf(file)[:26] for file in existing_feature_files]
                point_stubs = [path_leaf(file)[:26] for file in point_file_list]
                files_to_analyze = [point_file_list[i] for i in range(len(point_file_list)) if point_stubs[i] not in feature_stubs]

            elif outpath is None:
                print("Outpath provided but overwrite set to True. Did you mean to set overwrite_flag=False?")


            n_files = len(files_to_analyze)
            n_train = np.floor(n_files * train_test_val_split[0]).astype(int)
            n_test = np.floor(n_files * train_test_val_split[1]).astype(int)
            # n_val = n_files - n_train - n_test
            shuffled_file_ids = np.random.choice(range(n_files), n_files, replace=False)
            self.train_ids = shuffled_file_ids[:n_train]
            self.test_ids = shuffled_file_ids[n_train:n_train+n_test]
            self.val_ids = shuffled_file_ids[n_train+n_test:]

        else:
            raise Exception("split file option is not yet implemented")

        file_strings = [path_leaf(fn) for fn in files_to_analyze]
        file_strings = [fn.replace("_nuclei", "") for fn in file_strings]
        file_strings = [fn.replace(".csv", "") for fn in file_strings]
        if split=="train":
            use_ids = self.train_ids
        elif split=="val":
            use_ids = self.val_ids
        elif split=="test":
            use_ids = self.test_ids
        elif split=="all":
            use_ids = np.asarray(self.test_ids.tolist() + self.train_ids.tolist() + self.val_ids.tolist())
        else:
            raise Exception("Unknown split: %s" % split)

        self.datapath = []
        for id in use_ids:
            self.datapath.append((file_strings[id], files_to_analyze[id]))


        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg, fn = self.cache[index]
        else:
            fn = self.datapath[index]
            # cat = self.datapath[index][0]
            # cls = self.classes[cat]
            # cls = np.array([cls]).astype(np.int32)
            cls = np.array(self.cls_label).astype(np.int32)  # using motorbike because it appears the most complex
            data = pd.read_csv(fn[1])

            # if not self.normal_channel:
            #     point_set = data[:, 0:3]
            # else:
            #     point_set = data[:, 0:6]
            point_set = data.loc[:, ["Z", "Y", "X"]].to_numpy().astype(np.float32)
            seg = np.zeros(point_set[:, 0].shape)  # placeholder for now
            # seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.split != "all":
            choice = np.random.choice(len(seg), self.npoints, replace=True)
        elif self.npoints <= point_set.shape[0]:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            choice0 = np.random.choice(len(seg), len(seg), replace=False)
            choice1 = np.random.choice(len(seg), self.npoints - len(seg), replace=True)
            choice = np.concatenate((choice0, choice1))

        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg, fn, choice

    def __len__(self):
        return len(self.datapath)
if __name__ == '__main__':
    data = ModelNetDataLoader('modelnet40_normal_resampled/',
                              split='train', uniform=False, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
