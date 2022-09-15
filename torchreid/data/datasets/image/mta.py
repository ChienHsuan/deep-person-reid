import os.path as osp
import glob
import re

import imagesize

from ..dataset import ImageDataset

class MTA(ImageDataset):
    """
    MTA(Multi Camera Track Auto) dataset
    Reference:
    The MTA Dataset for Multi Target Multi Camera Pedestrian Tracking by Weighted Distance Aggregation. CVPRW 2020
    """
    dataset_dir = 'mta/MTA_reid/'

    def __init__(self, root='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        super(MTA, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'camid_(\d)_pid_(\d+)')

        pid_container = set()
        for i, img_path in enumerate(img_paths):
            width, height = imagesize.get(img_path)
            if height > 65:
                _, pid = map(int, pattern.search(img_path).groups())
                pid_container.add(pid)
            else:
                img_paths[i] = None
        img_paths = list(filter(None, img_paths))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            camid, pid = map(int, pattern.search(img_path).groups())
            assert 0 <= pid
            assert 0 <= camid <= 5
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
