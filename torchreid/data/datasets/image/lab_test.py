import re
import glob
import os
import os.path as osp
import warnings

from ..dataset import ImageDataset


class LabTest(ImageDataset):
    """Lab data test
    """
    _junk_pids = [0, -1]
    dataset_dir = 'lab_testset'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = self.data_dir
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'Data structure error'
            )

        self.train_dir = osp.join(self.data_dir, 'query')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        super(LabTest, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = os.listdir(dir_path)

        data = []
        for img_path in img_paths:
            path = osp.join(dir_path, img_path)
            cam, id, _ = img_path.split('_')
            pid = int(id)
            camid = int(cam[1])
            assert 0 <= pid <= 10
            assert 0 <= camid <= 3

            data.append((path, pid, camid))

        return data
