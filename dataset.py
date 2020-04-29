import collections
import os.path as osp

import numpy as np
import PIL.Image
import torch
from torch.utils import data

class VOC2012(data.Dataset):
  mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
  def __init__(self, root, split):
    self.root = root
    self.split = split

    dataset_dir = osp.join(self.root, 'VOCdevkit/VOC2012')
    self.files = collections.defaultdict(list)
    for split in ['train', 'val']:
      imgsets_file = osp.join(
        dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
      for did in open(imgsets_file):
        did = did.strip()
        img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
        lbl_file = osp.join(
            dataset_dir, 'SegmentationClass/%s.png' % did)
        self.files[split].append({
            'img': img_file,
            'lbl': lbl_file,
        })

  def __len__(self):
    return len(self.files[self.split])

  def __getitem__(self, index):
    data_file = self.files[self.split][index]
    img_file = data_file['img']
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=np.uint8)
    lbl_file = data_file['lbl']
    lbl = PIL.Image.open(lbl_file)
    lbl = np.array(lbl, dtype=np.int32)
    lbl[lbl == 255] = -1
    return self.transform(img, lbl)

  def transform(self, img, lbl):
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= self.mean_bgr
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    lbl = torch.from_numpy(lbl).long()
    return img, lbl

  def get_palette(self):
    lbl = PIL.Image.open(self.files[self.split][0]['lbl'])
    return np.array(lbl.getpalette()).reshape(-1, 3)