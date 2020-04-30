import os
import numpy as np
import PIL.Image
import torch

class VOC2012(torch.utils.data.Dataset):
  mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
  def __init__(self, root, split):
    self.root = os.path.join(root, 'VOCdevkit/VOC2012/')
    self.split = split

    self.imgs = []
    self.labels = []

    names = open(os.path.join(self.root, 'ImageSets/Segmentation/%s.txt' % self.split)).read().strip().split('\n')
    for name in names:
      name = name.strip()
      self.imgs.append(os.path.join(self.root, 'JPEGImages/%s.jpg' % name))
      self.labels.append(os.path.join(self.root, 'SegmentationClass/%s.png' % name))

    self.imgs = self.imgs[:100]
    self.labels = self.labels[:100]

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, index):
    img = PIL.Image.open(self.imgs[index])
    img = np.array(img, dtype=np.float)
    img = img[:, :, ::-1]
    img -= self.mean_bgr
    img = img.transpose(2, 0, 1)
    label = PIL.Image.open(self.labels[index])
    label = np.array(label, dtype=np.int32)
    label[label == 255] = -1
    return torch.from_numpy(img.copy()).float(), torch.from_numpy(label).long()