import numpy as np
import torch.nn as nn
import torch

class FCN32s(nn.Module):
  def __init__(self, n_class, pretrained_model):
    # model structure
    super(FCN32s, self).__init__()
    # conv1
    self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
    self.relu1_1 = nn.ReLU(inplace=True)
    self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
    self.relu1_2 = nn.ReLU(inplace=True)
    self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

    # conv2
    self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
    self.relu2_1 = nn.ReLU(inplace=True)
    self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
    self.relu2_2 = nn.ReLU(inplace=True)
    self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

    # conv3
    self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
    self.relu3_1 = nn.ReLU(inplace=True)
    self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_2 = nn.ReLU(inplace=True)
    self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
    self.relu3_3 = nn.ReLU(inplace=True)
    self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

    # conv4
    self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
    self.relu4_1 = nn.ReLU(inplace=True)
    self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_2 = nn.ReLU(inplace=True)
    self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu4_3 = nn.ReLU(inplace=True)
    self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

    # conv5
    self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_1 = nn.ReLU(inplace=True)
    self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_2 = nn.ReLU(inplace=True)
    self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
    self.relu5_3 = nn.ReLU(inplace=True)
    self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

    # fc6
    self.fc6 = nn.Conv2d(512, 3192, 7)
    self.relu6 = nn.ReLU(inplace=True)
    self.drop6 = nn.Dropout2d()

    # fc7
    self.fc7 = nn.Conv2d(3192, 3192, 1)
    self.relu7 = nn.ReLU(inplace=True)
    self.drop7 = nn.Dropout2d()

    self.score_fr = nn.Conv2d(3192, n_class, 1)
    self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
    
    self.pretrained_model = pretrained_model
    self.initialize_weight()

  def forward(self, x):
    h = x
    h = self.relu1_1(self.conv1_1(h))
    h = self.relu1_2(self.conv1_2(h))
    h = self.pool1(h)

    h = self.relu2_1(self.conv2_1(h))
    h = self.relu2_2(self.conv2_2(h))
    h = self.pool2(h)

    h = self.relu3_1(self.conv3_1(h))
    h = self.relu3_2(self.conv3_2(h))
    h = self.relu3_3(self.conv3_3(h))
    h = self.pool3(h)

    h = self.relu4_1(self.conv4_1(h))
    h = self.relu4_2(self.conv4_2(h))
    h = self.relu4_3(self.conv4_3(h))
    h = self.pool4(h)

    h = self.relu5_1(self.conv5_1(h))
    h = self.relu5_2(self.conv5_2(h))
    h = self.relu5_3(self.conv5_3(h))
    h = self.pool5(h)

    h = self.relu6(self.fc6(h))
    h = self.drop6(h)

    h = self.relu7(self.fc7(h))
    h = self.drop7(h)

    h = self.score_fr(h)

    h = self.upscore(h)
    h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

    return h

  # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
  def get_upsampling_weight(self, layer):
    """Make a 2D bilinear kernel suitable for upsampling"""
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    kernel_size = layer.kernel_size[0]
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
          (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

  def initialize_weight(self):
    assert self.pretrained_model is not None
    VGG16 = self.pretrained_model

    features = [
      self.conv1_1, self.relu1_1,
      self.conv1_2, self.relu1_2,
      self.pool1,
      self.conv2_1, self.relu2_1,
      self.conv2_2, self.relu2_2,
      self.pool2,
      self.conv3_1, self.relu3_1,
      self.conv3_2, self.relu3_2,
      self.conv3_3, self.relu3_3,
      self.pool3,
      self.conv4_1, self.relu4_1,
      self.conv4_2, self.relu4_2,
      self.conv4_3, self.relu4_3,
      self.pool4,
      self.conv5_1, self.relu5_1,
      self.conv5_2, self.relu5_2,
      self.conv5_3, self.relu5_3,
      self.pool5
    ]

    for l1, l2 in zip(VGG16.features, features):
      if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
        wshape = l2.weight.size()
        l2.weight.data.copy_(l1.weight.data[:wshape[0], :wshape[1], :, :])
        bshape = l2.bias.size()
        l2.bias.data.copy_(l1.bias.data[:bshape[0]])

    wshape = self.fc6.weight.size()
    w_tot = torch.numel(self.fc6.weight)
    self.fc6.weight.data.copy_(VGG16.classifier[0].weight.data[:wshape[0], :int(w_tot / wshape[0])].view(self.fc6.weight.size()))
    bshape = self.fc6.bias.shape()
    self.fc6.bias.data.copy_(VGG16.classifier[0].bias.data[:bshape[0]].view(self.fc6.bias.size()))

    wshape = self.fc7.weight.size()
    w_tot = torch.numel(self.fc7.weight)
    self.fc7.weight.data.copy_(VGG16.classifier[3].weight.data[:wshape[0], :int(w_tot / wshape[0])].view(self.fc7.weight.size()))
    bshape = self.fc7.bias.shape()
    self.fc7.bias.data.copy_(VGG16.classifier[3].bias.data[:bshape[0]].view(self.fc7.bias.size()))

    self.score_fr.weight.data.zero_()
    self.score_fr.bias.data.zero_()
    self.upscore.weight.data.copy_(self.get_upsampling_weight(self.upscore))

  