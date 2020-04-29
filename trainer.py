import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    return acc, acc_cls, mean_iu

class Trainer(object):
  def __init__(self, n_class, cuda, log_dir, model, writer, optimizer, scheduler, train_loader, val_loader, max_iter, val_interval, log_interval):
    self.model = model
    self.cuda = cuda
    self.optim = optimizer
    self.scheduler = scheduler
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.max_iter = max_iter
    self.val_interval = val_interval
    self.log_interval = log_interval
    self.n_class = n_class
    self.writer = writer
    self.epoch = 0
    self.log_dir = log_dir

  def validate(self):
    self.model.eval()
    val_loss = 0.0
    y_preds, y_trues = [], []
    palette = self.val_loader.dataset.get_palette()
    for batch_idx, (data, target) in enumerate(self.val_loader):
      if self.cuda:
        data, target = data.cuda(), target.cuda()
      data, target = Variable(data), Variable(target)
      with torch.no_grad():
        score = self.model(data)

      loss = F.nll_loss(F.log_softmax(score, dim=1), target, ignore_index=-1)
      val_loss += loss.item()
      
      y_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
      y_true = target.data.cpu().numpy()[:, :, :]
      if batch_idx < 6:
        self.writer.add_images('epoch_%d_preds' % self.epoch, np.array([palette[x] for x in y_pred]), dataformats='NHWC')
        self.writer.add_images('epoch_%d_trues' % self.epoch, np.array([palette[x] for x in y_true]), dataformats='NHWC')

      for yt, yp in zip(y_true, y_pred):
          y_trues.append(yt)
          y_preds.append(yp)

    acc, mean_acc, mean_iu = label_accuracy_score(y_trues, y_preds, self.n_class)
    val_loss /= len(self.val_loader)
    self.model.train()
    return val_loss, acc, mean_acc, mean_iu

  def train(self):
    self.model.train()
    max_epoch = int(math.ceil(1.0 * self.max_iter / len(self.train_loader)))
    running_loss = 0.0
    for epoch in range(max_epoch):
      self.epoch = epoch
      for batch_idx, (data, target) in enumerate(self.train_loader):
        iteration = batch_idx + epoch * len(self.train_loader) + 1

        assert self.model.training

        if self.cuda:
          data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        self.optim.zero_grad()
        score = self.model(data)
        loss = F.nll_loss(F.log_softmax(score, dim=1), target, ignore_index=-1)
        running_loss += loss.item()

        loss.backward()
        self.optim.step()

        if iteration % self.log_interval == 0:
          running_loss /= self.log_interval
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx+1) * len(data), len(self.train_loader.dataset),
            100. * (batch_idx+1) / len(self.train_loader), running_loss))
          self.writer.add_scalar('Loss/train', running_loss, iteration)
          running_loss = 0.0

        if iteration % self.val_interval == 0:
          val_loss, acc, mean_acc, mean_iu = self.validate()
          print('val_loss: {:.6f} acc: {:.6f} mean_acc: {:.6f} mean_iu: {:.6f}'.format(val_loss, acc, mean_acc, mean_iu))
          self.writer.add_scalar('Loss/val', val_loss, iteration)
          self.writer.add_scalar('Acc/val', acc, iteration)
          self.writer.add_scalar('Mean_acc/val', mean_acc, iteration)
          self.writer.add_scalar('Mean_IU/val', mean_iu, iteration)

          torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
          }, os.path.join(self.log_dir, 'iter_%d.pth.tar' % iteration))
        
        if iteration >= self.max_iter:
          break
      if scheduler:
        self.scheduler.step()