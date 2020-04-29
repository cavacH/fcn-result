import torch
import csv
import os
from torch.utils.tensorboard import SummaryWriter

def conf_summary(conf):
  return '{}_{}_{}{}'.format(conf['optim'], str(conf['lr']), conf['scheduler'], '_' + conf['special'] if conf['special'] else '')

def merge_conf(confs):
  cur = {}
  for k, v in confs[0].items():
    cur[k] = v
  for conf in confs:
    for k, v in conf.items():
      if cur[k] != v:
        cur[k] = str(cur[k]) + 'VS' + str(v)
  return conf_summary(cur)

def go(confs, mets, max_iter):
  data = []
  legends = []
  writer = SummaryWriter('logs/' + merge_conf(confs))
  for conf in confs:
    doc_name = conf_summary(conf)
    legends.append(doc_name)
    content = list(csv.reader(open('drive/My Drive/' + doc_name + '/log.csv')))
    metrics = content[0]
    content = content[1:]
    iters = [[None for j in range(len(metrics))] for i in range(max_iter)]
    for row in content:
      cur_iter = int(row[1])
      if cur_iter >= max_iter:
        continue
      assert len(row) == len(metrics)
      for j in range(len(row)):
        if len(row[j]) > 0:
          iters[cur_iter][j] = float(row[j])
    #print(iters[:5])
    meta = {}
    for x in metrics:
      meta[x] = []
    for i in range(max_iter):
      row = iters[i]
      for j in range(len(metrics)):
        if row[j] is not None:
          meta[metrics[j]].append((int(row[1]), row[j]))
    data.append(meta)
  
  final = {}
  for met in mets:
    final[met] = []
    L = len(data[0][met])
    for doc in data:
      try:
        assert len(doc[met]) == L
      except:
        print(L, len(doc[met]))
    for i in range(L):
      cur_iter = data[0][met][i][0]
      val = {}
      for j in range(len(data)):
        doc = data[j]
        assert doc[met][i][0] == cur_iter
        val[legends[j]] = doc[met][i][1]
      final[met].append((cur_iter, val))

  for met in mets:
    for datum in final[met]:
      try:
        writer.add_scalars(met, datum[1], datum[0])
      except:
        print(datum)

if __name__ == '__main__':
  confs = [{
      'optim': 'Adam',
      'lr': 1e-5,
      'scheduler': 'poly',
      'special': '3192fts'
  }, {
      'optim': 'Adam',
      'lr': 1e-5,
      'scheduler': 'poly',
      'special': '384ftmap'
  }, {
      'optim': 'Adam',
      'lr': 1e-5,
      'scheduler': 'poly',
      'special': 'no_conv5'
  }]
  mets = ['train/loss', 'train/mean_iu', 'valid/loss', 'valid/mean_iu', 'elapsed_time']
  go(confs, mets, 40001)
      