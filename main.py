import os
import torch
import torchvision
from dataset import VOC2012
from trainer import Trainer
from FCN32s import FCN32s
from torch.utils.tensorboard import SummaryWriter

def get_dir(args):
  return os.path.join('./', args['optim'] + '_' + str(args['lr']) + '_' + args['scheduler'] + ('_' + args['special'] if args['special'] else ''))

def poly_lr(epoch):
  return (1 - (1.0 * epoch / 28.0)) ** 0.9

if __name__ == '__main__':
    args = {
        'max_iteration': 40000,
        'lr': 1e-10,
        'optim': 'SGD',
        'scheduler': 'base',
        'val_interval': 5000,
        'special': None,
        'dataset_root': './',
        'pretrained_vgg': '/content/drive/My Drive/vgg16_from_caffe.pth',
        'max_iter': 40000
    }

    log_dir = get_dir(args)
    try:
      os.makedirs(log_dir)
    except:
      pass

    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        VOC2012(args['dataset_root'], split='train'),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        VOC2012(args['dataset_root'], split='val'),
        batch_size=1, shuffle=False, **kwargs)

    vgg16 = torchvision.models.vgg16(pretrained=False)
    vgg16.load_state_dict(torch.load(args['pretrained_vgg']))
    model = FCN32s(n_class=21, pretrained_model=vgg16)
    if cuda:
        model = model.cuda()

    optim = torch.optim.SGD(model.parameters(), lr=args['lr'], weight_decay=0.0005, momentum=0.99)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, poly_lr)
    summary_writer = SummaryWriter()

    trainer = Trainer(
        n_class=21,
        cuda=cuda,
        model=model,
        log_dir=log_dir,
        writer=summary_writer,
        optimizer=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        max_iter=args['max_iter'],
        val_interval=args['val_interval'],
        log_interval=20
    )
    
    trainer.train()