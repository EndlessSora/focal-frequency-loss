import os
import random

import numpy as np
import torch
import torch.nn.init as init
import torch.utils.data
import torchvision.transforms as transforms

from data import ImageFolderAll, ImageFilelist, ImagePairFilelist


def get_dataloader(opt):
    if opt.dataroot is None:
        raise ValueError('`dataroot` parameter is required for dataset \"%s\"' % opt.dataset)

    if opt.dataset == 'folderall':
        dataset = ImageFolderAll(root=opt.dataroot,
                                 transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]),
                                 return_paths=not opt.is_train)
        nc = 3
    elif opt.dataset == 'filelist':
        assert opt.datalist != '', 'Please specify `--datalist` if you choose `filelist` dataset mode'
        dataset = ImageFilelist(root=opt.dataroot,
                                flist=opt.datalist,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]),
                                return_paths=not opt.is_train)
        nc = 3
    elif opt.dataset == 'pairfilelist':
        assert opt.datalist != '', 'Please specify `--datalist` if you choose `pairfilelist` dataset mode'
        dataset = ImagePairFilelist(root=opt.dataroot,
                                    flist=opt.datalist,
                                    transform=transforms.Compose([
                                        transforms.Resize(opt.imageSize),
                                        transforms.CenterCrop(opt.imageSize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]),
                                    transform_matrix=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    return_paths=not opt.is_train)
        nc = 3
    else:
        raise ValueError('Dataset type is not implemented!')

    assert dataset
    assert nc > 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, drop_last=opt.is_train,
                                             shuffle=opt.is_train, num_workers=int(opt.workers))
    return dataloader, nc


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def print_and_write_log(log_file, message):
    print(message)
    with open(log_file, 'a+') as f:
        f.write('%s\n' % message)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
