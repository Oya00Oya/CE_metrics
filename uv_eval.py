import argparse
import os
import random
import scipy.stats as stats
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
from data.cvs import CreateDataLoader
from utils.uv import cal_tvar

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to sketch dataset')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')

opt = parser.parse_args()
print(opt)

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
####### regular set up end

dataloader = CreateDataLoader(opt.dataroot)

data_iter = iter(dataloader)
i = 0

uv_cums = []
while i < len(dataloader):
    cim = data_iter.next()
    cim = cim.cuda()
    print(f'now {i}')
    uv_cum = cim.transpose(1, 3).contiguous().view(-1, 3) + 1
    uv_cums.append(uv_cum)
    i += 1

uv_cums = torch.cat(uv_cums, 0)

print(f'final uv var score: {cal_tvar(uv_cums)}')

