import torch
import torch.nn as nn

from image_restoration.I_sub_layer import ISubLayer
from image_restoration.network_usrnet import HyPaNet
from image_restoration.operations_m import *


class Network_Generator(nn.Module):

  def __init__(self, C,  layers, genotype,multi=False):
    super(Network_Generator, self).__init__()
    self.multi = multi
    self._layers = layers
    self.stem = nn.Sequential(
      nn.Conv2d(2, C, 3, padding=1, bias=True)
    )
    self.stem_out = nn.Sequential(
      nn.Conv2d(C, C, 3, padding=1, bias=True),
      nn.Conv2d(C,1, 3, padding=1, bias=True),
    )

    self.tanh = nn.Tanh()
    self.Cell_encoder_1 = OneModel()

    self.Cell_encoder_4 = OneModel() #basic modules

  def forward(self, input):
    b,c,h,w = input.shape
    s1 = self.stem(input)
    s1 = self.Cell_encoder_1(s1)
    # s1= self.Cell_encoder_2(s1)
    # s1= self.Cell_encoder_3(s1)
    s1= self.Cell_encoder_4(s1)
    res = self.stem_out(s1)
    output = input[:,0,:,:].view(b,1,h,w) + self.tanh(res)
    return output


class Network_Generator_total(nn.Module):
  def __init__(self, C,  n_iter, genotype,multi=False):
    super(Network_Generator_total, self).__init__()
    self.ilayer = ISubLayer()
    self.p = Network_Generator(C,2,genotype)
    self.h = HyPaNet(in_nc=1, out_nc=n_iter)
    self.n = n_iter
    self.sigma  = torch.tensor(torch.ones([1,1,1,1],requires_grad=False)*0.01).clone().detach().cuda()
  def forward(self, x, k,sf=1):
    w, h = x.shape[-2:]
    bt = x.shape[0]
    sig, ab = self.h(self.sigma)
    # unfolding
    res =x
    for i in range(self.n):
      res = self.ilayer(res,x, k,sig)
      input = torch.cat((res, ab[:, i :i  + 1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1)
      res = self.p(input)
    return res
# the non-blind
