import torch
from torch import nn
from torch.nn import functional as F


class SReLU(nn.Module):
  def __init__(self, units):
    super().__init__()
    self.units=units
    self.t_left=nn.Parameter(torch.zeros(units))
    self.a_left=nn.Parameter(torch.rand(units))
    self.t_right=nn.Parameter(torch.ones(units))
    self.a_right=nn.Parameter(torch.rand(units))
    self.register_parameter("t_left",self.t_left)
    self.register_parameter("t_right",self.t_right)
    self.register_parameter("a_left",self.a_left)
    self.register_parameter("a_right",self.a_right)
    


  def forward(self, inputs):
    l=[]
    for i in range(self.units):
      y_left=torch.where(inputs<=self.t_left[i], self.a_left[i]*(inputs-self.t_left[i]), 0)
      y_right=torch.where(inputs>=self.t_right[i], self.a_right[i]*(inputs-self.t_right[i]), 0)
      ris=torch.abs(y_left)+torch.abs(y_right)  #find all the non-zero elements  
      center=torch.where(ris==0,inputs,0) #all the elements that are not out of the boundaries -> inputs
      l.append(y_left+y_right+center)
    b = torch.Tensor(self.units, inputs.shape[0], inputs.shape[1])
    b=torch.cat(l)
    return b