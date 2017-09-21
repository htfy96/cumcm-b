from models.fg_model2 import FGModel2
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim
import random


class OptimalModel(nn.Module):
    def __init__(self, user_num):
        super(OptimalModel, self).__init__()
        self.LOSS_LIMIT = 0.3
        self.base_model = FGModel2(user_num)
        self.k = Variable(torch.FloatTensor(1))
        self.a = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        self.k.data.set_(torch.FloatTensor([3]))

    def before_epoch(self, epoch):
        if epoch < 5000:
            self.k.requires_grad = False
        else:
            self.k.requires_grad = epoch % 3 > 0

    def load_base_state_dict(self, state_dict):
        self.base_model.load_state_dict(state_dict)

    def forward(self, input):

        base_input = input[:2]
        price = input[2].resize(len(input[2]))
        f_out, g_out = self.base_model.forward(base_input)
        #print('g_out,price=', g_out.data, price.data)
        #print('delta', (price - g_out).data)
        #print('k delta=',(self.k * (price - g_out)).data)
        #print('a=', self.a(self.k * (price - g_out)).data)
        if random.random() < 0.1:
            print('k=', self.k)
            print('price-g_out=', price-g_out)
        return g_out, self.a(self.k * (price - g_out))
