import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim

class FGModel2(nn.Module):
    """
    input: ((N, USER_NUM, (distance, credit, task_capacity)), (N, (x, y))
    user_data: [(distance, credit, task_capacity)] * USER_NUM

    output: (N, price)
    """

    def __init__(self, user_num):
        super(FGModel2, self).__init__()
        activation = nn.SELU
        self.LOSS_LIMIT = 15
        self.user_num = user_num
        self.f1 = nn.Linear(20, 8)
        self.f_a1 = activation()
        self.f2 = nn.Linear(8, 1)
        self.f_a2 = activation()
        self.do = nn.Dropout(0.1)
        self.do2 = nn.Dropout(0.1)
        self.do3 = nn.Dropout(0.1)

        self.h1 = nn.Linear(10, 16)
        self.h_a1 = activation()
        self.h2 = nn.Linear(16, 2)
        self.h_a2 = activation()

        g_temp_size = int(self.user_num * 0.15)
        self.g1 = nn.Linear(self.user_num * 1 + 2, g_temp_size)
        self.g_a1 = activation()
        self.g2 = nn.Linear(g_temp_size, 1)

        self.init_weights()

    def init_weights(self):
        self.f1.weight.data.uniform_(-.1, .1)
        self.f2.weight.data.uniform_(-.1, .1)
        self.g1.weight.data.uniform_(-.1, .1)
        self.g2.weight.data.uniform_(-.1, .1)
        self.h1.weight.data.uniform_(-.1, .1)
        self.h2.weight.data.uniform_(-.1, .1)

        self.f1.bias.data.fill_(0)
        self.f2.bias.data.fill_(0)
        self.g1.bias.data.fill_(0)
        self.g2.bias.data.fill_(0)
        self.h1.bias.data.fill_(0)
        self.h2.bias.data.fill_(0)

    def forward(self, raw_input):
        input, task_input = raw_input
        print(input.data.shape, task_input.data.shape)
        input_sz = input.size()
        assert(input_sz[1] == self.user_num)
        ii = input.resize(input_sz[0] * input_sz[1], input_sz[2])
        f_out = self.f_a2(self.f2(self.do(self.f_a1(self.f1(ii)))))
        f_out_reshaped = f_out.resize(input_sz[0], input_sz[1] * 1)
        h_out = self.h_a2(self.h2(self.do2(self.h_a1(self.h1(task_input)))))
        f_dropout = self.do3(f_out_reshaped)
        #print(f_dropout, h_out)
        g_out = self.g2(self.g_a1(self.g1(torch.cat(
            (f_dropout, h_out), 1))))
        return f_out_reshaped, g_out.resize(input_sz[0])
