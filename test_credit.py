from reader import labeled_tasks, users
from dataset.fit2 import Fit2Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import pandas as pd

import os
is_value = os.getenv('VALUE')

res = []
if is_value is not None:
    m = torch.load('./saved_model/IMPORTANT/optimal_k=3_model_12672_loss_0.10014522820711136_test_0.1925690919160843_10:42PM on September 16, 2017.t7').base_model
    output_filename = 'value_credit.csv'
else:
    m = torch.load('./saved_model/IMPORTANT/fit2_model_18836_loss_0.33178260922431946_test_0.47454267740249634_06:56PM on September 16, 2017.t7')
    output_filename = 'price_credit.csv'

for i in range(-3000, 3000):
    def mapper(x):
        x.iloc[0]['credit'] = i * 0.001
        return x
    print(i, ' ', end=',')
    if i % 100 == 0:
        print('')
    ds = Fit2Dataset(32, labeled_tasks, users, mapper)
    loader = DataLoader(ds)
    data = next(iter(loader))
    input_data = [Variable(x) for x in data[0]]
    output = m.forward(input_data)
    res.append(output[1].data[0])

print(res)
pd.DataFrame(res).to_csv(output_filename)
