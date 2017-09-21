import torch.utils.data as tdata
import torch.utils.data.sampler as tsampler
import torch.optim as toptim
from torch.autograd import Variable
import torch.nn as tnn
import torch
import os
import datetime

def _make_variable(d, *args, **kwargs):
    if type(d) in (list, tuple):
        return [_make_variable(x) for x in d]
    else:
        return Variable(d, *args, **kwargs)

def evaluate(model, test_dataset):
    model.eval()

    criterion = tnn.MSELoss()
    test_sampler = tsampler.RandomSampler(test_dataset)
    test_loader = tdata.DataLoader(test_dataset, batch_size=len(test_dataset), sampler=test_sampler)
    d = next(iter(test_loader))
    data_var = _make_variable(d[0], volatile=True)
    target_var = _make_variable(d[1], volatile=True)
    output_f, output_g = model.forward(data_var)
    loss = criterion(output_g, target_var)

    return loss


def train(model, name, train_dataset, test_dataset):
    train_sampler = tsampler.RandomSampler(train_dataset)

    train_loader = tdata.DataLoader(train_dataset, batch_size=len(train_dataset), sampler=train_sampler)

    optimizer = toptim.Adadelta(model.parameters(), weight_decay=0.001)
    last_loss = 1e10
    last_test_loss = 1e19

    epoch = 0
    while True:
        for i, d in enumerate(train_loader):
            model.train()
            if hasattr(model, 'before_epoch'):
                model.before_epoch(epoch)
            epoch += 1
            data_var = _make_variable(d[0])
            target_var = _make_variable(d[1]).resize(len(d[1]))

            criterion = tnn.MSELoss()
            optimizer.zero_grad()
            output_f, output_g = model.forward(data_var)
            # print('output=', output)
            # print('output shape:', output.shape, 'target_var_shape:', target_var.shape)
            # print(output_g, target_var)
            print(output_g.data.shape, ' target shape:', target_var.data.shape)
            if epoch % 10 == 0:
                print(((output_g - target_var) ** 2).sum() / len(output_g))
            loss = criterion(output_g, target_var)
            loss.backward()
            optimizer.step()
            print('Batch {} | loss {}'.format(epoch, loss.data[0]))
            if (loss.data[0] < model.LOSS_LIMIT and loss.data[0] < last_loss) or epoch % 10 == 0:
                test_loss = evaluate(model, test_dataset)
                test_loss_num = test_loss.data[0]
                print('Evaluated test loss {}'.format(test_loss_num))
                if (test_loss_num < model.LOSS_LIMIT and test_loss_num < last_test_loss) or epoch % 50 == 0:
                    last_test_loss = test_loss_num
                    os.makedirs('saved_model/' + name, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join('saved_model', name,
                                            '{}_loss_{}_test_{}_{}.t7'.format(
                                                epoch, loss.data[0], test_loss_num,
                                                datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
                                            )))
                    torch.save(model,
                               os.path.join('saved_model', name, 'model_{}_loss_{}_test_{}_{}.t7'.format(
                                   epoch, loss.data[0], test_loss_num,
                                   datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
                               )))


            last_loss = loss.data[0]
