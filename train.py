import argparse
from models.fg_model import FGModel
from models.optimal_price import OptimalModel
from models.fg_model2 import FGModel2
import torch
from trainer import train
from dataset.fit import FitDataset
from dataset.fit2 import Fit2Dataset
from dataset.optimal import OptimalPriceDataset
from reader import labeled_tasks, users

model_map = {
    'fgModelFitPrice': {
        'model': FGModel,
        'dataset': FitDataset
    },
    'fgModelFitPrice2': {
        'model': FGModel2,
        'dataset': Fit2Dataset
    },
    'fgModelFitOptimalPrice': {
        'model': OptimalModel,
        'dataset': OptimalPriceDataset
    }
}

parser = argparse.ArgumentParser(description='RNN Future predictor')
parser.add_argument('--name', type=str, default='', help='Name of this experiment')
parser.add_argument('--read_old_model', type=str, default=None, help='Path of old model')
parser.add_argument('--user_number', type=int, default=32, help='Only consider nearest K people')
parser.add_argument('--model', type=str, default='fgModelFitPrice', help='Model name')
parser.add_argument('--base_model', type=str, default=None, help='Path of base model')
args = parser.parse_args()

model = model_map[args.model]['model'](user_num=args.user_number)
if args.base_model is not None:
    model.load_base_state_dict(torch.load(args.base_model))
if args.read_old_model is not None:
    model.load_state_dict(torch.load(args.read_old_model))

divider = int(len(labeled_tasks) * .9)
train_dataset = model_map[args.model]['dataset'](args.user_number, labeled_tasks[:divider], users)
test_dataset = model_map[args.model]['dataset'](args.user_number, labeled_tasks[divider:], users)

train(model, name=args.name, train_dataset=train_dataset, test_dataset=test_dataset)

