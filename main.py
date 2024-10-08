from __future__ import print_function
import os
gpu_index = (1,)
gpu_ids = tuple(gpu_index)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
import argparse
from utils.utils import *
from utils.core_utils import train,get_train_layer
from datasets.dataset_generic import Generic_MIL_Dataset
import torch
import pandas as pd
import numpy as np

def mak_result_dir(args):

    if args.no_inst_cls:
        if args.FilterInst:
            args.results_dir = os.path.join(args.results_dir, 'filter_only',
                                            'smmoth_{}&gamma_{}'.format(args.smoothE, args.gamma))
        else:
            args.results_dir = os.path.join(args.results_dir, 'noinst')
    else:
        if args.training_control:
            if args.FilterInst:
                args.results_dir = os.path.join(args.results_dir, 'all_model&control',
                                                'weight_{}&B_{}&smmoth_{}&gamma_{}'.format(args.bag_weight, args.B,
                                                                                         args.smoothE, args.gamma))
            else:
                args.results_dir = os.path.join(args.results_dir, 'inst_only&control',
                                                'weight_{}&B_{}'.format(args.bag_weight, args.B))
        else:
            if args.FilterInst:
                args.results_dir = os.path.join(args.results_dir, 'all_model',
                                                'weight_{}&B_{}&smmoth_{}&gamma_{}'.format(args.bag_weight, args.B,
                                                                                         args.smoothE, args.gamma))
            else:
                args.results_dir = os.path.join(args.results_dir, 'inst_only',
                                                'weight_{}&B_{}'.format(args.bag_weight, args.B))

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

def main(args):
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                  csv_path='{}/splits_{}.csv'.format(args.split_dir,i))

        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}_{}.csv'.format(start, end, args.stop_rule)
    else:
        save_name = 'summary_{}.csv'.format(args.stop_rule)
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--dataset', type=str, choices=['TCGA', 'BARCS'], default='TCGA',help='which dataset')
parser.add_argument('--results_dir', default='./results_B', help='results directory')
parser.add_argument('--split_dir', type=str, default='712_split', help='manually specify the set of splits to use')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')

# split_fold 
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')

# model
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--model_type', type=str, choices=['my_model', 'clam_model', 'sb_model', 'single_model', 'single_model_sb'],
                    default='my_model', help='type of model (default: my_model)')
parser.add_argument('--exp_code', type=str,default='no_inst', help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task',  type=str,default='task_2_tumor_subtyping', choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')

parser.add_argument('--max_epochs', type=int, default=100, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5, help='weight decay (default: 1e-5)')
parser.add_argument('--stop_rule', type=str, choices=['loss', 'acc', 'auc'], default='loss', help='Which early stop standard to use')

parser.add_argument('--no_inst_cls', action='store_true', default = False, help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=True, help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

### other
parser.add_argument('--training_control', action='store_true',default=False, help='disable no positive inst-classifier train')
parser.add_argument('--FilterInst', action='store_true',default=False, help='whether using smoothing attention strategy')
parser.add_argument('--smoothE', type=int, default=10, help='num of epoch to apply StuFilter')
parser.add_argument('--filter_num', type=int, default=3, help='num of inst to filter')
parser.add_argument('--gamma', type=float, default=0.50, help='filter formula')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = './dataset/'

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(args.seed)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'lr': args.lr,
            'reg': args.reg,
            'opt': args.opt,
            'seed': args.seed,
            'model_type': args.model_type,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'stop_rule': args.stop_rule,
            'training_control': args.training_control,
            'FilterInst': args.FilterInst,
            'smoothE': args.smoothE,
            'filter_num':args.filter_num,
            'gamma':args.gamma
            }

if args.model_type in ['my_model', 'clam_model', 'sb_model','single_model', 'single_model_sb']:
   settings.update({'bag_weight': args.bag_weight,
                    'no_inst_cls': args.no_inst_cls,
                    'subtyping': args.subtyping,
                    'inst_loss_fn': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = os.path.join(base_path, args.dataset, '2_csv/{}.csv').format(args.split_dir),
                            data_dir= os.path.join(base_path, args.dataset, '1_feature'),
                            shuffle = False,
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])
elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path = os.path.join(base_path,args.dataset,'2_csv/{}.csv').format(args.dataset),
                            data_dir= os.path.join(base_path, args.dataset, '1_feature'),
                            shuffle = False,
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1},
                            patient_strat= False,
                            ignore=[])

    if args.model_type in ['my_model', 'clam_model','sb_model', 'single_model', 'single_model_sb']:
        assert args.subtyping
else:
        raise NotImplementedError

args.results_dir = os.path.join(args.results_dir, args.split_dir, args.dataset, args.model_type)
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
mak_result_dir(args)


args.split_dir = os.path.join(base_path, args.dataset, '3_splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({ 'split_dir': args.split_dir})
settings.update({ 'results_dir': args.results_dir})

with open(args.results_dir + '/experiment_{}_{}.txt'.format(args.exp_code, args.stop_rule), 'w') as f:
    for key, val in settings.items():
        print("{}:  {}".format(key, val),file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":

    results = main(args)
    print("finished!")
    print("end script")
