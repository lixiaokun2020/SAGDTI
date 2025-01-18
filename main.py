import random
import numpy as np
import torch.nn.init as init
import torch
from copy import deepcopy
from config import SAGDTI_config
from models import SAGDTI
from argparse import ArgumentParser
from dataloader import  LoadData
from helper import *
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tqdm import  tqdm
from sklearn.metrics import  accuracy_score


np.random.seed(1234)

#定义超参数
parser = ArgumentParser(description='SAGDTI Training.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--task', choices=['bindingdb', 'davis', 'kiba'],
                    default='kiba', type=str, metavar='TASK',
                    help='Task name. Could be bindingdb, davis and kiba.')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--nd_layer', default= [1,2,3,4,5,6,7], type=int,
                    nargs= '+', help= 'The number of the molecular transformer encoders for drugs' )
parser.add_argument('--nt_layer', default= [1,2,3,4,5,6,7], type=int,
                    nargs= '+', help= 'The number of the molecular transformer encoders for targets' )
parser.add_argument('--GATlayer', default= [1,2,3,4,5,6,7], type=int,
                    nargs= '+', help= 'The number of the graph attention network for targets' )
parser.add_argument('--filters', type=int, nargs='+', default=[32, 64, 96],
    help='Space seperated list of the number of filters.')
parser.add_argument('--problem_type', type=int, default=2,
                    help='Type of the prediction problem (1-4)')
parser.add_argument('--problem_type', type=int, default=2,
    help='Type of the prediction problem (1-4)')

config = SAGDTI_config()
args = parser.parse_args()

#获取数据集
def get_dataset(task_name):
    if task_name.lower() == 'kiba':
        return './dataset/KIBA'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'

#权重初始化
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        if isinstance(m, nn.LSTM):
            init.orthogonal_(m.all_weights[0][0])
            init.orthogonal_(m.all_weights[0][1])
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)


#交叉验证
def get_random_folds(number, foldcount):
    folds = []
    indices = set(range(number))
    foldsize = number / foldcount
    leftover = number % foldcount
    for i in range(foldcount):
        sample_size = foldsize
        if leftover > 0:
            sample_size += 1
            leftover -= 1
        fold = random.sample(indices, int(sample_size))
        indices = indices.difference(fold)
        folds.append(fold)

    #保证样本不出现丢失
    # assert stuff
    foldunion = set([])
    for find in range(len(folds)):
        fold = set(folds[find])
        assert len(fold & foldunion) == 0, str(find)
        foldunion = foldunion | fold
    assert len(foldunion & set(range(number))) == number

    return folds

#New_drug setting
def get_drugwise_folds(label_row_inds, drugcount, foldcount):
    assert len(np.array(label_row_inds).shape) == 1, 'label_row_inds should be one dimensional array'
    row_to_indlist = {}
    rows = sorted(list(set(label_row_inds)))
    for rind in rows:
        alloccs = np.where(np.array(label_row_inds) == rind)[0]
        row_to_indlist[rind] = alloccs
    drugfolds = get_random_folds(drugcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        drugfold = drugfolds[foldind]
        for drugind in drugfold:
            fold = fold + row_to_indlist[drugind].tolist()
        folds.append(fold)
    return folds

# New-target setting
def get_targetwise_folds(label_col_inds, targetcount, foldcount):
    assert len(np.array(label_col_inds).shape) == 1, 'label_col_inds should be one dimensional array'
    col_to_indlist = {}
    cols = sorted(list(set(label_col_inds)))
    for cind in cols:
        alloccs = np.where(np.array(label_col_inds) == cind)[0]
        col_to_indlist[cind] = alloccs
    target_ind_folds = get_random_folds(targetcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        targetfold = target_ind_folds[foldind]
        for targetind in targetfold:
            fold = fold + col_to_indlist[targetind].tolist()
        folds.append(fold)
    return folds

#计算 drug-target pair interaction
def prepare_interaction_pairs(XD, XT, DTI, rows, cols):
    dataset = [[]]
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        dataset[pair_ind].append(np.array(drug, dtype=np.float32))
        target = XT[cols[pair_ind]]
        dataset[pair_ind].append(np.array(target, dtype=np.float32))
        dataset[pair_ind].append(np.array(DTI[rows[pair_ind], cols[pair_ind]], dtype=np.float32))
        if pair_ind < len(rows) - 1:
            dataset.append([])
    return dataset

#训练
def train(train_loader, model, ARGS, feature, graph, graph2adj):
    model.train()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    with tqdm(train_loader) as t:
        for drug_SMILES, target_protein, affinity in t:
            drug_SMILES = torch.Tensor(drug_SMILES)
            target_protein = torch.Tensor(target_protein)
            affinity = torch.Tensor(affinity)
            optimizer.zero_grad()

            affinity = Variable(affinity)           #.cuda(), 使用GPU
            pre_affinity = model(drug_SMILES, target_protein, feature, graph, graph2adj)
            loss = loss_func(pre_affinity, affinity)
            accuracy = accuracy(pre_affinity, affinity)
            auc = auc(pre_affinity,affinity)
            loss.backward()
            optimizer.step()
            t.set_postfix(train_loss=loss.item(), accuracy=accuracy, auc = auc)
    return model

#测试
def test(model,test_loader,ARGS, feature, graph, graph2adj):
    model.eval()
    loss_func = nn.BCE()
    affinities = []
    pre_affinities = []
    loss_d=0
    loss_t=0
    with torch.no_grad():
        for i,(drug_SMILES, target_protein, affinity) in enumerate(test_loader):
            pre_affinity = model(drug_SMILES, target_protein, ARGS, feature, graph, graph2adj)
            pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
            affinities += affinity.cpu().detach().numpy().tolist()
        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)
        loss = loss_func(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        accuracy = accuracy_score(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        auc = roc_auc_score(np.int32(affinities >7), pre_affinities)
    return auc, accuracy, loss

#获取数据集路径
path  = get_dataset(args.task)

#加载数据
dataset = LoadData(
                  setting_no=config['setting_no'],  ##BUNU ARGS A EKLE
                  seqlen=config['max_protein_seq'],
                  smilen=config['max_drug_seq'],
                  )


args.charseqset_size = dataset.charseqset_size
args.charsmiset_size = dataset.charsmiset_size

#获取药物、蛋白质特征
XD,XT,DTI = dataset.parse_data(path)

XD = np.asarray(XD)
XT = np.asarray(XT)
DTI = np.asarray(DTI)

#药物数量
drugcount = XD.shape[0]
print(drugcount)

#蛋白质数量
targetcount = XT.shape[0]
print(targetcount)

#获取interaction的index
label_row_inds, label_col_inds = np.where(DTI == 1)

foldcount = 6 #train data (train + validation ) + test data = 5 + 1

#experimental settings
if args.problem_type == 1:
    nfolds = get_random_folds(len(label_row_inds), foldcount)
if args.problem_type == 2:
    nfolds = get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount)
if args.problem_type == 3:
    nfolds = get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount)

test_data = nfolds[5]
train_data = nfolds[0:5]

foldinds = len(train_data)

val_sets = []
train_sets = []
test_sets = []

#训练集、验证集和测试集划分
for folding in range(foldinds):
    val_fold = train_data[folding]
    val_sets.append(val_fold)
    otherfolds = deepcopy(train_data)
    otherfolds.pop(folding)
    otherfoldsinds = [item for sublist in otherfolds for item in sublist]
    train_sets.append(otherfoldsinds)
    test_sets.append(test_data)
    print("val set", str(len(val_sets)))
    print("train set", str(len(train_sets)))


#grid search
parameter1 = args.nd_layer
parameter2 = args.nt_layer
parameter3 = args.GATlayer
parameter4 = args.filters

w = len(val_sets)
h = len(parameter1) * len(parameter2) & len(parameter3) * len(parameter4)

all_predictions = [[0 for x in range(w)] for y in range(h)]
all_losses = [[0 for x in range(w)] for y in range(h)]

print("five-fold cv, here we go!")
for folding in range(len(test_sets)):

    valinds = val_sets[folding]
    labeledinds = train_dataset[folding]

    trrows = label_row_inds[labeledinds]
    trcols = label_col_inds[labeledinds]

    train_dataset = prepare_interaction_pairs(XD, XT, DTI, trrows, trcols)

    terows = label_row_inds[valinds]
    tecols = label_col_inds[valinds]

    test_dataset = prepare_interaction_pairs(XD, XT, DTI, terows, tecols)

    #加载biological interactive information
    config.train_path = "./biological interactive information/graph/dti/train00{}.txt".format(folding + 1)
    config.test_path = "./biological interactive information/graph/dti/test00{}.txt".format(folding + 1)



    graph = load_graph(config['graph_path'])
    graph2adj = get_adj(graph)

    features, labels, idx_train, idx_test = load_data(config)

    pointer = 0

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True)
    print("Let's do the grid search for SAGDTI training")
    for paramvalue1 in parameter1:
        for paramvalue2 in parameter2:
            for paramvalue3 in parameter3:
                for paramvalue4 in parameter4:
                    model = SAGDTI(paramvalue1, paramvalue2, paramvalue3, paramvalue4, config)
                    perf_list = []
                    for epoch in range(100):
                        model = train(train_loader, model, features, graph, graph2adj)
                        auc, accuracy, loss = test(model, test_loader, features, graph, graph2adj)
                        perf_list.append(auc)
                    all_predictions[pointer][folding] = accuracy
                    all_losses[pointer][folding] = loss
                    pointer += 1

bestperf = -float('Inf')
bestpointer = None

best_param_list = []
##Take average according to folds, then chooose best params
pointer = 0
for param1value in parameter1:
    for param2value in parameter2:
        for param3value in parameter3:
            for param4value in parameter4:
                avgperf = 0.
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= len(val_sets)

                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1value, param2value, param3value, param4value]
                pointer += 1


testperfs = []
testloss = []
testaupr = []
avgperf = 0.

for test_foldind in range(len(test_sets)):
    foldperf = all_predictions[test_foldind]
    foldloss = all_losses[test_foldind]
    testperfs.append(foldperf)
    testloss.append(foldloss)
    avgperf += foldperf

avgperf = avgperf / len(test_sets)
avgloss = np.mean(testloss)
perf_std = np.std(testperfs)
loss_std = np.std(testloss)
avg_auc = np.mean(testperfs)
auc_std = np.std(testperfs)
avg_aupr = np.mean(testaupr)


print(best_param_list)