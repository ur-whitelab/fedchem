import os
import random
import sys
from datetime import datetime
import numpy as np
import torch
import wandb
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from data_loader import load_partition_data
from fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from easydict import EasyDict
import argparse



def load_data(args):
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128
    else:
        full_batch = False

    data_loader = load_partition_data
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(args)

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]

def custom_model_trainer(args, model):
    return MyModelTrainerCLS(model)


if __name__ == "__main__":
    # test
    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='freesolv',
                            help='esol, lipo, freesolv, BACE, BBBP, ClinTox, SIDER, Tox21, qm9')
    parser.add_argument('-fedmid', type=str, default='oursvatFLITPLUS', help='avg, oursFLIT, oursvatFLITPLUS')
    parser.add_argument('-comm_round', type=int, default=30, help='number of communication rounds in total')
    parser.add_argument('-numClient', type=int, default=4, help='number of clients in totoal')
    parser.add_argument('-tmpFed', type=float, default=0.5, help='temperature scale for weight, search from [0.5, 1, 2]')
    parser.add_argument('-weightReg', type=float, default=1, help='weight for regulization term, set as 1 for FLIT+')
    parser.add_argument('-lambdavat', type=float, default=0.5, help='parameter for weight of vat, search from [0.01, 0.1, 1]')
    parser.add_argument('-xi', type=float, default=0.001, help='xi for vat')
    parser.add_argument('-part_alpha', type=float, default=0.1, help='partition alpha, vary from [0.1, 0.5, 1]')
    parser.add_argument('-seed', type=int, default=0, help='random seed')
    argsinput = parser.parse_args()
    wandb.init(project="fedChem")

    argsinput = argsinput

    args = EasyDict()
    for k, v in vars(argsinput).items():
        args[k] = v
    args.numWorker = 4
    args.frequency_of_the_test = 1
    args.client_num_in_total = argsinput.numClient
    clientSelectdict = {4:3, 5:3, 6:4, 7:4, 8:6}
    args.client_num_per_round = clientSelectdict[args.client_num_in_total]
    args.client_optimizer = 'adam'
    args.batch_size = 64
    args.xi = argsinput.xi
    args.seed = argsinput.seed
    args.partition_method = 'hetero'
    totalsteps = 10000
    args.epochs = 1000

    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

    if argsinput.dataset == 'qm9':
        totalsteps = 1000000
        args.client_num_in_total = 1
        args.client_num_per_round = 1

    args.comm_round = argsinput.comm_round
    args.localStepsPerRound = int(totalsteps/args.comm_round)

    args.dataset = argsinput.dataset
    args.partition_alpha = argsinput.part_alpha

    dataset = load_data(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.dataset == 'qm9':
        from network.myschnet import SchNet
        model = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                       num_gaussians=50, cutoff=10.0)
    elif args.dataset in ['esol', 'lipo', 'freesolv']:
        from network.myMPNNPredictor import MPNNPredictor
        model = MPNNPredictor(
            node_in_feats=dataset[2].dataset.dataset.graphs[0].ndata['h'].shape[1],
            edge_in_feats=dataset[2].dataset.dataset.graphs[0].edata['e'].shape[1],
            node_out_feats=64,
            edge_hidden_feats=16,
            num_step_message_passing=3,
            num_step_set2set=3,
            num_layer_set2set=3,
            n_tasks=1
        )
    elif args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', 'HIV', 'PCBA', 'Tox21']:
        from network.myMPNNPredictor import MPNNPredictor
        model = MPNNPredictor(
            node_in_feats=dataset[2].dataset.dataset.graphs[0].ndata['h'].shape[1],
            edge_in_feats=dataset[2].dataset.dataset.graphs[0].edata['e'].shape[1],
            node_out_feats=64,
            edge_hidden_feats=16,
            num_step_message_passing=3,
            num_step_set2set=3,
            num_layer_set2set=3,
            n_tasks=dataset[2].dataset.dataset.n_tasks
        )
    else:
        raise ValueError('not found dataset')

    model_trainer = MyModelTrainerCLS(model, args, argsinput)
    print(argsinput)
    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer, argsinput)
    print(args)
    fedavgAPI.train()
