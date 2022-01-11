import logging
import os
import random
import sys
from datetime import datetime
import numpy as np
import torch
import wandb
import argparse
from data_loader import load_partition_data
from fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from easydict import EasyDict
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BBBP', type=str, help='which dataset, [esol, lipo, freesolve, BACE, BBBP, ClinTox, SIDER, Tox21]')
    parser.add_argument('--fedmid', default='avg', type=str, help='federated method')
    parser.add_argument('--part_alpha', default=1, type=float, help='alpha value for LDA; controling heterogenerity')
    parser.add_argument('--numClient', default=4, type=int, help='number of clients')
    option = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    now = datetime.now()
    dt_string = now.strftime("%Y/%m/%d %H:%M")
    hyperparameter_defaults = dict(
        weightFed = 0.1,
        tmpFed = 0.5,
        ita = 0.5,
        comm_round = 30,
        weightVat = 1,
        normalizeFocalWeight = True,
        seed = 1
    )

    wandb.init(
        project="fedchem",
        config=hyperparameter_defaults
    )
    wandbConfigTmp = wandb.config
    wandbConfig = EasyDict()
    for k,v in wandbConfigTmp.items():
        wandbConfig[k] = v
    wandbConfig.dataset = option.dataset
    wandbConfig.fedmid = option.fedmid
    wandbConfig.part_alpha = option.part_alpha
    wandbConfig.numClient = option.numClient
    args = EasyDict()
    args.numWorker = 4
    args.frequency_of_the_test = 1
    args.client_num_in_total = wandbConfig.numClient
    clientSelectdict = {4:3, 5:3, 6:4, 7:4, 8:6}
    args.client_num_per_round = clientSelectdict[args.client_num_in_total]
    args.client_optimizer = 'adam'
    args.batch_size = 64

    args.partition_method = 'hetero'

    totalsteps = 10000
    args.epochs = 1000

    logger.info(args)
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    if wandbConfig.dataset == 'qm9':
        totalsteps = 100000
        args.client_num_in_total = 8
        args.client_num_per_round = 4

    args.comm_round = wandbConfig.comm_round
    args.localStepsPerRound = int(totalsteps/args.comm_round)

    args.dataset = wandbConfig.dataset
    args.partition_alpha = wandbConfig.part_alpha

    dataset = load_data(args)

    random.seed(wandbConfig.seed)
    np.random.seed(wandbConfig.seed)
    torch.manual_seed(wandbConfig.seed)
    torch.cuda.manual_seed_all(wandbConfig.seed)

    if args.dataset == 'qm9':
        from schnet import SchNet
        model = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                       num_gaussians=50, cutoff=10.0)

    elif args.dataset in ['esol', 'lipo', 'freesolve']:
        from myMPNNPredictor import MPNNPredictor
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

    elif args.dataset in ['BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', 'PCBA', 'Tox21']:
        from myMPNNPredictor import MPNNPredictor
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

    model_trainer = MyModelTrainerCLS(model, args, wandbConfig)
    logging.info(args)
    print(wandbConfig)
    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer, wandbConfig)
    fedavgAPI.train()
