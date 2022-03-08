import logging
import os.path as osp
import numpy as np
import torch
# import  as data
import torchvision.transforms as transforms
from torch_geometric.datasets import QM9
import dgl
from dgl.data.utils import Subset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


def rearrangeLabel(y_train, minClsNum=0):
    if minClsNum>0:
        ycount = 0
    else:
        ycount = 1
    y_trainNew = torch.zeros_like(y_train)
    maxN = y_train.max()+1
    for i in range(maxN):
        tmpInd = y_train == i
        if tmpInd.sum() > minClsNum:
            y_trainNew[tmpInd] = ycount
            ycount += 1
    return y_trainNew


def partition_data(partition, n_nets, alpha, args):
    logging.info("*********partition data***************")
    datasetName = args.dataset
    if datasetName == 'qm9':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
        dataset = QM9(path)
        # atomsLabel.pt; scaffoldLabel.pt
        y = torch.load('data/scaffold_result/scffoldLabel_qm9.pt').int()
        idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
        dataset.data.y = dataset.data.y[:, idx]
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(len(dataset))))
        train_idx = perm[:110000]
        val_idx = perm[110000:]
        y_train = y[train_idx]
        train_dataset, val_dataset = dataset[train_idx], dataset[val_idx]

    elif datasetName in ['esol', 'freesolv', 'lipo','MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21']:
        from dgllife.utils import smiles_to_bigraph
        from functools import partial

        from dgllife.utils import CanonicalAtomFeaturizer
        node_featurizer = CanonicalAtomFeaturizer()
        from dgllife.utils import CanonicalBondFeaturizer
        edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        if datasetName == 'freesolv':
            from data import FreeSolv
            dataset = FreeSolv(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                               node_featurizer=node_featurizer,
                               edge_featurizer=edge_featurizer,
                               n_jobs=1, load=True)
        elif datasetName == 'lipo':
            from data import Lipophilicity
            dataset = Lipophilicity(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                    node_featurizer=node_featurizer,
                                    edge_featurizer=edge_featurizer,
                                    n_jobs=1, load=True)
        elif datasetName == 'esol':
            from data import ESOL
            dataset = ESOL(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer,
                           n_jobs=1, load=True)
        elif datasetName == 'MUV':
            from data import MUV
            dataset = MUV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                          node_featurizer=node_featurizer,
                          edge_featurizer=edge_featurizer,
                          n_jobs=1, load=True)
        elif datasetName == 'BACE':  #
            from data import BACE
            dataset = BACE(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer,
                           n_jobs=1, load=True)
        elif datasetName == 'BBBP':  #
            from data import BBBP
            dataset = BBBP(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer,
                           n_jobs=1, load=True)
        elif datasetName == 'ClinTox':  #
            from data import ClinTox
            dataset = ClinTox(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                              node_featurizer=node_featurizer,
                              edge_featurizer=edge_featurizer,
                              n_jobs=1, load=True)
        elif datasetName == 'SIDER':  #
            from data import SIDER
            dataset = SIDER(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer,
                            n_jobs=1, load=True)
        elif datasetName == 'ToxCast':
            from data import ToxCast
            dataset = ToxCast(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                              node_featurizer=node_featurizer,
                              edge_featurizer=edge_featurizer,
                              n_jobs=1, load=True)
        elif datasetName == 'HIV':
            from data import HIV
            dataset = HIV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                          node_featurizer=node_featurizer,
                          edge_featurizer=edge_featurizer,
                          n_jobs=1, load=True)
        elif datasetName == 'PCBA':
            from data import PCBA
            dataset = PCBA(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=node_featurizer,
                           edge_featurizer=edge_featurizer,
                           n_jobs=1, load=True)
        elif datasetName == 'Tox21':  #
            from data import Tox21
            dataset = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer,
                            n_jobs=1, load=True)
        else:
            raise ValueError('Unexpected dataset: {}'.format(datasetName))

        y = torch.load('data/scaffold_result/scffoldLabel_'+datasetName+'.pt').int()
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(len(dataset))))
        train_idx = perm[:int(len(perm)*0.8)]
        val_idx = perm[int(len(perm)*0.8):]

        y_train = y[train_idx]
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)

    n_train = len(train_dataset)

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        y_train = rearrangeLabel(y_train, 0)
        y_train = y_train.numpy()
        K = len(np.unique(y_train))
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        intRandomseed = 1
        minNumPerClient = n_train / n_nets / 2
        minNumPerClient = args.batch_size if minNumPerClient<args.batch_size else minNumPerClient
        minNumPerClient = 64
        count = 0
        while min_size < minNumPerClient:
            intRandomseed = intRandomseed+1
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.seed(intRandomseed)
                np.random.shuffle(idx_k)
                proportions1 = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions2 = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions1, idx_batch)])
                proportions3 = proportions2 / proportions2.sum()
                proportions = (np.cumsum(proportions3) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            count += 1
            if count>1000:
                break
                raise ValueError('Not valid training')

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    indexlist = []
    index0 = np.where(y_train == 0)[0]
    for k,v in net_dataidx_map.items():
        indexlist = indexlist + v
        print(np.intersect1d(index0, v).shape)


# traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, val_dataset, net_dataidx_map


def load_partition_data(args):
    datasetName = args.dataset
    partition_method, partition_alpha, client_number = args.partition_method,args.partition_alpha, args.client_num_in_total
    train_dataset, val_dataset, net_dataidx_map = partition_data(partition_method,
                                                                 client_number,
                                                                 partition_alpha, args)
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    # trainDL = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.numWorker, drop_last=True)
    # valDL = DataLoader(val_dataset, batch_size=args.bs, num_workers=args.numWorker)
    # collate_fn
    # collate_fn
    if datasetName in ['esol', 'freesolv', 'lipo','MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21']:
        from torch.utils.data import DataLoader
        collate_fn1 = collate_molgraphs
    elif datasetName=='qm9':
        from torch_geometric.data import DataLoader
        collate_fn1 = None
    trainDL = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.numWorker,
                         drop_last=True, pin_memory=False, collate_fn=collate_fn1)
    valDL = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.numWorker, pin_memory=False, collate_fn=collate_fn1)
    logging.info("train_dl_global number = " + str(len(trainDL)))
    logging.info("test_dl_global number = " + str(len(valDL)))
    test_data_num = len(val_dataset)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        dataidxs = torch.Tensor(dataidxs).long()
        # training batch size = 64; algorithms batch size = 32

        if datasetName in ['esol', 'freesolv', 'lipo', 'MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21']:
            localDataset = Subset(train_dataset, dataidxs)
        elif datasetName == 'qm9':
            localDataset = train_dataset[dataidxs]

        train_data_local = DataLoader(localDataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0, drop_last=True, pin_memory=True,collate_fn=collate_fn1)
        test_data_local = train_data_local
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return train_data_num, test_data_num, trainDL, valDL, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, 1
