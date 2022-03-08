import logging
import os.path as osp
import numpy as np
import torch
# import  as data
import torchvision.transforms as transforms
from torch_geometric.datasets import QM9
import dgl
from dgl.data.utils import Subset

def downloaddataset(datasetName):
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

    elif datasetName in ['esol', 'freesolv', 'lipo', 'MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
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

for i in ["esol","lipo","freesolv","BACE","BBBP","ClinTox","SIDER", "Tox21","qm9"]:
    print("start to download "+i)
    downloaddataset(i)
    print(i+" finished ")