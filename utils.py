import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
# from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,balanced_accuracy_score
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, SpectralEmbedding
import dgl

def tsneDemoXY(train_feature, cls_indicator_datacenter):
    X_embedded = TSNE(n_jobs=8).fit_transform(train_feature)

    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.cm.jet
    # make the scatter
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=cls_indicator_datacenter, s=20, cmap=cmap)
    # ax.scatter(center_emb[:, 0], center_emb[:, 1], c=np.arange(len(center)), s=230, cmap=cmap, marker='*')
    #
    # ax.legend(cls_indicator_datacenter)
    plt.legend(*scatter.legend_elements())
    plt.show()

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

class mybceloss(torch.nn.Module):

    def __init__(self):
        super(mybceloss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        target = torch.sigmoid(target)
        loss = target * torch.log(pred + 1e-7) + (1 - target) * torch.log(
            (1 - pred) + 1e-7)
        return -loss