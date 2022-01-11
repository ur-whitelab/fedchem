import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
import argparse

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Lipophilicity', type=str, help='which dataset')
args = parser.parse_args()
datasetName = args.dataset

if datasetName == 'FreeSolv':
    from dgllife.data import FreeSolv
    dataset = FreeSolv()
elif datasetName == 'Lipophilicity':
    from dgllife.data import Lipophilicity
    dataset = Lipophilicity()
elif datasetName == 'ESOL':
    from dgllife.data import ESOL
    dataset = ESOL()
elif datasetName == 'TencentAlchemyDataset':
    from dgllife.data import TencentAlchemyDataset
    dataset = TencentAlchemyDataset()
elif datasetName == 'MUV':
    from dgllife.data import MUV
    dataset = MUV()
elif datasetName == 'BACE':  #
    from dgllife.data import BACE
    dataset = BACE()
elif datasetName == 'BBBP':  #
    from dgllife.data import BBBP
    dataset = BBBP()
elif datasetName == 'ClinTox':
    from dgllife.data import ClinTox
    dataset = ClinTox()
elif datasetName == 'SIDER':  #
    from dgllife.data import SIDER
    dataset = SIDER()
elif datasetName == 'ToxCast':
    from dgllife.data import ToxCast
    dataset = ToxCast()
elif datasetName == 'HIV':
    from dgllife.data import HIV
    dataset = HIV()
elif datasetName == 'PCBA':
    from dgllife.data import PCBA
    dataset = PCBA()
elif datasetName == 'Tox21':  #
    from dgllife.data import Tox21
    dataset = Tox21()
else:
    raise ValueError('Unexpected dataset: {}'.format(datasetName))

N = len(dataset)
print(N)
scaffolds = {}
data_len = len(dataset)
smilesList = dataset.smiles
for ind, smiles in enumerate(smilesList):

    scaffold = generate_scaffold(smiles)
    if scaffold not in scaffolds:
        scaffolds[scaffold] = [ind]
    else:
        scaffolds[scaffold].append(ind)

scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
clslabel = torch.zeros(data_len)
count = 0
for k, v in scaffolds.items():
    clslabel[v] = count
    count = count + 1

torch.save(torch.Tensor(clslabel), 'scffoldLabel_'+datasetName+'.pt')
y = torch.load('scffoldLabel_' + datasetName + '.pt').int()
print('completed')