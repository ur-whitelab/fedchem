from dgllife.utils import smiles_to_bigraph
from functools import partial

from dgllife.utils import CanonicalAtomFeaturizer

node_featurizer = CanonicalAtomFeaturizer()
from dgllife.utils import CanonicalBondFeaturizer

edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
datasetName = 'freesolve'
if datasetName == 'freesolve':
    from dgllife.data import FreeSolv

    dataset = FreeSolv(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                       node_featurizer=node_featurizer,
                       edge_featurizer=edge_featurizer,
                       n_jobs=1, load=True)
elif datasetName == 'lipo':
    from dgllife.data import Lipophilicity

    dataset = Lipophilicity(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer,
                            n_jobs=1, load=True)
elif datasetName == 'esol':
    from dgllife.data import ESOL

    dataset = ESOL(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                   node_featurizer=node_featurizer,
                   edge_featurizer=edge_featurizer,
                   n_jobs=1, load=True)
elif datasetName == 'MUV':
    from dgllife.data import MUV

    dataset = MUV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                  node_featurizer=node_featurizer,
                  edge_featurizer=edge_featurizer,
                  n_jobs=1, load=True)
elif datasetName == 'BACE':  #
    from dgllife.data import BACE

    dataset = BACE(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                   node_featurizer=node_featurizer,
                   edge_featurizer=edge_featurizer,
                   n_jobs=1, load=True)
elif datasetName == 'BBBP':  #
    from dgllife.data import BBBP

    dataset = BBBP(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                   node_featurizer=node_featurizer,
                   edge_featurizer=edge_featurizer,
                   n_jobs=1, load=True)
elif datasetName == 'ClinTox':  #
    from dgllife.data import ClinTox

    dataset = ClinTox(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                      node_featurizer=node_featurizer,
                      edge_featurizer=edge_featurizer,
                      n_jobs=1, load=True)
elif datasetName == 'SIDER':  #
    from dgllife.data import SIDER

    dataset = SIDER(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                    node_featurizer=node_featurizer,
                    edge_featurizer=edge_featurizer,
                    n_jobs=1, load=True)
elif datasetName == 'ToxCast':
    from dgllife.data import ToxCast

    dataset = ToxCast(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                      node_featurizer=node_featurizer,
                      edge_featurizer=edge_featurizer,
                      n_jobs=1, load=True)
elif datasetName == 'HIV':
    from dgllife.data import HIV

    dataset = HIV(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                  node_featurizer=node_featurizer,
                  edge_featurizer=edge_featurizer,
                  n_jobs=1, load=True)
elif datasetName == 'PCBA':
    from dgllife.data import PCBA

    dataset = PCBA(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                   node_featurizer=node_featurizer,
                   edge_featurizer=edge_featurizer,
                   n_jobs=1, load=True)
elif datasetName == 'Tox21':  #
    from dgllife.data import Tox21

    dataset = Tox21(smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                    node_featurizer=node_featurizer,
                    edge_featurizer=edge_featurizer,
                    n_jobs=1, load=True)
else:
    raise ValueError('Unexpected dataset: {}'.format(datasetName))