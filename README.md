# FedChem and FLIT(+)
We provide the script for running FLIT for the proposed benchmark. Our code is developed based on FedML (https://fedml.ai/)

We provide the scaffold splitting results for all datasets and name them as scffoldLabel_dataset.pt

## Requirements
    dgl==0.6.1
    dgllife==0.2.6
    easydict==1.9
    pytorch-geometric==1.7.2
    rdkit=2019.09.3
    pytorch=1.8.1

## Dataset Download (optional)
All dataset will be downloaded with first run or you can download them by
```angular2html
python downloadDataset.py
```

## Usage
You need a gpu to run the code. We log the results with wandb.
1. Train FedAvg for FreeSolv with heterogeneous partatition 0.1 by 
```
python main.py -dataset esol -fedmid avg -part_alpha 0.1
```
2. Train FLIT+ (gamma(tmpFed)=0.5 and lambda(lambdavat)=0.01) for FreeSolv with heterogeneous partatition 0.1 by
```
python main.py -dataset esol -fedmid oursvatFLITPLUS -tmpFed 0.5 -lambdavat 0.01 -part_alpha 0.1
```

## Citation
Cite our paper
```angular2html
@article{zhu2021federated,
  title={Federated Learning of Molecular Properties with Graph Neural Networks in a Heterogeneous Setting},
  author={Zhu, Wei and White, Andrew and Luo, Jiebo},
  journal={Available at SSRN 4002763},
  year={2021}
}
```
