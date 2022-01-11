# FedChem and FLIT(+)
We provide the script for running FLIT on the proposed benchmark. Our code is developed based on [FedML](https://fedml.ai/).

## Requirements
    pytorch==1.7.0
    dgl==0.6.1
    dgllife==0.2.6
    easydict==1.9
    pytorch-geometric==1.7.2
    rdkit=2019.09.3

## Usage
You need a gpu card to run the code. All required datasets will be downloaded with the first running.

1. (optional) We provide the scaffold splitting results for all datasets and name them as scffoldLabel_dataset.pt. You can generate the scaffold label by running
    ```
    python generateScaffoldLabel.py --dataset BBBP
    ``` 
2. Train FedAvg for BBBP by 
    ```
    python main.py --dataset BBBP --fedmid avg --numClient 4 --part_alpha 1
    ```
3. Train FLIT for BBBP by
    ```
    python main.py --dataset BBBP --fedmid oursFLIT --numClient 4 --part_alpha 1
    ```
4. Train FLIT+ for BBBP by 
    ```
    python main.py --dataset BBBP --fedmid oursFLITvatPLUS --numClient 4 --part_alpha 1
    ```
You need to edit the hyperparameter_defaults to run other settings.

You may select datasets by changing dataset from [esol, lipo, freesolve, BACE, BBBP, ClinTox, SIDER, Tox21, qm9]

Tune $\gamma$ by changing tmpFed for FLIT and FLIT+

Tune $\alpha$ by changing ita for FLIT+
# fedchem
# fedchem
# fedchem
# fedchem
# fedchem
# fedchem
