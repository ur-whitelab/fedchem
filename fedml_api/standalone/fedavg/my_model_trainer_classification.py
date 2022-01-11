import torch
from torch import nn
from tqdm import tqdm
import pickle
from vat import VATLoss
import torch.nn.functional as F
import dgl
from utils import mybceloss
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, roundidx, clientidx):

        fedmid = self.wandbconfig.fedmid
        weightFed = self.wandbconfig.weightFed
        tmpFed = self.wandbconfig.tmpFed
        weightVat = self.wandbconfig.weightVat
        warmupRound = 1

        model = self.model
        model.to(device)
        model.train()
        # init a new model
        if args.dataset == 'qm9':
            from schnet import SchNet
            globalModel = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                                 num_gaussians=50, cutoff=10.0)
            lossCriterion = nn.L1Loss(reduction='none')

        elif args.dataset in ['esol', 'lipo', 'freesolve']:
            from myMPNNPredictor import MPNNPredictor
            globalModel = MPNNPredictor(
                node_in_feats=train_data.dataset.dataset.dataset.graphs[0].ndata['h'].shape[1],
                edge_in_feats=train_data.dataset.dataset.dataset.graphs[0].edata['e'].shape[1],
                node_out_feats=64,
                edge_hidden_feats=16,
                num_step_message_passing=3,
                num_step_set2set=3,
                num_layer_set2set=3,
                n_tasks=1
            )
            lossCriterion = nn.MSELoss(reduction='none')
        elif args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                              'ToxCast', 'HIV', 'PCBA', 'Tox21']:
            from myMPNNPredictor import MPNNPredictor
            globalModel = MPNNPredictor(
                node_in_feats=train_data.dataset.dataset.dataset.graphs[0].ndata['h'].shape[1],
                edge_in_feats=train_data.dataset.dataset.dataset.graphs[0].edata['e'].shape[1],
                node_out_feats=64,
                edge_hidden_feats=16,
                num_step_message_passing=3,
                num_step_set2set=3,
                num_layer_set2set=3,
                n_tasks=train_data.dataset.dataset.dataset.n_tasks
            )
            lossCriterion = mybceloss()
        else:
            raise ValueError('not found dataset')
        globalModel = globalModel.to(device)
        for param_q, param_k in zip(model.parameters(), globalModel.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = True  # not update by gradient

        if args.client_optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        elif args.client_optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        localsteps = args.localStepsPerRound
        tmpstep = 0

        predGAll, predGAll_emb, predGAll_vat = self.globalEpoch(train_data, globalModel, args, device, lossCriterion)
        weight_denomaitor = None
        tbar = tqdm(train_data, mininterval=2, disable=True)
        while tmpstep < localsteps:
            tbarLocalAll = tqdm(range(args.epochs))
            for epoch in enumerate(tbarLocalAll):
                batch_loss = []
                for batch_idx, data in enumerate(tbar):
                    if tmpstep >= localsteps:
                        del globalModel
                        return
                    tmpstep += 1
                    optimizer.zero_grad()
                    if args.dataset == 'qm9':
                        z, pos, batch, labels = data.z.to(device), data.pos.to(device), data.batch.to(
                            device), data.y.to(
                            device)
                        if 'vat' in fedmid:
                            vatloss = VATLoss(framework='geometric', criterion=lossCriterion)  # xi, and eps
                            xCombined = [z, pos, batch]
                            localVATLoss = vatloss(model, xCombined)

                        pred, latentEmb = model(z, pos, batch)
                        index = data.idx
                        predG = predGAll[index]
                        latentEmbG = predGAll_emb[index]
                        globalVATLoss = predGAll_vat[index]


                    elif args.dataset in ['esol', 'lipo', 'freesolve', 'MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                          'ToxCast', 'HIV', 'PCBA', 'Tox21']:
                        smiles, bg, labels, index = data
                        if len(smiles) == 1:
                            # Avoid potential issues with batch normalization
                            continue
                        labels, index = labels.to(device), index.to(device)
                        # prediction = predict(args, model, bg)
                        bg = bg.to(device)
                        node_feats = bg.ndata.pop('h').to(device)
                        edge_feats = bg.edata.pop('e').to(device)

                        if 'vat' in fedmid:
                            vatloss = VATLoss(framework='dgl', criterion=lossCriterion)  # xi, and eps
                            xCombined = [bg, node_feats, edge_feats]
                            localVATLoss = vatloss(model, xCombined)

                        pred, latentEmb = model(bg, node_feats, edge_feats)
                        predG = predGAll[index]
                        latentEmbG = predGAll_emb[index]
                        globalVATLoss = predGAll_vat[index]


                    if roundidx < warmupRound:
                        loss = lossCriterion(pred, labels)
                        loss = loss.mean()
                    else:
                        if fedmid == 'avg':
                            loss = lossCriterion(pred, labels)
                            loss = loss.mean()

                        elif fedmid == 'prox':
                            loss = lossCriterion(pred, labels)
                            loss = loss.mean()
                            loss_regulize = 0
                            params1 = model.named_parameters()
                            params2 = globalModel.named_parameters()
                            dict_params2 = dict(params2)
                            for name1, param1 in params1:
                                if name1 in dict_params2:
                                    loss_regulize = loss_regulize + torch.sum((param1 - dict_params2[
                                        name1].detach()) ** 2)
                            loss = loss + loss_regulize * weightFed

                        elif fedmid == 'vat':
                            lossLocalLabel = lossCriterion(pred, labels)
                            lossLocalVAT = localVATLoss
                            loss = lossLocalLabel.mean() + weightVat * lossLocalVAT.mean()

                        elif fedmid == 'oursFLITvatPLUS':
                            lossGlobalLabel = lossCriterion(predG, labels)
                            lossLocalLabel = lossCriterion(pred, labels)
                            lossLocalVAT = localVATLoss
                            lossGlobalVAT = globalVATLoss

                            weightloss = lossLocalLabel + localVATLoss + self.wandbconfig.ita * torch.relu(lossLocalLabel+lossLocalVAT - lossGlobalLabel.detach()-lossGlobalVAT.detach())
                            factor_ema = 0.8
                            if weight_denomaitor == None:
                                weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()
                            else:
                                weight_denomaitor = factor_ema * weight_denomaitor + (1 - factor_ema) * weightloss.mean(
                                    dim=0, keepdim=True).detach()
                            # if self.wandbconfig.normalizeFocalWeight:
                            loss = (1 - torch.exp(-weightloss / (weight_denomaitor + 1e-7))) ** tmpFed * (
                                    lossLocalLabel+lossLocalVAT)
                            # else:
                            #     loss = (1 - torch.exp(-weightloss)) ** tmpFed * (
                            #         lossLocalLabel+lossLocalVAT)
                            loss = loss.mean()

                        elif fedmid == 'oursFLIT':
                            lossGlobalLabel = lossCriterion(predG, labels)
                            lossLocalLabel = lossCriterion(pred, labels)

                            weightloss = lossLocalLabel+ self.wandbconfig.ita * torch.relu(lossLocalLabel - lossGlobalLabel.detach())
                            factor_ema = 0.8
                            if weight_denomaitor == None:
                                weight_denomaitor = weightloss.mean(dim=0, keepdim=True).detach()
                            else:
                                weight_denomaitor = factor_ema * weight_denomaitor + (1 - factor_ema) * weightloss.mean(
                                    dim=0, keepdim=True).detach()

                            loss = (1 - torch.exp(-weightloss / (weight_denomaitor + 1e-7))) ** tmpFed * (
                                    lossLocalLabel)
                            loss = loss.mean()
                        else:
                            print(fedmid)
                            raise ValueError('no fed method')
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()
                    batch_loss.append(loss.item())
                    del loss


    def globalEpoch(self, train_data, globalModel, args, device, criterion, uncertaintyMethod='loss'):
        if args.dataset=='qm9':
            numData = train_data.dataset.data.idx[-1]
            numTasks = train_data.dataset.data.y.shape[-1]
        else:
            numData = len(train_data.dataset.dataset.dataset)
            numTasks = train_data.dataset.dataset.dataset.labels.shape[-1]
        predGAll = torch.zeros(numData,numTasks).cuda()
        predGAll_loss = torch.zeros(numData,numTasks).cuda()
        predGAll_vat = torch.zeros(numData,numTasks).cuda()
        predGAll_emb = torch.zeros(numData, 128).cuda()
        tbar1 = tqdm(train_data)

        for batch_idx, data in enumerate(tbar1):
            # if batch_idx>5:
            #     break
            if args.dataset == 'qm9':
                z, pos, batch, labels = data.z.to(device), data.pos.to(device), data.batch.to(
                    device), data.y.to(
                    device)
                globalModel.zero_grad()
                vatloss = VATLoss(framework='geometric', criterion=criterion)  # xi, and eps
                x = [z, pos, batch]
                index = data.idx
                predGAll_vat[index.squeeze()] = vatloss(globalModel, x).detach()

                predG, latentEmbG = globalModel(z, pos, batch)
                losstmp = criterion(predG, labels)

                predGAll_loss[index.squeeze()] = losstmp.detach()
                predGAll[index.squeeze()] = predG.detach()
                # predGAll_emb[index.squeeze()] = latentEmbG.clone().detach()

            elif args.dataset in ['esol', 'lipo', 'freesolve', 'MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                  'ToxCast', 'HIV', 'PCBA', 'Tox21']:
                smiles, bg, labels, index = data
                labels, index = labels.to(device), index.to(device)
                # prediction = predict(args, model, bg)
                bg = bg.to(device)
                node_feats = bg.ndata.pop('h').to(device)
                edge_feats = bg.edata.pop('e').to(device)
                # pred, latentEmb = model(bg, node_feats, edge_feats)
                globalModel.zero_grad()
                vatloss = VATLoss(framework='dgl', criterion=criterion)  # xi, and eps
                x = [bg, node_feats, edge_feats]
                predGAll_vat[index.squeeze()] = vatloss(globalModel, x).detach()

                with torch.no_grad():
                    predG, latentEmbG = globalModel(bg, node_feats, edge_feats)

                losstmp = criterion(predG, labels)
                predGAll_loss[index.squeeze()] = losstmp.detach()
                predGAll[index.squeeze()] = predG.detach()
                predGAll_emb[index.squeeze()] = latentEmbG.clone().detach()
        return predGAll, predGAll_emb, predGAll_vat




