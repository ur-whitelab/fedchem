import copy
import logging
from tqdm import tqdm
import numpy as np
import torch
import wandb
from datetime import datetime
from client import Client
from sklearn.metrics import roc_auc_score


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer, wandbConfig):
        self.device = device
        self.args = args
        self.wandbConfig = wandbConfig
        self.fedmid = self.wandbConfig.fedmid
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num



        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        # self._instanciate_opt()
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        if self.args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                 'ToxCast', 'HIV', 'PCBA', 'Tox21']:
            self.bestTest = -1e7
            self.bestVal = -1e7
            self.bestQm9EveryTask = []
        else:
            self.bestTest = 1e7
            self.bestVal = 1e7
            self.bestQm9EveryTask = []

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def _instanciate_opt(self):
        self.opt = torch.optim.Adam(
                # self.model_global.parameters(), lr=self.args.server_lr
                self.model_trainer.model.parameters(), lr=self.wandbConfig.weightFed,
                # momentum=0.9 # for fedavgm
                # eps = 1e-3 for adaptive optimizer
            )

    def train(self):
        if self.fedmid == 'opt':
            for round_idx in range(self.args.comm_round):
                w_global = self.model_trainer.get_model_params()
                logging.info("################ Communication round : {}".format(round_idx))

                w_locals = []

                """
                for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                """
                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                       self.args.client_num_per_round)
                logging.info("client_indexes = " + str(client_indexes))

                for idx, client in enumerate(self.client_list):
                    # update dataset
                    client_idx = client_indexes[idx]

                    client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                                self.test_data_local_dict[client_idx],
                                                self.train_data_local_num_dict[client_idx])

                    # train on new dataset
                    w = client.train(w_global, round_idx, client_idx)
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                    # loss_locals.append(copy.deepcopy(loss))
                    # logging.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))

                # reset weight after standalone simulation
                self.model_trainer.set_model_params(w_global)
                # update global weights
                w_avg = self._aggregate(w_locals)
                # server optimizer
                self.opt.zero_grad()
                opt_state = self.opt.state_dict()
                self._set_model_global_grads(w_avg)
                self._instanciate_opt()
                self.opt.load_state_dict(opt_state)
                self.opt.step()
                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateGlobal(round_idx)



        else:
            w_global = self.model_trainer.get_model_params()
            for round_idx in range(self.args.comm_round):

                logging.info("################Communication round : {}".format(round_idx))

                w_locals = []

                """
                for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
                Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
                """
                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                       self.args.client_num_per_round)
                logging.info("client_indexes = " + str(client_indexes))

                for idx, client in enumerate(self.client_list):
                    # update dataset
                    client_idx = client_indexes[idx]
                    print('Start Training: round_' + str(round_idx) + '_client_' + str(client_idx))
                    client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                                self.test_data_local_dict[client_idx],
                                                self.train_data_local_num_dict[client_idx])

                    # train on new dataset
                    w = client.train(w_global, round_idx, client_idx)
                    # self.logger.info("local weights = " + str(w))
                    w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

                    # update global weights
                w_global = self._aggregate(w_locals)
                self.model_trainer.set_model_params(w_global)

                # test results
                # at last round
                if round_idx % self.args.frequency_of_the_test == 0:
                    self.validateGlobal(round_idx)

    def _set_model_global_grads(self, new_state):
        new_model = copy.deepcopy(self.model_trainer.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(
                self.model_trainer.model.parameters(), new_model.parameters()
            ):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.model_trainer.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.model_trainer.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        self.model_trainer.set_model_params(new_model_state_dict)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def validateGlobal(self, epoch):
        epoch = int(epoch*self.args.localStepsPerRound)
        model = self.model_trainer.model
        tbar = tqdm(self.test_global)
        device = self.device
        model.to(device)
        model.eval()
        predList = []

        labelList = []
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar):
                if self.args.dataset == 'qm9':
                    z, pos, batch, y = data.z.to(device), data.pos.to(device), data.batch.to(device), data.y.to(
                        device)
                    pred, latentEmb = model(z, pos, batch)
                    # mae = (pred.view(-1) - y[:, self.args.target]).abs()
                    predList.append(pred.squeeze())
                    labelList.append(y.squeeze())
                elif self.args.dataset in ['esol', 'lipo', 'freesolv']:
                    smiles, bg, labels, masks = data
                    labels, masks = labels.to(device), masks.to(device)
                    # prediction = predict(args, model, bg)
                    bg = bg.to(device)
                    node_feats = bg.ndata.pop('h').to(device)
                    edge_feats = bg.edata.pop('e').to(device)
                    pred, latentEmb = model(bg, node_feats, edge_feats)
                    predList.append(pred.squeeze())
                    labelList.append(labels.squeeze())
                elif self.args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21']:
                    smiles, bg, labels, masks = data
                    labels, masks = labels.to(device), masks.to(device)
                    # prediction = predict(args, model, bg)
                    bg = bg.to(device)
                    node_feats = bg.ndata.pop('h').to(device)
                    edge_feats = bg.edata.pop('e').to(device)
                    pred, latentEmb = model(bg, node_feats, edge_feats)
                    predList.append(torch.sigmoid(pred).squeeze())
                    labelList.append(labels.squeeze())
                # z, pos, batch, y = data.z.to(device), data.pos.to(device), data.batch.to(device), data.y.to(device)
                # pred, _ = model(z, pos, batch)

                # loss = 1 * (mae.mean())
                tbar.set_description('Round: {:d} Iter: {:d} / {:d}'.format(epoch, batch_idx, len(self.test_global)))
            # predAll = torch.cat(predList).flatten()
            # labelAll = torch.cat(labelList).flatten()
            if self.args.dataset == 'qm9':
                valSize = 10000
            else:
                valSize = int(0.5*len(self.test_global.dataset))

            torch.random.manual_seed(123)
            indexShuffle = torch.randperm(len(self.test_global.dataset))
            if predList[-1].size()==torch.Size([]):
                predList[-1] = predList[-1].unsqueeze(0)
                labelList[-1] = labelList[-1].unsqueeze(0)
            predAll = torch.cat(predList, dim=0)[indexShuffle]
            labelAll = torch.cat(labelList, dim=0)[indexShuffle]
            if self.args.dataset == 'qm9':
                # predAll=predAll.flatten()
                # labelAll=labelAll.flatten()
                # resultsNoMean = (predAll - labelAll).abs().mean(dim=0)
                maeAll = (predAll - labelAll).abs()
                valResult = maeAll[:valSize].mean().item()
                valResultStd = maeAll[:valSize].std().item()
                testResult = maeAll[valSize:].mean().item()
                testResultStd = maeAll[valSize:].std().item()
                resultsNoMean = (predAll - labelAll).abs().mean(dim=0)
                metricName = ' mae '
            elif self.args.dataset in ['esol', 'lipo', 'freesolv']:
                predAll=predAll.flatten()
                labelAll=labelAll.flatten()
                mseAll = (predAll - labelAll)**2
                valResult = torch.sqrt(mseAll[:valSize].mean()).item()
                valResultStd = mseAll[:valSize].std().item()
                testResult = torch.sqrt(mseAll[valSize:].mean()).item()
                testResultStd = mseAll[valSize:].std().item()
                metricName = ' rmse '
            elif self.args.dataset in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER',
                                                    'ToxCast', 'HIV', 'PCBA', 'Tox21']:


                predVal = predAll[:valSize]
                labelVal = labelAll[:valSize]
                predTest = predAll[valSize:]
                labelTest = labelAll[valSize:]
                valResultsList = []
                testResultsList = []
                if predAll.size().__len__() == 1:
                    valResultsList.append(roc_auc_score(labelVal.cpu(), predVal.cpu()))
                    testResultsList.append(roc_auc_score(labelTest.cpu(), predTest.cpu()))
                else:
                    for itask in range(predAll.shape[1]):
                        valResultsList.append(roc_auc_score(labelVal.cpu()[:, itask], predVal.cpu()[:, itask]))
                        testResultsList.append(roc_auc_score(labelTest.cpu()[:, itask], predTest.cpu()[:, itask]))


                valResult = torch.Tensor(valResultsList).mean().item()
                valResultStd = 0
                testResult = torch.Tensor(testResultsList).mean().item()
                testResultStd = 0
                metricName = ' auc '

            if metricName==' auc ':
                if valResult > self.bestVal:
                    self.bestVal = valResult
                    self.bestTest = testResult
            else:
                if valResult < self.bestVal:
                    self.bestVal = valResult
                    self.bestTest = testResult
                    if self.args.dataset == 'qm9':
                        self.bestQm9EveryTask = resultsNoMean.tolist()
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            curValResult = 'cur Val Steps: ' + str(epoch) +  metricName  + str(valResult) +' std ' + str(valResultStd) + '\n'
            curTestResult = 'cur Test Steps: ' + str(epoch) +  metricName  + str(testResult) + ' std ' + str(testResultStd) + '\n'
            bestValResult = 'best Val Steps: ' + str(epoch) +  metricName  + str(self.bestVal) + '\n'
            bestTestResult = 'best Test Steps: ' + str(epoch) +  metricName  + str(self.bestTest) + '\n'

            # stats = {'val_mae': valResult, 'test_mae': testResult}
            stats = {"Val": valResult, "Test": testResult,"bestVal": self.bestVal, "bestTest": self.bestTest, "round": epoch, "results":self.bestQm9EveryTask}
            wandb.log(stats)
            # wandb.log({"TestMae": testResult, "steps": epoch})
            # wandb.log({"Valmae": valResult, "steps": epoch})
            res = dt_string + '\n' + curValResult + curTestResult + bestValResult + bestTestResult  + 'detail results'+str(self.bestQm9EveryTask) + '\n'
            logging.info(res)
