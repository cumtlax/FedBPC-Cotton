import torch
import numpy as np
import cvxpy as cvx

from typing import Iterator
from collections import Counter
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import MSELoss, Parameter
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from entities.base import Client, Server


class FedPACClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.mse_criterion = MSELoss().to(args["device"])
        self.clf_keys = None
        self.label_distribution = torch.zeros(args["num_classes"])
        self.label_prob = torch.zeros(args["num_classes"])
        self.local_protos = None

    def set_rep_params(self, new_params: Iterator[Parameter]):
        """
        Set local model's representation parameters as new parameters.
        :param new_params: The new parameters to be set.
        """
        rep_params = [param for name, param in self.model.named_parameters() if name not in self.clf_keys]
        for new_param, local_param in zip(new_params, rep_params):
            local_param.data = new_param.data.clone()

    def set_clf_params(self, new_params: Iterator[Parameter]):
        """
        Set local model's classifier parameters as new parameters.
        :param new_params: The new parameters to be set.
        """
        clf_params = [param for name, param in self.model.named_parameters() if name in self.clf_keys]
        for new_param, local_param in zip(new_params, clf_params):
            local_param.data = new_param.data.clone()

    def train_with_protos(self, global_protos):
        """
        Perform decoupled local training with global prototypes.
        First, train the classifier for clf_epoch. Then, train the representation for rep_epoch.
        :param global_protos: The global prototypes.
        """
        self.model.train()

        # get local prototypes before local training
        local_protos = self.get_local_protos(self.model)

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        # -------------------Start the training of classifier-------------------
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["clf_lr"],
                        weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["clf_epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # -------------------Start the training of representation-------------------
        for name, param in self.model.named_parameters():
            if name in self.clf_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args["rep_lr"],
                        weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])

        for epoch in range(self.args["rep_epochs"]):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                outputs, features = self.model(inputs, True)
                loss_sup = self.criterion(outputs, labels)
                loss_proto = 0.0

                if len(global_protos) > 0:
                    # the global prototypes is empty at the first round
                    protos = features.clone().detach()
                    for i in range(len(labels)):
                        label = labels[i].item()
                        if label in global_protos:
                            protos[i] = global_protos[label].detach()
                        else:
                            protos[i] = local_protos[label].detach()
                    loss_proto = self.mse_criterion(features, protos)
                loss = loss_sup + self.args["lambda"] * loss_proto

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            # update local prototypes after local training
            self.local_protos = self.get_local_protos(self.model)

    def get_local_protos(self, model):
        """
        Get the local prototypes for each class.
        :param model: The local model.
        :return: The average of prototypes for each class.
        """
        proto_dict = {}
        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])
                outputs, features = model(inputs, True)
                protos = features.clone().detach()
                for i in range(len(labels)):
                    label = labels[i].item()
                    if label in proto_dict:
                        proto_dict[label].append(protos[i, :])
                    else:
                        proto_dict[label] = [protos[i, :]]

        # get the average of prototypes for each class
        proto_mean = {}
        for label, proto_list in proto_dict.items():
            proto_mean[label] = torch.mean(torch.stack(proto_list), dim=0)

        return proto_mean

    def statistics_extraction(self):
        g_params = self.model.state_dict()[self.clf_keys[0]]
        d = g_params.shape[1]
        feature_dict = {}
        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)

        with torch.no_grad():
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])
                outputs, features = self.model(inputs, True)
                protos = features.clone().detach().cpu()

                for i in range(len(labels)):
                    label = labels[i].item()
                    if label in feature_dict:
                        feature_dict[label].append(protos[i, :])
                    else:
                        feature_dict[label] = [protos[i, :]]
        for label, feature_list in feature_dict.items():
            feature_dict[label] = torch.stack(feature_list).to(torch.float64)

        py = self.label_prob.to(torch.float64)
        v = 0
        h_ref = torch.zeros((self.args["num_classes"], d), dtype=torch.float64)

        for label in range(self.args["num_classes"]):
            if label in feature_dict:
                label_feature = feature_dict[label]
                label_feature_mu = label_feature.mean(dim=0)
                h_ref[label] = py[label] * label_feature_mu
                # Equivalent to p E||z||² - p²||E z||², without subtracting
                # nearly equal second moments for constant/low-variance data.
                within = (label_feature - label_feature_mu).square().sum(dim=1).mean()
                v += (py[label] * within + py[label] * (1 - py[label]) *
                      label_feature_mu.square().sum()).item()

        v /= len(self.train_set)

        return v, h_ref

    def update_label_distribution(self):
        distribution = Counter(self.train_set.targets)
        for label, label_size in distribution.items():
            self.label_distribution[label] = label_size
            self.label_prob[label] = label_size / len(self.train_set)


class FedPACServer(Server):
    def __init__(self, args):
        super().__init__(args)

        self.clf_keys = None
        self.global_protos = {}

    def send_rep_params(self, clients):
        rep_params = [param for name, param in self.model.named_parameters() if name not in self.clf_keys]
        for client in clients:
            client.set_rep_params(rep_params)

    def aggregate_rep(self, clients):
        """
        Aggregate all clients' local representation parameters.
        """
        rep_params = [param for name, param in self.model.named_parameters() if name not in self.clf_keys]
        new_params = torch.zeros_like(parameters_to_vector(rep_params))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            client_rep_params = [param for name, param in client.model.named_parameters() if name not in self.clf_keys]
            client_params = parameters_to_vector(client_rep_params)
            new_params += client_size * client_params

        new_params /= total_size
        new_params.to(self.args["device"])

        vector_to_parameters(new_params, rep_params)

    def aggregate_protos(self, clients):
        """
        Aggregate the global prototypes according to the label distributions of clients.
        """
        aggregate_proto_dict = {}
        label_size_dict = {}

        for client in clients:
            # get this client's label distribution
            label_distribution = client.label_distribution
            local_protos = client.local_protos
            for label in local_protos.keys():
                if label in aggregate_proto_dict:
                    aggregate_proto_dict[label] += local_protos[label] * label_distribution[label]
                    label_size_dict[label] += label_distribution[label]
                else:
                    aggregate_proto_dict[label] = local_protos[label] * label_distribution[label]
                    label_size_dict[label] = label_distribution[label]

        for label, proto in aggregate_proto_dict.items():
            aggregate_proto_dict[label] = proto / label_size_dict[label]

        self.global_protos = aggregate_proto_dict

    def get_head_agg_weight(self, Vars, Hs, client_num):
        """Solve each head's simplex QP; None explicitly retains its local head.

        The distance matrix is a Gram matrix of flattened class statistics.
        Compute in float64 and clip negative round-off eigenvalues, not small
        positive eigenvalues (which carry information in low-variance cases).
        """
        if client_num < 1 or len(Vars) != client_num or len(Hs) != client_num:
            raise ValueError("FedPAC statistics must match a positive client count")
        v = np.asarray(Vars, dtype=np.float64)
        statistics = torch.stack([h.detach().cpu().to(torch.float64) for h in Hs]).numpy()
        if (v.shape != (client_num,) or statistics.ndim != 3 or
                not np.isfinite(v).all() or not np.isfinite(statistics).all()):
            raise ValueError("FedPAC statistics must be finite class-feature arrays")
        # A tiny negative variance can result from cancellation in extraction.
        variance_tolerance = 1e-10 * max(1.0, float(np.abs(v).max()))
        if np.any(v < -variance_tolerance):
            raise ValueError("FedPAC variances must be nonnegative")
        v = np.maximum(v, 0.0)
        avg_weight = []
        self.last_head_solver_stats = []
        for index in range(client_num):
            difference = (statistics[index] - statistics).reshape(client_num, -1)
            p_matrix = np.diag(v) + difference @ difference.T
            p_matrix = (p_matrix + p_matrix.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(p_matrix)
            if eigenvalues[0] < 0:
                p_matrix = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
                p_matrix = (p_matrix + p_matrix.T) / 2
            # Positive scaling does not change the optimum; helps tiny matrices.
            scale = float(np.abs(p_matrix).max())
            if scale > 0:
                p_matrix /= scale
            alpha = None
            diagnostic = dict(index=index, status=None, fallback=False, reason=None)
            try:
                alphav = cvx.Variable(client_num)
                obj = cvx.Minimize(cvx.quad_form(alphav, cvx.psd_wrap(p_matrix)))
                prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
                prob.solve()
                diagnostic["status"] = prob.status
                if prob.status in (cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE):
                    alpha = self.normalize_head_weights(alphav.value, client_num, threshold=1e-3)
                if alpha is None:
                    diagnostic["reason"] = "missing_or_invalid_simplex_solution"
            except (cvx.error.SolverError, cvx.error.DCPError, ArithmeticError) as error:
                diagnostic["reason"] = type(error).__name__ + ": " + str(error)
            diagnostic["fallback"] = alpha is None
            self.last_head_solver_stats.append(diagnostic)
            avg_weight.append(alpha)
        return avg_weight

    @staticmethod
    def normalize_head_weights(values, client_num, threshold=0.0):
        if values is None:
            return None
        weights = np.asarray(values, dtype=np.float64)
        if (weights.shape != (client_num,) or not np.isfinite(weights).all() or
                np.any(weights < -1e-6) or abs(float(weights.sum()) - 1) > 1e-3):
            return None
        weights = np.maximum(weights, 0)
        weights[weights <= threshold] = 0
        total = float(weights.sum())
        if total <= 0:
            return None
        return (weights / total).tolist()

    def aggregate_clf(self, all_clf_params, avg_weight):
        weights = self.normalize_head_weights(avg_weight, len(all_clf_params))
        if weights is None:
            raise ValueError("FedPAC aggregation requires valid simplex weights")
        new_params = torch.zeros_like(all_clf_params[0])
        for i in range(len(all_clf_params)):
            new_params += weights[i] * all_clf_params[i]

        return new_params
