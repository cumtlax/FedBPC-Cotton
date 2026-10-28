import torch

from copy import deepcopy
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from entities.base import Client, Server
from utils.metric import get_accuracy


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, _lambda=0.1, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, _lambda=_lambda, momentum=momentum, weight_decay=weight_decay)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_params):
        group = None
        for group in self.param_groups:
            for p, local_param in zip(group['params'], local_params):
                if p.grad is None:
                    continue

                if group['weight_decay'] != 0:
                    p.grad.data.add_(p.data, alpha=group['weight_decay'])

                grad = p.grad.data + group['_lambda'] * (p.data - local_param.data)

                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(grad)
                    grad = buf

                p.data.add_(grad, alpha=-group['lr'])

        return group['params']


class pFedMeClient(Client):
    def __init__(self, client_id, args, train_set, test_set, global_test_id):
        super().__init__(client_id, args, train_set, test_set, global_test_id)

        self.local_params = deepcopy(list(self.model.parameters()))
        self.personalized_state = None

    def personalized_model(self):
        """Disposable theta for evaluation; model itself remains the upload w."""
        model = deepcopy(self.model)
        if self.personalized_state is not None:
            model.load_state_dict(self.personalized_state)
        return model

    def local_test(self):
        return get_accuracy(self.personalized_model(), self.test_set, self.args["device"])

    def global_test(self, global_tests):
        return get_accuracy(self.personalized_model(), global_tests[self.global_test_id], self.args["device"])

    def train(self):
        """
        Perform local training using Moreau Envelope.
        """
        self.model.train()

        train_loader = DataLoader(self.train_set, batch_size=self.args["batch_size"], shuffle=True)
        optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.args["p_lr"], _lambda=self.args["lambda"])

        for epoch in range(self.args["epochs"]):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=self.args["device"]), labels.to(device=self.args["device"])

                for _ in range(self.args["K"]):
                    # personalized steps
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    personalized_params_bar = optimizer.step(self.local_params)

                # update local weight after finding approximate theta
                for new_param, local_param in zip(personalized_params_bar, self.local_params):
                    local_param.data -= self.args["lr"] * self.args["lambda"] * (local_param.data - new_param.data)

        # Save the last inner theta before restoring the outer w for upload.
        self.personalized_state = {name: value.detach().clone()
                                   for name, value in self.model.state_dict().items()}
        self.set_params(self.local_params)


class pFedMeServer(Server):
    def __init__(self, args):
        super().__init__(args)

    def aggregate_by_params(self, clients):
        prev_params = deepcopy(parameters_to_vector(self.model.parameters()).detach())

        new_params = torch.zeros_like(parameters_to_vector(self.model.parameters()))
        total_size = 0
        for client in clients:
            client_size = self.client_data_size[client.id]
            total_size += client_size
            client_params = parameters_to_vector(client.model.parameters())
            new_params += client_size * client_params

        new_params /= total_size
        new_params.to(self.args["device"])

        personalized_params = (1 - self.args["beta"]) * prev_params + self.args["beta"] * new_params

        vector_to_parameters(personalized_params, self.model.parameters())
