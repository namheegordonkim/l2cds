from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch.distributions import Normal

from utils.utils import device


class ProbabilisticNet(nn.Module):

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    def get_log_prob(self, state, action):
        """
        Get the probability of committing the specified action in state, given current weights of the network
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = get_normal_dist(mean, std)
        log_prob = normal.log_prob(action)
        # log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6)
        return log_prob.sum(1, keepdim=True)

    def sample(self, inputs):
        mean, log_std = self.forward(inputs)
        std = log_std.exp()
        normal = get_normal_dist(mean, std)
        action = normal.sample()
        log_prob = torch.sum(normal.log_prob(action), dim=1)
        return action, log_prob, mean, log_std


class ActorCriticNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(ActorCriticNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.v_fcs = nn.ModuleList()
        p_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        v_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        self.p_fcs.append(p_fc)
        self.v_fcs.append(v_fc)
        for i in range(len(self.hidden_layer) - 1):
            p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i + 1])
            v_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i + 1])
            self.p_fcs.append(p_fc)
            self.v_fcs.append(v_fc)
        self.mu = nn.Linear(self.hidden_layer[-1], num_outputs)
        self.log_std = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)
        self.v = nn.Linear(self.hidden_layer[-1], 1)
        self.noise = 0

        self.mu.weight.data.fill_(0.0)
        self.mu.bias.data.fill_(0.0)
        # self.train()

    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        for i in range(len(self.hidden_layer) - 1):
            x = F.relu(self.p_fcs[i + 1](x))
        mu = torch.tanh(self.mu(x))
        log_std = Variable(self.noise * torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)

        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer) - 1):
            x = F.relu(self.v_fcs[i + 1](x))
        v = self.v(x)
        # print(mu)
        return mu, log_std, v

    def set_noise(self, noise):
        self.noise = noise


def get_normal_dist(mean, std):
    normal = Normal(mean, std)
    normal.loc = normal.loc.to(device)
    normal.scale = normal.scale.to(device)
    return normal


class ActorNet(ProbabilisticNet):
    def __init__(self, num_inputs, num_outputs, hidden_layer=[64, 64]):
        super(ActorNet, self).__init__()
        self.num_outputs = num_outputs
        self.hidden_layer = hidden_layer
        self.p_fcs = nn.ModuleList()
        self.log_stds = nn.ModuleList()
        p_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        log_std = nn.Linear(num_inputs, self.hidden_layer[0])
        self.p_fcs.append(p_fc)
        self.log_stds.append(log_std)
        for i in range(len(self.hidden_layer) - 1):
            p_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i + 1])
            log_std = nn.Linear(self.hidden_layer[i], self.hidden_layer[i + 1])
            self.p_fcs.append(p_fc)
            self.log_stds.append(log_std)
        self.mu = nn.Linear(self.hidden_layer[-1], num_outputs)
        self.log_std = nn.Parameter(torch.zeros(num_outputs), requires_grad=True)
        self.noise = -2.0
        self.noises = torch.Tensor(num_outputs)
        self.log_std_linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, inputs):
        # actor
        x = F.relu(self.p_fcs[0](inputs))
        log_std = F.relu(self.log_stds[0](inputs))
        for i in range(len(self.hidden_layer) - 1):
            x = F.relu(self.p_fcs[i + 1](x))
            log_std = F.relu(self.log_stds[i + 1](log_std))
        mu = torch.tanh(self.mu(x))
        log_std = Variable(-2.0 * torch.ones(self.num_outputs)).unsqueeze(0).expand_as(mu)
        # log_std = torch.tanh((self.log_std_linear(inputs)))
        # log_std = torch.clamp(log_std, min=-2, max=2)
        return mu, log_std

    def set_noise(self, noise):
        self.noise = noise


class ValueNet(nn.Module):
    def __init__(self, num_inputs, hidden_layer=[64, 64]):
        super(ValueNet, self).__init__()
        self.hidden_layer = hidden_layer
        self.v_fcs = nn.ModuleList()
        v_fc = nn.Linear(num_inputs, self.hidden_layer[0])
        self.v_fcs.append(v_fc)
        for i in range(len(self.hidden_layer) - 1):
            v_fc = nn.Linear(self.hidden_layer[i], self.hidden_layer[i + 1])
            self.v_fcs.append(v_fc)
        self.v = nn.Linear(self.hidden_layer[-1], 1)

    def forward(self, inputs):
        # critic
        x = F.relu(self.v_fcs[0](inputs))
        for i in range(len(self.hidden_layer) - 1):
            x = F.relu(self.v_fcs[i + 1](x))
        v = self.v(x)
        # print(mu)
        return v


class TwoHandedNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64]):
        super(TwoHandedNet, self).__init__()
        self.hidden_layer = hidden_dims
        self.output_dim = output_dim
        self.q_fcs1 = nn.ModuleList()
        self.q_fcs2 = nn.ModuleList()
        q_fc1 = nn.Linear(input_dim, self.hidden_layer[0])
        q_fc2 = nn.Linear(input_dim, self.hidden_layer[0])
        self.q_fcs1.append(q_fc1)
        self.q_fcs2.append(q_fc2)
        for i in range(len(self.hidden_layer) - 1):
            q_fc1 = nn.Linear(self.hidden_layer[i], self.hidden_layer[i + 1])
            q_fc2 = nn.Linear(self.hidden_layer[i], self.hidden_layer[i + 1])
            self.q_fcs1.append(q_fc1)
            self.q_fcs2.append(q_fc2)
        self.q_1 = nn.Linear(self.hidden_layer[-1], 1)
        self.q_2 = nn.Linear(self.hidden_layer[-1], 1)

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], 1)
        q1 = F.relu(self.q_fcs1[0](inputs))
        q2 = F.relu(self.q_fcs2[0](inputs))
        for i in range(len(self.hidden_layer) - 1):
            q1 = F.relu(self.q_fcs1[i + 1](q1))
            q2 = F.relu(self.q_fcs2[i + 1](q2))
        q1 = (self.q_1(q1))
        q2 = (self.q_2(q2))
        return q1, q2


class NNet(ProbabilisticNet):
    """
    Generic neural network model, mainly used for projection of state spaces onto
    the canonical state space.
    """

    def __init__(self, input_dim, output_dim, final_layer_activation, hidden_dims=[64, 64]):
        super(NNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        # self.log_std = nn.Parameter(-1.0 * torch.ones(self.output_dim).reshape(1, -1))
        self.log_std = -2 * torch.ones(self.output_dim).reshape(1, -1).to(device)

        # self.batch_norm = nn.BatchNorm1d(input_dim)
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.fcs = create_fcs(dims)
        self.final_layer_activation = final_layer_activation

    def forward(self, x):
        # x = self.batch_norm(x)
        x = forward_fcs(x, self.fcs)
        x = self.fcs[-1](x)
        x = self.final_layer_activation.forward(x)
        log_std = self.log_std.expand_as(x).to(device)
        return x, log_std


class IdentityNet(ProbabilisticNet):

    def forward(self, x):
        return x, None


class DummyNet(ProbabilisticNet):

    def forward(self, x):
        return x


class FakeNet(ProbabilisticNet):
    """
    Generic neural network model, mainly used for projection of state spaces onto
    the canonical state space.
    """

    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, x):
        n = x.shape[0]
        x = torch.stack([self.output for _ in range(n)]).to(device)
        return x, None


class TorchInputMaker:
    """
    Allows extraneous inputs into black boxes
    """

    def __init__(self, x):
        self.x = x

    def make_input(self, x):
        return torch.cat([x, self.x])


def create_fcs(dims):
    fcs = nn.ModuleList()
    for i in range(len(dims) - 1):
        fc = nn.Linear(dims[i], dims[i + 1])
        fcs.append(fc)
    return fcs


def forward_fcs(x, fcs):
    for i in range(len(fcs) - 1):
        x = fcs[i](x)
        x = F.leaky_relu(x)
    return x


class ScalerWrapper(nn.Module):

    def __init__(self, scaler: StandardScaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, x):
        x = x.cpu().detach().numpy()
        return torch.as_tensor(self.scaler.transform(x)).float().to(device)

    def reverse(self, x):
        x = x.cpu().detach().numpy()
        return torch.as_tensor(self.scaler.inverse_transform(x)).float().to(device)


class DoubleNNet(nn.Module):

    def __init__(self, left_model: NNet, right_model: NNet):
        super().__init__()
        self.left_model = left_model
        self.right_model = right_model

    def forward(self, left_input, right_input):
        left_output, _ = self.left_model.forward(left_input)
        right_output, _ = self.right_model.forward(right_input)
        return left_output, right_output
