#############
# NEW VERSION#
#############
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 name, chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)  # output a scalar

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.devide = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        v = self.v(state_value)
        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

## The Gaussian
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action, n_actions, name, chkpt_dir='tmp/sac'):
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        # networks
        self.fc1 = nn.Linaer(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mean = nn.Linear(self.fc2_dims, self.n_actions)
        self.log_std = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,state):
        temp = F.relu(self.fc1(state))
        temp = F.relu(self.fc2(temp))
        mean = self.mean(temp)
        log_std = self.log_std(temp)
        log_std = T.clamp(log_std, -20, 2) ## The clamp

        return mean, log_std

    def sample_normal(self,state):
        # may need to send to device not sure
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean,std)
        xt = normal.rsample()
        yt = T.tanh(xt)
        log_probs = normal.log_prob(xt) - T.log(1-yt.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        mean = T.tanh(mean)

        return yt, log_probs, mean

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# Double Q
class CriticNetwork(nn.Module):
    def __init__(self,lr,input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Q1 Network
        self.fc1_q1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2_q1 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims,1)  # output scalar

        # Q2 Network
        self.fc1_q2 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2_q2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q2 = nn.Linear(self.fc2_dims, 1)

        self.optimizaer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
       state_action = T.cat([state,action])

       state_action_value = T.relu(self.fc1_q1(state_action))
       state_action_value = T.relu(self.fc2_q1(state_action_value))
       q1 = self.q1(state_action_value)

       state_action_value = T.relu(self.fc1_q2(state_action))
       state_action_value = T.relu(self.fc2_q2(state_action_value))
       q2 =self.q2(state_action_value)

       return q1, q2





