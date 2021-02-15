import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class ActorCriticNetwork(nn.Module):
    def __init__(self, lr, input_dim, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims,n_actions) # policy
        self.v = nn.Linear(fc2_dims,1) # value func
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # device
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)




    def foward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.pi(v)

        return (pi,v)


class Agenet():
    def __init__(self, lr, input_dim, fc1_dims, fc2_dims, n_actions, gamma = 0.99):
        self.gamma = gamma
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr

        self.actor_critic = ActorCriticNetwork(lr,input_dim,n_actions,fc1_dims, fc2_dims)
        