import torch
import numpy as np
from sac import SAC
from replay_memory import ReplayMemory


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NiaveHRL():

    def __init__(self, state_space, action_space, H, args):

        # SAC needs state_dim and the whole action space and the args
        # nHRL needs H
        self.state_dim = state_space.shape[0]
        self.action_space = action_space
        self.H = H

        # Low Level Agents yield primitive actions
        self.low_level_agent = SAC(self.state_dim, action_space, args)
        self.low_level_buffer = ReplayMemory(args.replay_size, args.seed)

        # High Level Agent yields a state
        self.high_level_agent = SAC(self.state_dim, state_space, args)
        self.high_level_buffer = ReplayMemory(args.replay_size, args.seed)

        # Other Params
        self.rewards = 0
        self.timesteps = 0
        self.goals =

    def run_low_level(self):


