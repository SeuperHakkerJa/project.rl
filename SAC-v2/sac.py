import torch as T
import torch.nn.functional as F
from buffer import ReplayMemory
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, env_id, gamma=0.99,
                 n_actions=2,max_size=1000000,layer1_size=256,layer2_size=256, batch_size=100, reward_scale=1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayMemory()
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,layer2_size,max_action=env.action_space.high
                                  , n_actions=n_actions,name=env_id+'_actor',)
        self.critic_1 = CriticNetwork(beta,input_dims,layer1_size,layer2_size,
                                      n_actions,env_id+'_critic_1')
        self.critic_target = CriticNetwork(beta,input_dims,layer1_size, layer2_size
                                           ,n_actions,env_id+'_critic_target')
        self.update_network_parameters(tau=1)
        self.scale = reward_scale



    def choose_action(self,observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor

