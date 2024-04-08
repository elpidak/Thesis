import os
import torch
import torch.nn as nn
import torch.optim 
import numpy as np
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.observation_dimension = obs_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.convolution_var = torch.full((self.action_dim,), action_std_init)
        self.convolution_matrix = torch.diag(self.convolution_var).unsqueeze(dim=0)
        self.actor = nn.Sequential(nn.Linear(self.observation_dimension, 500),nn.Tanh(),nn.Linear(500, 300),nn.Tanh(),nn.Linear(300, 100), nn.Tanh(),nn.Linear(100, self.action_dim),nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(self.observation_dimension, 500),nn.Tanh(),nn.Linear(500, 300),nn.Tanh(),nn.Linear(300, 100),nn.Tanh(),nn.Linear(100, 1))

    def set_policy_clip(self, new_action_std):
        self.convolution_var = torch.full((self.action_dim,), new_action_std)

    def evaluate(self, observation, action):
        mean = self.actor(observation)
        covolution_var = self.convolution_var.expand_as(mean)
        covolution_matrix = torch.diag_embed(covolution_var)
        dist = MultivariateNormal(mean, covolution_matrix)
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(observation)
        return logprobs, values, dist_entropy
    
    def action_propability(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float)
        mean = self.actor(observation)
        dist = MultivariateNormal(mean, self.convolution_matrix)
        action = dist.sample()
        propability = dist.log_prob(action)
        return action.detach(), propability.detach()
    