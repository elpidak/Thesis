import os
import numpy as np

import torch
import torch.nn as nn
from vae.encoder_initialization import EncodeInit
from algorithm.ppo import ActorCritic

device = torch.device("cpu")
policies_directory = 'policy/' 
latent_dimension = 95

class Buffer:
    def __init__(self):
        self.rewards = []         
        self.dones = []
        self.observation = []  
        self.actions = []         
        self.log_probs = []     

    def clear(self):
        del self.rewards[:]
        del self.dones[:]
        del self.observation[:]    
        del self.actions[:]        
        del self.log_probs[:]      

class PPOAgent(object):
    def __init__(self, town, action_std_init=0.4):
        
        self.obs_dim = 100
        self.action_dim = 2
        self.clip = 0.2
        self.gamma = 0.99
        self.n_updates_per_iteration = 7
        self.lr = 1e-4
        self.policy_clip = action_std_init
        self.encode = EncodeInit(latent_dimension)
        self.memory = Buffer()
        self.town = town
        self.checkpoint_start = 0
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, self.policy_clip)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr}])
        self.old_policy = ActorCritic(self.obs_dim, self.action_dim, self.policy_clip)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def get_action(self, obs, train):

        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)
            action, logprob = self.old_policy.action_propability(obs.to(device))
        if train:
            self.memory.observation.append(obs.to(device))
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)

        return action.detach().cpu().numpy().flatten()
    
    def set_policy_clip(self, new_policy_clip):
        self.policy_clip = new_policy_clip
        self.policy.set_policy_clip(new_policy_clip)
        self.old_policy.set_policy_clip(new_policy_clip)
 
    def reduction_policy_clip(self, action_std_decay_rate, min_action_std):
        self.policy_clip = self.policy_clip - action_std_decay_rate
        if (self.policy_clip <= min_action_std):
            self.policy_clip = min_action_std
        self.set_policy_clip(self.policy_clip)
        return self.policy_clip


    def learn_policy(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)

        for _ in range(self.n_updates_per_iteration):
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            values = torch.squeeze(values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(values, rewards) - 0.01*dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()
    
    def save_new_checkpoint(self):
        self.checkpoint_start = len(next(os.walk(policies_directory+self.town))[2])
        checkpoint_file = policies_directory+self.town+"/ppo_policy_" + str(self.checkpoint_start)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def load_last_checkpoint(self):
        try :
            self.checkpoint_start = len(next(os.walk(policies_directory+self.town))[2]) - 1
            checkpoint_file = policies_directory+self.town+"/ppo_policy_" + str(self.checkpoint_start)+"_.pth"
            self.old_policy.load_state_dict(torch.load(checkpoint_file,map_location=torch.device('cpu')))
            self.policy.load_state_dict(torch.load(checkpoint_file,map_location=torch.device('cpu')))
        except Exception as e:
            print(e)

    def checkpoint_save(self):
        self.checkpoint_start = len(next(os.walk(policies_directory+self.town))[2])
        if self.checkpoint_start !=0:
            self.checkpoint_start -=1
        checkpoint_file = policies_directory+self.town+"/ppo_policy_" + str(self.checkpoint_start)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)
   
      