import sys
import torch
from vae.encoder import VariationalEncoder

class EncodeInit():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cpu")
        try:
            self.encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.encoder.load()
            self.encoder.eval()
            for params in self.encoder.parameters():
                params.requires_grad = False
        except Exception as e:
            print(e)
            sys.exit()
    
    def encode_observations(self, observation):
        image_observation = torch.tensor(observation[0], dtype=torch.float).to(self.device)
        image_observation = image_observation.unsqueeze(0)
        image_observation = image_observation.permute(0,3,2,1)
        image_observation = self.encoder(image_observation)
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        observation = torch.cat((image_observation.view(-1), navigation_obs), -1)      
        return observation