import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from vae.encoder_initialization import EncodeInit
from algorithm.agent import PPOAgent
from carla_env.client import ClientConnection
from carla_env.environment import CarlaEnvironment


def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--town', type=str, default="Town02")
    parser.add_argument('--load-checkpoint', type=bool, default=False)
    arguments = parser.parse_args()
    return arguments


def start_process():

    args = parse_arguments()
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    action_std_init = 0.2
    
    if train == True:
        writer = SummaryWriter(f"runs/Training/Clip_{action_std_init}/{town}")
    else:
        writer = SummaryWriter(f"runs/Testing/Clip_{action_std_init}/{town}")
    writer.add_text("hyperparameters","|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    latent_dimension = 95
    total_timesteps = 2e6
    episode_length = 7500
    action_std_decay_rate = 0.05
    min_action_std = 0.05   
    action_std_decay_freq = 5e5
    timestep = 0
    episode = 0
    episode_steps = 0
    deviation_from_center = 0
    distance_covered = 0
    ep_fuel_consumption_average=0
    cumulative_score = 0
    episodic_length = list()
    scores = list()


    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has start successfully.")
    except:
        logging.error("Connection is lost.")
        ConnectionRefusedError
    if train:
        env = CarlaEnvironment(client, world,town)
    else:
        env = CarlaEnvironment(client, world,town, checkpoint_frequency=None)
    encode = EncodeInit(latent_dimension)

    try:
        time.sleep(0.5)
        
        if checkpoint_load:
            chkt_file_nums = len(next(os.walk(f'checkpoints/{town}'))[2]) - 1
            chkpt_file = f'checkpoints/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']
                action_std_init = data['action_std_init']
                print(f"Episode: {episode}")
                print(f"Timestep: {timestep}")
                print(f"Cumulative score: {cumulative_score}")
                print(f"Action Std Int: {action_std_init}")
            agent = PPOAgent(town, action_std_init)
            agent.load_last_checkpoint()
        else:
            if train == False:
                agent = PPOAgent(town, action_std_init)
                agent.load_last_checkpoint()
                for params in agent.old_policy.actor.parameters():
                    params.requires_grad = False
            else:
                agent = PPOAgent(town, action_std_init)
        if train:
            try:
  
                while timestep < total_timesteps:
                    observation = env.reset()
                    observation = encode.encode_observations(observation)
                    current_ep_reward = 0
                    t1 = datetime.now()
                    for t in range(episode_length):

                        action = agent.get_action(observation, train=True)
                        episode_steps +=episode_steps
                        observation, reward, done, info = env.step(action)
                        if observation is None:
                            break
                        observation = encode.encode_observations(observation) 
                        agent.memory.rewards.append(reward)
                        agent.memory.dones.append(done)
                        timestep +=1
                        current_ep_reward += reward

                        if timestep % action_std_decay_freq == 0:
                            action_std_init =  agent.reduction_policy_clip(action_std_decay_rate, min_action_std)

                        if timestep == total_timesteps -1:
                            agent.checkpoint_save()

                        if done:
                            episode += 1
                            t2 = datetime.now()
                            t3 = t2-t1
                            episodic_length.append(abs(t3.total_seconds()))
                            break
                    
                    deviation_from_center += info[1]
                    distance_covered += info[0]
                    ep_fuel_consumption_average += info[2]
                    
                    scores.append(current_ep_reward)
                 
                    if checkpoint_load:
                        cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                    else:
                        cumulative_score = np.mean(scores)

                    print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                    if episode % 10 == 0: 
                        agent.learn_policy()
                        agent.checkpoint_save()
                        chkt_file_nums = len(next(os.walk(f'checkpoints/{town}'))[2])
                        if chkt_file_nums != 0:
                            chkt_file_nums -=1
                        chkpt_file = f'checkpoints/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                        data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                        with open(chkpt_file, 'wb') as handle:
                            pickle.dump(data_obj, handle)
                        
                    
                    if episode % 5 == 0:
                        writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                        writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                        writer.add_scalar("Fuel Consumption/episode",ep_fuel_consumption_average/5, episode)
                        writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                        writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5]), episode)
                        writer.add_scalar("Average Reward/(t)", np.mean(scores[-5]), timestep)
                        writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                        writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                        writer.add_scalar("Average Deviation from Center/episode", deviation_from_center/5, episode)
                        writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center/5, timestep)
                        writer.add_scalar("Average Distance Covered (m)/episode", distance_covered/5, episode)
                        writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered/5, timestep)

                        episodic_length = list()
                        deviation_from_center = 0
                        distance_covered = 0
                        ep_fuel_consumption_average = 0

                    if episode % 100 == 0:
                        agent.save_new_checkpoint()
                        chkt_file_nums = len(next(os.walk(f'checkpoints/{town}'))[2])
                        chkpt_file = f'checkpoints/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                        data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                        with open(chkpt_file, 'wb') as handle:
                            pickle.dump(data_obj, handle)
            except Exception as e:
                print(f"The exception is {e}")
                            
            print("Terminating the run.")
            sys.exit()
        else:
            while timestep < 5e4:
                observation = env.reset()
                observation = encode.encode_observations(observation)

                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(episode_length):
                    action = agent.get_action(observation, train=False)
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    observation = encode.encode_observations(observation)
                    
                    timestep +=1
                    current_ep_reward += reward
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info[1]
                distance_covered += info[0]
                ep_fuel_consumption_average = info[2]  
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score)) 
                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Fuel Consumption/episode",ep_fuel_consumption_average, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run here.")
            sys.exit()
    except Exception as e :
        print (f"The exception here is :{e}")
    finally:
        sys.exit()


if __name__ == "__main__":
    try:        
        start_process()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
