import random

import numpy as np
import torch
import wandb
from alg_parameters import *


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True

def write_to_wandb_im(step, mb_loss=None):
    """record performance using wandb"""
    mb_loss = np.nanmean(mb_loss, axis=0)
    wandb.log({'Loss/Im_loss':mb_loss[0]}, step=step)
    wandb.log({'Grad/Im_grad_norm':mb_loss[1]}, step=step)


def write_to_wandb(step, performance_dict=None, mb_loss=None):
    """record performance using wandb"""
    loss_vals = np.nanmean(mb_loss, axis=0)
    wandb.log({'Perf/Reward': performance_dict['reward']}, step=step)
    wandb.log({'Perf/Valid_rate': performance_dict['invalid']}, step=step)
    wandb.log({'Perf/Episode_length': performance_dict['num_step']}, step=step)
    wandb.log({'Perf/Final_goals': performance_dict['final_goals']}, step=step)
    wandb.log({'Perf/Num_dynamic_collide': performance_dict['num_dynamic_collide']},
              step=step)
    wandb.log({'Perf/Num_agent_collide': performance_dict['num_agent_collide']},
              step=step)
    wandb.log({'Perf/Diff_collide': performance_dict['diff_collide']},
              step=step)
    wandb.log({'Perf/Team_better': performance_dict["team_better"]},
              step=step)
    wandb.log({'Perf/Real_reward': performance_dict["real_reward"]},
              step=step)
    wandb.log({'Perf/Num_collide': performance_dict['num_collide']},
              step=step)

    for (val, name) in zip(loss_vals, RecordingParameters.LOSS_NAME):
        if name == 'grad_norm':
            wandb.log({'Grad/' + name: val}, step=step)
        else:
            wandb.log({'Loss/' + name: val}, step=step)


def perf_dict_driver():
    performance_dict = {'num_step': [], 'reward': [], 'invalid': [],
                        'num_dynamic_collide': [], "num_agent_collide": [],"final_goals":[],"diff_collide":[],
                        "real_reward":[],"team_better":[],"num_collide":[]}

    return performance_dict

def update_perf(one_episode_perf, performance_dict, num_on_goals,num_agent):
    """record batch performance"""
    performance_dict['num_step'].append(one_episode_perf['num_step'])
    performance_dict['reward'].append(one_episode_perf['reward'])
    performance_dict['invalid'].append((one_episode_perf['num_step'] * num_agent -
                                        one_episode_perf['invalid']) / (
                                               one_episode_perf['num_step'] * num_agent))
    performance_dict['num_dynamic_collide'].append(one_episode_perf['num_dynamic_collide'])
    performance_dict['num_agent_collide'].append(one_episode_perf['num_agent_collide'])
    performance_dict['final_goals'].append(num_on_goals)
    performance_dict['diff_collide'].append(one_episode_perf['diff_collide'])
    performance_dict['real_reward'].append(one_episode_perf['real_reward'])
    performance_dict['team_better'].append(one_episode_perf['team_better'])
    performance_dict['num_collide'].append(one_episode_perf['num_collide'])
    return performance_dict

def one_episode_perf():
    one_episode_perf = {'num_step': 0, 'reward': 0, 'invalid': 0,
                        'num_dynamic_collide': 0, "num_agent_collide": 0,"final_goals":0,"diff_collide":0,
                        "real_reward":0,"team_better":0,"num_collide":0}
    return one_episode_perf
