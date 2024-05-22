import torch
from torch import no_grad
import wandb
from model import Model
from alg_parameters import *
from util import set_global_seeds,global_list,global_dict,global_std
import numpy as np
from numpy import zeros,stack
from mapf_gym import MAPFEnv
from time import time
from collections import deque
import ray

ray.init()

USE_WANDB=False
NUM_TIMES=100
GLOBAL_N_AGENT=60
ROW=10
COLUMN=10
OBSTACLE_PROB=0.175
LOCAL_N_AGENTS=8
GLOBAL_TIME_LIMIT=100
SIPPS_AG=8
PAIR_WIN_THRESHOLD=0.3
PAIR_WIN_PROB_LEN=20
NUM_thraed=10
SWICTH_FACTOR=1.2
MAX_FACTOR=2.2
SEED=1234
EPISODE_LEN = 356
FOLDER_NAME='maps_60_10_10_0.175'
set_global_seeds(SEED)

all_config={'NUM_TIMES':NUM_TIMES, "GLOBAL_N_AGENT":GLOBAL_N_AGENT,"ROW":ROW,"COLUMN":COLUMN,
            "OBSTACLE_PROB":OBSTACLE_PROB,"LOCAL_N_AGENTS": LOCAL_N_AGENTS,"GLOBAL_TIME_LIMIT":GLOBAL_TIME_LIMIT,
            "SIPPS_AG":SIPPS_AG,"PAIR_WIN_THRESHOLD":PAIR_WIN_THRESHOLD,
            "PAIR_WIN_PROB_LEN":PAIR_WIN_PROB_LEN,"SWICTH_FACTOR":SWICTH_FACTOR,"MAX_FACTOR":MAX_FACTOR}


@ray.remote(num_cpus=1, num_gpus=1/NUM_thraed)
class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """initialize model0 and environment"""
        set_global_seeds(SEED)
        self.env_id=env_id
        self.local_device = torch.device('cuda')
        self.env = MAPFEnv(env_id, PAIR_WIN_PROB_LEN)

        self.local_model = Model(env_id, self.local_device)
        model_path = './final'
        path_checkpoint = model_path + "/net_checkpoint.pkl"
        self.local_model.network.load_state_dict(torch.load(path_checkpoint, map_location=self.local_device)['model'])

    def local_reset_env(self):
        global_done=self.env._local_reset(LOCAL_N_AGENTS)
        if global_done:
            return True,0,0,0, global_done
        self.num_local_agents=self.env.local_num_agents
        obs = zeros((1,  self.num_local_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                        EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = zeros((1,  self.num_local_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        self.env.mapf_env.next_valid_actions()
        actions=[0 for _ in range(self.num_local_agents)]
        self.env.mapf_env.observe(actions)
        obs[0, :, :, :]=self.env.mapf_env.all_obs
        vector[0, :, :] = self.env.mapf_env.all_vector
        return False, self.env.mapf_env.valid_actions, obs, vector,global_done

    def run(self,task_num):
        with no_grad():
            rl_record = global_list()
            final_record = global_list()
            iter_num=0
            self.env.load_map(task_num,FOLDER_NAME)
            swicth = False
            start_time = time()
            global_done=self.env._global_reset(SEED,GLOBAL_N_AGENT,ROW,COLUMN,SWICTH_FACTOR,MAX_FACTOR,EPISODE_LEN)
            if global_done:
                final_record['run_time'] =time() - start_time
                self.env.valid_solution()
                final_record['success']+=1
                final_record['makespan']=max([len(path) for path in self.env.paths])
                sum_of_cost=self.env.calculate_cost()
                final_record['sum_of_cost']=sum_of_cost
                return rl_record,final_record,self.env_id,task_num
            while (time()-start_time)<GLOBAL_TIME_LIMIT:
                done, valid_actions, obs, vector, global_done = self.local_reset_env()
                if global_done:
                    rl_record['success']+=1
                    break
                if len(self.env.adaptive_pair_win)>=PAIR_WIN_PROB_LEN and np.mean(self.env.adaptive_pair_win)<PAIR_WIN_THRESHOLD:
                    rl_record['run_time'] = time() - start_time
                    swicth=True
                    rl_record['global_coll'] = self.env.global_num_collision
                    rl_record['makespan'] = self.env.makespan
                    final_record['success'],final_record['global_coll'],final_record['iter'],\
                        final_record['makespan'],final_record['run_time'],_=self.env.follow_lns(SIPPS_AG, GLOBAL_TIME_LIMIT - (time() - start_time))
                    final_record['iter']+=iter_num
                    final_record['run_time']+=rl_record['run_time']
                    break
                all_obs = deque([zeros((1,  self.num_local_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                              EnvParameters.FOV_SIZE),
                             dtype=np.float32) for _ in range(NetParameters.TIME_DEPT)])
                all_obs[-1] = obs
                hidden_state = (torch.zeros(( self.num_local_agents, NetParameters.NET_SIZE)).to(self.local_device),
                                torch.zeros(( self.num_local_agents, NetParameters.NET_SIZE)).to(self.local_device))

                while not done:
                    cat_all_obs = stack(all_obs, axis=2)
                    actions, hidden_state = self.local_model.eval(cat_all_obs, vector, valid_actions, hidden_state, self.num_local_agents)
                    obs, vector,  done, valid_actions \
                        = self.env.joint_step(actions)
                    all_obs.pop()
                    all_obs.append(obs)
                iter_num += 1

            rl_record['iter'] = iter_num
            if not swicth:
                rl_record['run_time'] = time() - start_time
                rl_record['global_coll'] = self.env.global_num_collision
                rl_record['makespan'] = self.env.makespan
                final_record=rl_record
            if final_record['success'] or rl_record['success']:
                sum_of_cost=self.env.calculate_cost()
                final_record['sum_of_cost']=sum_of_cost
                if not swicth:
                    rl_record['sum_of_cost']=sum_of_cost
            return rl_record,final_record,self.env_id,task_num


if __name__ == "__main__":
    # recording
    if USE_WANDB:
        wandb_id = wandb.util.generate_id()
        wandb.init(project='MAPF',
                   name='evaluation',
                   entity='your_name',
                   notes='none',
                   config=all_config,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    try:
        print(all_config)
        rl_record=global_dict()
        final_record = global_dict()
        envs = [Runner.remote(i) for i in range(NUM_thraed)]
        job_list=[]
        for i, env in enumerate(envs):
            job_list.append(env.run.remote(i+1))
        j=0
        task_id=NUM_thraed
        while j < NUM_TIMES:
            done_id, job_list = ray.wait(job_list)
            tasks_results = ray.get(done_id)
            for task_num, results in enumerate(tasks_results):
                rl_perf, final_perf,env_id, self_task = results
                print(j)
                print("RL")
                print(rl_perf)
                print("LNS2")
                print(final_perf)
                print('----------------------------------------------------------------------------------------------------------\n')
                j += 1
                task_id += 1
                if task_id <= NUM_TIMES:
                    job_list.append(envs[env_id].run.remote(task_id))

                for key in rl_record.keys():
                    if key == "makespan" or key == 'sum_of_cost':
                        if rl_perf['success']:
                            rl_record[key].append(rl_perf[key])
                    else:
                        rl_record[key].append(rl_perf[key])
                for key in final_record.keys():
                    if key == "makespan" or key == 'sum_of_cost':
                        if final_perf['success']:
                            final_record[key].append(final_perf[key])
                    else:
                        final_record[key].append(final_perf[key])
    finally:
        for e in envs:
            ray.kill(e)
        ray.shutdown()
        rl_std=global_std()
        final_std=global_std()
        for key in rl_std.keys():
            rl_std[key]= round(np.std(rl_record[key]),3)
        for key in final_std.keys():
            final_std[key]= round(np.std(final_record[key]),3)

        for key in rl_record.keys():
            rl_record[key] = round(np.nanmean(rl_record[key]), 3)
        for key in final_record.keys():
            final_record[key] = round(np.nanmean(final_record[key]), 3)

        print('RL')
        print(rl_record)
        print(rl_std)
        print('--------------------------------------------------------------------------------------------\n')
        print('LNS2')
        print(final_record)
        print(final_std)
        if USE_WANDB:
            wandb.finish()
