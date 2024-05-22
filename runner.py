import numpy as np
import ray
import torch

from alg_parameters import *
from mapf_gym import CL_MAPFEnv
from model import Model
from util import set_global_seeds,one_episode_perf,perf_dict_driver,update_perf


class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """initialize model0 and environment"""
        self.ID = env_id
        set_global_seeds(env_id*123)
        self.one_episode_perf = one_episode_perf()
        self.num_iteration=0
        self.im_num_iteration = 0
        self.local_num_agent=EnvParameters.LOCAL_N_AGENTS_LIST[0]

        self.env =CL_MAPFEnv(env_id)
        self.env_im = CL_MAPFEnv(env_id+TrainingParameters.N_ENVS+1,im_flag=True)

        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_model = Model(env_id, self.local_device)
        self.hidden_state = (
            torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
            torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))

        self.env._global_reset(0)
        self.env_im._global_reset(0)
        self.valid_actions, self.obs, self.vector, self.train_valid = self.rl_local_reset_env(True)
        self.done=False
        self.all_obs = [np.zeros((1, self.local_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                             EnvParameters.FOV_SIZE),
                            dtype=np.float32) for _ in range(NetParameters.TIME_DEPT)]
        self.all_obs[-1] = self.obs
        self.im_first_time=True

    def rl_run(self, weights,cl_task_num,switch_task):
        """run multiple steps and collect data for reinforcement learning"""
        with torch.no_grad():
            if switch_task:
                self.local_num_agent= EnvParameters.LOCAL_N_AGENTS_LIST[cl_task_num]
                self.env._global_reset(cl_task_num)
                self.valid_actions, self.obs, self.vector, self.train_valid = self.rl_local_reset_env(True)
                self.done=True
                self.num_iteration = 0
                self.hidden_state = (
                    torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
                    torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))
                self.one_episode_perf = one_episode_perf()
                self.all_obs = [np.zeros((1, self.local_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                                          EnvParameters.FOV_SIZE),
                                         dtype=np.float32) for _ in range(NetParameters.TIME_DEPT)]
                self.all_obs[-1] = self.obs

            mb_obs, mb_vector, mb_rewards, mb_values, mb_done, mb_ps, mb_actions = [], [], [], [], [], [], []
            mb_hidden_state = []
            mb_train_valid = []
            perf_dict = perf_dict_driver()

            self.local_model.set_weights(weights)
            for _ in range(TrainingParameters.N_STEPS):
                cat_all_obs = np.stack(self.all_obs, axis=2)
                mb_obs.append(cat_all_obs)
                mb_vector.append(self.vector)
                mb_train_valid.append(self.train_valid)
                mb_hidden_state.append(
                    [self.hidden_state[0].cpu().detach().numpy(), self.hidden_state[1].cpu().detach().numpy()])
                mb_done.append(self.done)

                actions, ps, values,  self.hidden_state, num_invalid = \
                    self.local_model.step(cat_all_obs, self.vector, self.valid_actions, self.hidden_state,self.local_num_agent)
                self.one_episode_perf['invalid'] += num_invalid

                mb_values.append(values)
                mb_ps.append(ps)
                mb_actions.append(actions)

                rewards, self.valid_actions, self.obs, self.vector, self.train_valid, self.done, \
                    num_on_goals= self.rl_one_step(actions)

                mb_rewards.append(rewards)

                self.all_obs.pop(0)
                self.all_obs.append(self.obs)

                self.one_episode_perf['reward'] += np.sum(rewards)

                if self.done:
                    self.num_iteration += 1
                    first_time = False
                    self.one_episode_perf['diff_collide'] = len(self.env.new_collision_pairs) - self.env.sipp_coll_pair_num
                    self.one_episode_perf["num_collide"]=self.one_episode_perf['num_dynamic_collide']+self.one_episode_perf['num_agent_collide']
                    perf_dict = update_perf(self.one_episode_perf, perf_dict, num_on_goals,self.local_num_agent)
                    self.one_episode_perf = one_episode_perf()
                    if self.num_iteration >= TrainingParameters.ITERATION_LIMIT_LIST[cl_task_num]:
                        self.num_iteration = 0
                        self.env._global_reset(cl_task_num)
                        first_time = True
                    self.valid_actions, self.obs, self.vector, self.train_valid = self.rl_local_reset_env(first_time)
                    self.hidden_state = (
                        torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
                        torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))
                    self.all_obs = [
                        np.zeros((1, self.local_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                                  EnvParameters.FOV_SIZE),
                                 dtype=np.float32) for _ in range(NetParameters.TIME_DEPT)]
                    self.all_obs[-1] = self.obs

            mb_obs = np.concatenate(mb_obs, axis=0)
            mb_vector = np.concatenate(mb_vector, axis=0)
            mb_rewards = np.concatenate(mb_rewards, axis=0)
            mb_values = np.squeeze(np.concatenate(mb_values, axis=0), axis=-1)
            mb_actions = np.asarray(mb_actions, dtype=np.int64)
            mb_ps = np.stack(mb_ps)
            mb_done = np.asarray(mb_done, dtype=np.bool_)
            mb_hidden_state = np.stack(mb_hidden_state)
            mb_train_valid = np.stack(mb_train_valid)
            cat_all_obs = np.stack(self.all_obs, axis=2)
            last_values = np.squeeze(
                self.local_model.value(cat_all_obs, self.vector, self.hidden_state))

            # calculate advantages
            mb_advs = np.zeros_like(mb_rewards)
            last_gaelam= 0
            for t in reversed(range(TrainingParameters.N_STEPS)):
                if t == TrainingParameters.N_STEPS - 1:
                    next_nonterminal = 1.0 - self.done
                    next_values = last_values
                else:
                    next_nonterminal = 1.0 - mb_done[t + 1]
                    next_values = mb_values[t + 1]

                delta = np.subtract(np.add(mb_rewards[t], TrainingParameters.GAMMA * next_nonterminal *
                                               next_values), mb_values[t])

                mb_advs[t] = last_gaelam = np.add(delta,
                                                        TrainingParameters.GAMMA * TrainingParameters.LAM
                                                        * next_nonterminal * last_gaelam)

            mb_returns = np.add(mb_advs, mb_values)

        return mb_obs, mb_vector, mb_returns, mb_values, mb_actions, mb_ps, mb_hidden_state, mb_train_valid, \
            len(perf_dict['num_step']), perf_dict

    def rl_local_reset_env(self,first_time=False):
        self.env._local_reset(self.local_num_agent, first_time)
        valid_actions = []
        obs = np.zeros((1, self.local_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                        EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.local_num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)
        train_valid = np.zeros((self.local_num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

        for i in range(self.local_num_agent):
            valid_action = self.env.list_next_valid_actions(i)
            s = self.env.observe(i)
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]
            valid_actions.append(valid_action)
            train_valid[i, valid_action] = 1
        vector[:, :, 3] = self.env.sipp_coll_pair_num / (self.env.sipp_coll_pair_num + 1)
        return valid_actions, obs, vector, train_valid

    def rl_one_step( self,actions):
        """run one step"""
        train_valid = np.zeros(( self.local_num_agent, EnvParameters.N_ACTIONS), dtype=np.float32)

        obs, vector, rewards, done, next_valid_actions, \
            num_on_goal, num_dynamic_collide, num_agent_collide,success,real_r  \
            = self.env.joint_step(actions)

        self.one_episode_perf['num_dynamic_collide'] += num_dynamic_collide
        self.one_episode_perf['num_agent_collide'] += num_agent_collide
        self.one_episode_perf['real_reward'] += real_r
        if success:
            self.one_episode_perf['team_better'] += 1

        for i in range( self.local_num_agent):
            train_valid[i, next_valid_actions[i]] = 1
        self.one_episode_perf['num_step'] += 1
        return rewards, next_valid_actions, obs, vector, train_valid, done, num_on_goal

    def im_run(self, weights,cl_task_num,switch_task):
        """run multiple steps and collect corresponding data for imitation learning"""
        if switch_task:
            self.local_num_agent = EnvParameters.LOCAL_N_AGENTS_LIST[cl_task_num]
            self.env_im._global_reset(cl_task_num)
            self.im_num_iteration = 0
            self.im_first_time = True

        mb_obs, mb_vector, mb_hidden_state, mb_actions = [], [], [], []
        step = 0
        episode = 0
        self.local_model.set_weights(weights)

        with torch.no_grad():
            while step <= TrainingParameters.N_STEPS:
                if self.im_num_iteration >= TrainingParameters.ITERATION_LIMIT_LIST[cl_task_num]:
                    self.im_num_iteration = 0
                    self.env_im._global_reset(cl_task_num)
                    self.im_first_time = True

                self.env_im._local_reset(self.local_num_agent,self.im_first_time)
                self.im_first_time = False
                self.im_num_iteration +=1
                obs, vector, actions, hidden_state = self.parse_path()
                mb_obs.append(obs)
                mb_vector.append(vector)
                mb_actions.append(actions)
                mb_hidden_state.append(hidden_state)
                step += np.shape(vector)[0]
                episode += 1

            mb_obs = np.concatenate(mb_obs, axis=0)
            mb_vector = np.concatenate(mb_vector, axis=0)
            mb_actions = np.concatenate(mb_actions, axis=0)
            mb_hidden_state = np.concatenate(mb_hidden_state, axis=0)

        return mb_obs, mb_vector, mb_actions, mb_hidden_state, episode, step

    def parse_path(self):
        """take the path generated from M* and create the corresponding inputs and actions"""
        mb_obs, mb_vector, mb_actions, mb_hidden_state = [], [], [], []

        hidden_state = (
            torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device),
            torch.zeros((self.local_num_agent, NetParameters.NET_SIZE)).to(self.local_device))
        all_obs = [np.zeros((1, self.local_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE,
                                  EnvParameters.FOV_SIZE),
                                 dtype=np.float32) for _ in range(NetParameters.TIME_DEPT)]

        obs = np.zeros(
            (1, self.local_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
            dtype=np.float32)
        vector = np.zeros((1, self.local_num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)

        for i in range(self.local_num_agent):
            s = self.env_im.observe(i)
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]
        vector[:, :, 3] = self.env_im.sipp_coll_pair_num / (self.env_im.sipp_coll_pair_num + 1)
        all_obs[-1] = obs
        max_makespan=max(self.env_im.makespan,max([len(self.env_im.sipps_path[i]) for i in range(self.local_num_agent)]))

        for t in range(max_makespan-1):
            cat_all_obs = np.stack(all_obs, axis=2)
            mb_obs.append(cat_all_obs)
            mb_vector.append(vector)
            mb_hidden_state.append([hidden_state[0].cpu().detach().numpy(), hidden_state[1].cpu().detach().numpy()])
            hidden_state = self.local_model.generate_state(cat_all_obs, vector, hidden_state)
            actions = np.zeros(self.local_num_agent)
            for i in range(self.local_num_agent):
                if (t > len(self.env_im.sipps_path[i]) - 2):
                    direction = (0, 0)
                else:
                    pos = self.env_im.sipps_path[i][t]
                    new_pos = self.env_im.sipps_path[i][t + 1]
                    direction = (new_pos[0] - pos[0], new_pos[1] - pos[1])
                actions[i] = self.env_im.world.get_action(direction)
            mb_actions.append(actions)
            obs, vector, _, _, _, _, _, _, _, _= self.env_im.joint_step(actions)
            all_obs.pop(0)
            all_obs.append(obs)

        mb_obs = np.concatenate(mb_obs, axis=0)
        mb_vector = np.concatenate(mb_vector, axis=0)
        mb_actions = np.asarray(mb_actions, dtype=np.int64)
        mb_hidden_state = np.stack(mb_hidden_state)
        return mb_obs, mb_vector, mb_actions, mb_hidden_state

@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)




