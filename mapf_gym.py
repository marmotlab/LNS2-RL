from collections import deque
from numpy import zeros
import numpy as np
from alg_parameters import *
from lns2.build import my_lns2
from mapf_env.build import my_env


class MAPFEnv():
    """map MAPF problems to a standard RL environment"""

    def __init__(self,env_id,adaptive_protect_len):
        """initialization"""
        self.env_id=env_id
        self.adaptive_pair_win = deque(maxlen=adaptive_protect_len)
        self.adaptive_protect_len=adaptive_protect_len

    def joint_step(self, actions):
        """execute joint action and obtain reward"""
        local_done=self.mapf_env.joint_step(actions)
        if local_done:
            return 0, 0, True, 0

        if self.mapf_env.timestep >= self.switch_len:
            self.mapf_env.replan_part1()
            self.lns2_model.add_sipps(self.mapf_env.old_path, self.local_agents, self.mapf_env.agents_poss)
            need_replan=self.mapf_env.replan_part2(self.max_len,self.lns2_model.add_sipps_path,self.lns2_model.add_neighbor.colliding_pairs)
            if need_replan:
                self.lns2_model.replan_sipps(self.mapf_env.local_path, self.mapf_env.replan_ag)
                self.mapf_env.replan_part3(self.lns2_model.replan_sipps_path, self.lns2_model.replan_neighbor.colliding_pairs)
            return 0, 0, True, 0

        obs = zeros((1, self.local_num_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = zeros((1, self.local_num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        self.mapf_env.update_ulti()
        self.mapf_env.predict_next()
        self.mapf_env.observe(actions)
        self.mapf_env.next_valid_actions()
        obs[0, :, :, :]=self.mapf_env.all_obs
        vector[0, :, :] = self.mapf_env.all_vector
        return obs, vector, False, self.mapf_env.valid_actions

    def load_map(self,times,folder_name):
        with open('./'+folder_name+'/eval_map_{}.npy'.format(times), 'rb') as f:
            self.map = np.load(f)
            self.fix_state = np.load(f)
            self.fix_state_dict = np.load(f, allow_pickle=True).item()
            self.start_list = np.load(f)
            self.goal_list = np.load(f)

        self.start_list = list(self.start_list)
        for i in range(len(self.start_list)):
            self.start_list[i] = tuple(self.start_list[i])

        self.goal_list = list(self.goal_list)
        for i in range(len(self.goal_list)):
            self.goal_list[i] = tuple(self.goal_list[i])

    def _global_reset(self,seed,global_num_agent,row,column,switch_factor,max_factor,episode_len):
        """restart a new task"""
        self.lns2_model=my_lns2.MyLns2(seed,self.map,self.start_list,self.goal_list, global_num_agent,row,column)
        self.lns2_model.init_pp()
        self.paths = self.lns2_model.vector_path
        self.global_num_collision = self.lns2_model.num_of_colliding_pairs
        if self.global_num_collision ==0:
            return True
        self.switch_len = switch_factor * max([len(path) for path in self.paths])
        self.max_len = max_factor * max([len(path) for path in self.paths])
        self.mapf_env=my_env.MapfEnv(seed,global_num_agent,EnvParameters.FOV_SIZE, row,column,episode_len,EnvParameters.K_STEPS,
                                     EnvParameters.NUM_TIME_SLICE,EnvParameters.WINDOWS,EnvParameters.UTI_WINDOWS,EnvParameters.DIS_TIME_WEIGHT,
                                     self.paths,self.fix_state,self.fix_state_dict,self.map,self.start_list,self.goal_list)
        self.first_time=True
        self.adaptive_pair_win = deque(maxlen=self.adaptive_protect_len)
        return False

    def _local_reset(self, local_num_agents):
        """restart a new task"""
        update = False
        if not self.first_time:
            if len(self.mapf_env.new_collision_pairs) <= self.lns2_model.old_coll_pair_num:
                new_path= self.mapf_env.local_path
                new_collsion_pair=self.mapf_env.new_collision_pairs
                update=True
            elif self.sipp_coll_pair_num <=self.lns2_model.old_coll_pair_num:
                new_path= self.lns2_model.sipps_path
                new_collsion_pair=self.lns2_model.neighbor.colliding_pairs
                update=True
            if len(self.mapf_env.new_collision_pairs) >= self.sipp_coll_pair_num:
                self.adaptive_pair_win.append(0)
            elif not self.mapf_env.rupt:
                self.adaptive_pair_win.append(1)

        if not update:
            new_path = [[(0, 0)] for _ in range(2)]
            new_collsion_pair = set()
            new_collsion_pair.add((0, 0))

        global_succ=self.lns2_model.select_and_sipps(update,self.first_time,new_path,new_collsion_pair,local_num_agents)
        self.makespan=self.lns2_model.makespan
        self.global_num_collision=self.lns2_model.num_of_colliding_pairs
        self.sipp_coll_pair_num=float(len(self.lns2_model.neighbor.colliding_pairs))
        path_new_agent ={}
        for global_index in self.lns2_model.shuffled_agents:
            path_new_agent[global_index]=self.paths[global_index]
        self.lns2_model.extract_path()
        self.paths = self.lns2_model.vector_path
        if global_succ:
            return True
        if not self.first_time:
            prev_path = {}
            for global_index in self.local_agents:
                prev_path[global_index]=self.paths[global_index]
            prev_agents=self.local_agents
        else:
            prev_agents = [-1]
            prev_path = {-1:[(0,0)]}
            self.first_time = False
        self.local_agents = self.lns2_model.shuffled_agents
        self.local_num_agents=len(self.local_agents)
        self.mapf_env.local_reset(self.makespan,self.lns2_model.old_coll_pair_num,self.sipp_coll_pair_num,self.lns2_model.sipps_path,
                                  path_new_agent,self.paths,prev_path,prev_agents,self.local_agents)

        return False

    def follow_lns(self,local_num_agents,remain_time):
        global_succ= self.lns2_model.rest_lns(local_num_agents,remain_time)
        self.paths=self.lns2_model.vector_path
        if global_succ:
            self.valid_solution()
        return global_succ,self.lns2_model.num_of_colliding_pairs,self.lns2_model.iter_times,self.lns2_model.makespan,self.lns2_model.runtime, self.paths

    def calculate_cost(self):
        sum_of_cost=0
        for i, single_path in enumerate(self.paths):
            clip_index=len(single_path)
            while clip_index>0 and single_path[clip_index-1] == self.goal_list[i]:
                clip_index -= 1
            sum_of_cost += clip_index
        return sum_of_cost

    def validMove(self, poss_t,poss_next):
        if poss_next[0]>= self.map.shape[0] or poss_next[0]<0 or poss_next[1]>=self.map.shape[1] or poss_next[1]<0:
            return False
        if self.map[poss_next]<0:
            return False
        manhattan_distance=abs(poss_t[0]-poss_next[0])+abs(poss_t[1]-poss_next[1])
        return manhattan_distance<2

    def valid_solution(self):
        for i,single_path in enumerate(self.paths):
            if len(single_path)==0:
                print('No solution for agent ',i)
            elif single_path[0]!=self.start_list[i]:
                print('The path of agent ',i,' starts from location ',single_path[0],
                      ', which is different from its start location ',self.start_list[i])
            elif single_path[-1]!=self.goal_list[i]:
                print('The path of agent ',i,' ends at location ',single_path[-1],
                      ', which is different from its goal location ',self.goal_list[i])
            for t in range(len(single_path)-1):
                if not self.validMove(single_path[t],single_path[t+1]):
                    print('The path of agent ',i,' at time ',t,' is not valid')

            for j, other_path in enumerate(self.paths):
                if (i>= j or len(other_path)==0):
                    continue
                if len(single_path)<=len(other_path):
                    shorter_path=single_path
                    longer_path=other_path
                    shorter_a=i
                    longer_a=j
                else:
                    shorter_path = other_path
                    longer_path = single_path
                    shorter_a=j
                    longer_a=i
                for t in range(len(shorter_path)):
                    if shorter_path[t]==longer_path[t]:
                        print('Find a vertex conflict between agents ',shorter_a,' and ',longer_a,
                              ' at location ',shorter_path[t],' at timestep ',t)
                    elif t>0 and shorter_path[t]==longer_path[t-1] and shorter_path[t-1]==longer_path[t]:
                        print('Find an edge conflict between agents ', shorter_a, ' and ', longer_a,
                              ' at edge (',shorter_path[t-1],',',shorter_path[t],') at timestep ', t)
                for t in range(len(shorter_path),len(longer_path)):
                    if longer_path[t]==shorter_path[-1] and len(longer_path)>1 and len(shorter_path)>1:
                        print('Find a target conflict between agents ', shorter_a, ' and ', longer_a,
                              ' at location ', longer_path[t], ' at timestep ', t)
