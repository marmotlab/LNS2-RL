import copy
import random
import sys

import gym
import numpy as np
from lns2.build import my_lns2
from world_property import State
from alg_parameters import *
from dynamic_state import DyState
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}


class CL_MAPFEnv(gym.Env):
    """map MAPF problems to a standard RL environment"""

    def __init__(self,env_id,global_num_agents_range=EnvParameters.GLOBAL_N_AGENT_LIST, fov_size=EnvParameters.FOV_SIZE, size=EnvParameters.WORLD_SIZE_LIST,
                 prob=EnvParameters.OBSTACLE_PROB_LIST,im_flag=False):
        """initialization"""
        self.global_num_agents_range = global_num_agents_range
        self.fov_size =fov_size
        self.SIZE = size  # size of a side of the square grid
        self.PROB = prob  # obstacle density
        self.env_id=env_id
        self.im_flag=im_flag

    def global_set_world(self,cl_num_task):
        """randomly generate a new task"""

        def get_connected_region(world0, regions_dict, x0, y0):
            # ensure at the beginning of an episode, all agents and their goal at the same connected region
            sys.setrecursionlimit(1000000)
            if (x0, y0) in regions_dict:  # have done
                return regions_dict[(x0, y0)]
            visited = set()
            sx, sy = world0.shape[0], world0.shape[1]
            work_list = [(x0, y0)]
            while len(work_list) > 0:
                (i, j) = work_list.pop()
                if i < 0 or i >= sx or j < 0 or j >= sy:
                    continue
                if world0[i, j] == -1:
                    continue  # crashes
                if world0[i, j] > 0:
                    regions_dict[(i, j)] = visited
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                work_list.append((i + 1, j))
                work_list.append((i, j + 1))
                work_list.append((i - 1, j))
                work_list.append((i, j - 1))
            regions_dict[(x0, y0)] = visited
            return visited

        prob = random.choice(self.PROB[cl_num_task])
        task_set=random.choice(range(len(self.SIZE)))
        self.size = self.SIZE[task_set]
        self.episode_len = EnvParameters.EPISODE_LEN[task_set]
        self.global_num_agent=int(round(random.choice(self.global_num_agents_range[cl_num_task])*self.size*self.size))
        self.map = -(np.random.rand(int(self.size), int(self.size)) < prob).astype(int)  # -1 obstacle,0 nothing
        self.fix_state=copy.copy(self.map)
        self.fix_state_dict = {}
        for i in range(int(self.size)):
            for j in range(int(self.size)):
                self.fix_state_dict[i,j]=[]

        # randomize the position of agents
        agent_counter = 0
        self.start_list = []
        while agent_counter < self.global_num_agent:
            x, y = np.random.randint(0, self.size), np.random.randint(0, self.size)
            if self.fix_state[x, y] == 0:
                self.fix_state[x, y] +=1
                self.fix_state_dict[x,y].append(agent_counter)
                self.start_list.append((x, y))
                agent_counter += 1
        assert(sum(sum(self.fix_state)) == self.global_num_agent + sum(sum(self.map)))

        # randomize the position of goals
        goals = np.zeros((int(self.size), int(self.size))).astype(int)
        goal_counter = 0
        agent_regions = dict()
        self.goal_list = []
        while goal_counter < self.global_num_agent:
            agent_pos = self.start_list[goal_counter]
            valid_tiles = get_connected_region(self.fix_state, agent_regions, agent_pos[0], agent_pos[1])
            x, y = random.choice(list(valid_tiles))
            if goals[x, y] == 0 and self.fix_state[x, y] != -1:
                goals[x, y] = goal_counter+1
                self.goal_list.append((x, y))
                goal_counter += 1

        self.world = State(self.fix_state, self.fix_state_dict, self.global_num_agent, self.start_list, self.goal_list)

    def joint_move(self, actions):
        """simultaneously move agents and checks for collisions on the joint action """
        if self.time_step<self.dynamic_state.max_lens:
            self.world.state = self.dynamic_state.state[self.time_step]+self.map
        else:
            self.world.state = self.dynamic_state.state[-1]+self.map
        for i in range(self.global_num_agent):
            if i not in self.world.local_agents:
                max_len = len(self.paths[i])
                if max_len <= self.time_step:
                    continue
                else:
                    self.world.agents_poss[i] = self.paths[i][self.time_step]
                    self.world.state_dict[self.paths[i][self.time_step - 1]].remove(i)
                    self.world.state_dict[self.paths[i][self.time_step]].append(i)

        local_past_position = copy.copy(self.world.local_agents_poss)
        dynamic_collision_status=np.zeros(self.local_num_agents)
        agent_collision_status = np.zeros(self.local_num_agents)
        reach_goal_status = np.zeros(self.local_num_agents)

        self.agent_util_map_action.pop(0)
        self.agent_util_map_vertex.pop(0)
        self.agent_util_map_action.append(np.zeros((5, self.map.shape[0], self.map.shape[1])))
        self.agent_util_map_vertex.append(np.zeros((self.map.shape[0], self.map.shape[1])))

        for local_i, i in enumerate(self.world.local_agents):
            direction = self.world.get_dir(actions[local_i])
            ax = self.world.local_agents_poss[local_i][0]
            ay = self.world.local_agents_poss[local_i][1]
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.world.state.shape[0] or ax + dx < 0 or ay + dy >= self.world.state.shape[1] or ay + dy < 0:
                raise ValueError("out of boundaries")

            if self.map[ax + dx, ay + dy] < 0:
                raise ValueError("collide with static obstacles")

            self.world.agents_poss[i] = (ax + dx, ay + dy)  # update agent's current position
            self.world.local_agents_poss[local_i] = (ax + dx, ay + dy)
            self.world.state[ax + dx, ay + dy] += 1
            self.world.state_dict[ax, ay].remove(i)
            self.world.state_dict[ax + dx, ay + dy].append(i)
            self.agent_util_map_action[-1][int(actions[local_i]),ax + dx, ay + dy]+=1
            self.agent_util_map_vertex[-1][ax + dx, ay + dy]+=1

        for local_i, i in enumerate(self.world.local_agents):
            if self.world.state[self.world.local_agents_poss[local_i]] > 1:
                collide_agents_index = self.world.state_dict[self.world.local_agents_poss[local_i]]
                assert (len(collide_agents_index)==self.world.state[self.world.local_agents_poss[local_i]])
                for j in collide_agents_index:
                    if j!=i:
                        if j in self.world.local_agents:
                            agent_collision_status[local_i]+=1
                        else:
                            dynamic_collision_status[local_i]+=1
                        self.new_collision_pairs.add((min(j , i), max(j, i)))

            if self.world.state[local_past_position[local_i]] > 0:
                collide_agents_index = self.world.state_dict[local_past_position[local_i]]  # now=past
                assert (len(collide_agents_index) == self.world.state[local_past_position[local_i]])
                for j in collide_agents_index:
                    if j!=i:
                        if j in self.world.local_agents:  # past=now
                            local_j=self.world.local_agents.index(j)
                            past_poss =local_past_position[local_j]
                            if past_poss==self.world.local_agents_poss[local_i] and self.world.agents_poss[j]!=past_poss:
                                agent_collision_status[local_i] += 1
                                self.new_collision_pairs.add((min(j , i), max(j , i)))
                        else:
                            max_len=len(self.paths[j])
                            if max_len<=self.time_step:
                                continue
                            else:
                                past_poss=self.paths[j][self.time_step-1]
                                if past_poss== self.world.local_agents_poss[local_i] and past_poss!=self.paths[j][self.time_step]:
                                    dynamic_collision_status[local_i] += 1
                                    self.new_collision_pairs.add((min(j, i), max(j , i)))

            if self.world.local_agents_poss[local_i] == self.goal_list[i]:
                reach_goal_status[local_i] = 1

        return dynamic_collision_status,agent_collision_status,reach_goal_status

    def observe(self, local_agent_index):
        """return one agent's observation"""
        agent_index = self.world.local_agents[local_agent_index]

        top_poss = max(self.world.agents_poss[agent_index][0] - self.fov_size // 2, 0)
        bottom_poss = min( self.world.agents_poss[agent_index][0] + self.fov_size // 2+1, self.size)
        left_poss = max(self.world.agents_poss[agent_index][1] - self.fov_size // 2, 0)
        right_poss = min(self.world.agents_poss[agent_index][1] + self.fov_size // 2+1, self.size)
        top_left = (self.world.agents_poss[agent_index][0] - self.fov_size // 2,
                    self.world.agents_poss[agent_index][1] - self.fov_size // 2)
        FOV_top, FOV_left = max(self.fov_size // 2 - self.world.agents_poss[agent_index][0], 0), max(self.fov_size // 2 - self.world.agents_poss[agent_index][1], 0)
        FOV_bottom, FOV_right = FOV_top + (bottom_poss - top_poss), FOV_left + (right_poss- left_poss)

        obs_shape = (self.fov_size, self.fov_size)
        goal_map = np.zeros(obs_shape)  # own goal
        local_poss_map = np.zeros(obs_shape)  # agents
        local_goals_map = np.zeros(obs_shape)  # other observable agents' goal
        obs_map = np.ones(obs_shape)  # obstacle
        guide_map= np.zeros((4,obs_shape[0],obs_shape[1]))
        visible_agents = set()
        dynamic_poss_maps= np.zeros((EnvParameters.NUM_TIME_SLICE, self.fov_size, self.fov_size))
        sipps_map = np.zeros(obs_shape)
        util_map_action=np.zeros((5, self.fov_size, self.fov_size))
        util_map = np.zeros(obs_shape)
        blank_map = np.zeros(obs_shape)
        occupy_map = np.zeros(obs_shape)
        next_step_map = np.zeros((EnvParameters.K_STEPS, self.fov_size, self.fov_size))

        if self.time_step-EnvParameters.WINDOWS<0:
            min_time=0
        elif self.time_step>=len(self.sipps_path[local_agent_index]):
            min_time = max(0, len(self.sipps_path[local_agent_index]) - EnvParameters.WINDOWS)
        else:
            min_time =self.time_step- EnvParameters.WINDOWS

        max_time = min(self.time_step+EnvParameters.WINDOWS, len(self.sipps_path[local_agent_index]))

        window_path=self.sipps_path[local_agent_index][min_time:max_time]

        if self.goal_list[agent_index][0] in range(top_poss, bottom_poss) and self.goal_list[agent_index][1] in range(left_poss, right_poss):
            goal_map[self.goal_list[agent_index][0] - top_left[0], self.goal_list[agent_index][1] - top_left[1]] = 1
        local_poss_map[self.world.agents_poss[agent_index][0] - top_left[0], self.world.agents_poss[agent_index][1] - top_left[1]] = 1
        obs_map[FOV_top:FOV_bottom, FOV_left:FOV_right] = -self.map[top_poss:bottom_poss, left_poss:right_poss]
        guide_map[:, FOV_top:FOV_bottom, FOV_left:FOV_right] = self.world.heuri_map[agent_index][:,top_poss:bottom_poss, left_poss:right_poss]
        util_map[FOV_top:FOV_bottom, FOV_left:FOV_right] = self.space_ulti_vertex[top_poss:bottom_poss, left_poss:right_poss]
        util_map_action[:,FOV_top:FOV_bottom, FOV_left:FOV_right] = self.space_ulti_action[:,top_poss:bottom_poss,
                                                           left_poss:right_poss]

        for i in range(top_left[0], top_left[0] + self.fov_size):
            for j in range(top_left[1], top_left[1] + self.fov_size):
                if i >= self.size or i < 0 or j >= self.size or j < 0:
                    # out of boundaries
                    occupy_map[i - top_left[0], j - top_left[1]] = 1-self.time_step/ self.episode_len
                    continue
                if self.world.state[i, j] == -1:
                    # obstacles
                    occupy_map[i - top_left[0], j - top_left[1]] = 1 - self.time_step/ self.episode_len
                    continue
                if (i,j) in window_path:
                    sipps_map[i - top_left[0], j - top_left[1]]=1
                for iter_a in range(self.local_num_agents):
                    if iter_a!=local_agent_index:
                        for k in range(EnvParameters.K_STEPS):
                            if (i,j)==self.all_next_poss[iter_a][k]:
                                next_step_map[k,i - top_left[0], j - top_left[1]]+=1
                if self.world.state[i, j] > 0:
                    for item in self.world.state_dict[i,j]:
                        if item in self.world.local_agents and item !=agent_index:
                            visible_agents.add(item)
                            local_poss_map[i - top_left[0], j - top_left[1]] += 1

                for t in range(EnvParameters.NUM_TIME_SLICE):
                    if self.time_step + t < self.dynamic_state.max_lens:
                        dynamic_poss_maps[t,i - top_left[0], j - top_left[1]] = self.dynamic_state.state[self.time_step + t,i,j]
                    else:
                        dynamic_poss_maps[t, i - top_left[0], j - top_left[1]] = self.dynamic_state.state[-1, i, j]

                if self.time_step>=self.makespan:
                    if self.dynamic_state.state[-1, i, j] > 0:
                        occupy_map[i - top_left[0], j - top_left[1]] = 1-self.time_step/ self.episode_len
                    else:
                        blank_map[i - top_left[0], j - top_left[1]] = 1-(self.time_step+1)/ self.episode_len
                else:
                    occupy_t=0
                    if self.dynamic_state.state[self.time_step, i, j] > 0:
                        for t in range(self.time_step, self.episode_len+ 1):
                            if t >=self.makespan:
                                if self.dynamic_state.state[-1, i, j] > 0:
                                    occupy_t=self.episode_len-self.time_step
                                break
                            if self.dynamic_state.state[t, i, j] > 0:
                                occupy_t += 1
                            else:
                                break
                    occupy_map[i - top_left[0], j - top_left[1]] = occupy_t/ self.episode_len
                    blank_t =0
                    for t in range(self.time_step+1, self.episode_len + 1):
                        if t >= self.makespan:
                            if self.dynamic_state.state[-1, i, j]==0:
                                blank_t = self.episode_len- self.time_step-1
                            break
                        if self.dynamic_state.state[t, i, j] == 0:
                            blank_t += 1
                        else:
                            break
                    blank_map[i - top_left[0], j - top_left[1]] = blank_t/ self.episode_len

        zero_mask = local_poss_map == 0
        local_poss_map=0.5+0.5*np.tanh((local_poss_map-1)/3)
        local_poss_map[zero_mask]=0
        zero_mask = next_step_map == 0
        next_step_map=0.5+0.5*np.tanh((next_step_map-1)/3)
        next_step_map[zero_mask]=0
        zero_mask = dynamic_poss_maps == 0
        dynamic_poss_maps=0.5+0.5*np.tanh((dynamic_poss_maps-1)/3)
        dynamic_poss_maps[zero_mask]=0

        for vis_agent_index in visible_agents:
            x, y = self.world.agents_goals[vis_agent_index]
            # project the goal out of FOV to the boundary of FOV
            min_node = (max(top_left[0], min(top_left[0] + self.fov_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.fov_size - 1, y)))
            local_goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx = self.world.agents_goals[agent_index][0] - self.world.agents_poss[agent_index][0]  # distance on x axes
        dy = self.world.agents_goals[agent_index][1] - self.world.agents_poss[agent_index][1]  # distance on y axes
        mag = (dx ** 2 + dy ** 2) ** .5  # total distance
        if mag != 0:  # normalized
            dx = dx / mag
            dy = dy / mag

        window_path = np.array(window_path)
        diff = window_path - self.world.agents_poss[agent_index]
        x = diff[:, 0]
        y = diff[:, 1]
        distance = np.sqrt(x ** 2 + y ** 2)
        off_rout_penalty = -np.min(distance) * self.off_route_factor

        return [dynamic_poss_maps[0],dynamic_poss_maps[1],dynamic_poss_maps[2],dynamic_poss_maps[3],
                dynamic_poss_maps[4],dynamic_poss_maps[5],dynamic_poss_maps[6],dynamic_poss_maps[7],dynamic_poss_maps[8],local_poss_map, goal_map, local_goals_map,
                obs_map,guide_map[0],guide_map[1],guide_map[2],guide_map[3],sipps_map,blank_map,occupy_map,util_map,util_map_action[0]
                ,util_map_action[1],util_map_action[2], util_map_action[3],util_map_action[4],next_step_map[0],next_step_map[1],next_step_map[2],next_step_map[3],next_step_map[4]], [dx, dy, mag],off_rout_penalty

    def predict_next(self):
        self.all_next_poss = []
        if self.time_step!=0:
            for local_agent_index in range(self.local_num_agents):
                next_poss_list = []
                for k in range(EnvParameters.K_STEPS):
                    if k == 0:
                        dis_x = self.world.local_agents_poss[local_agent_index][0] - np.array(self.sipps_path[local_agent_index])[:, 0]
                        dis_y = self.world.local_agents_poss[local_agent_index][1] - np.array(self.sipps_path[local_agent_index])[:, 1]
                        dis = np.sqrt(dis_x ** 2 + dis_y ** 2)
                        time_dis = np.abs(self.time_step - np.array(range(len(self.sipps_path[local_agent_index]))))
                        final_dis = dis * EnvParameters.DIS_TIME_WEIGHT[0] + time_dis * EnvParameters.DIS_TIME_WEIGHT[1]
                        poss_index = np.argmin(final_dis)
                        if poss_index + 1 < len(self.sipps_path[local_agent_index]):
                            next_poss = self.sipps_path[local_agent_index][poss_index + 1]
                        else:
                            next_poss = self.sipps_path[local_agent_index][-1]
                        pre_dis_x = next_poss[0] - self.world.local_agents_poss[local_agent_index][0]
                        pre_dis_y = next_poss[1] - self.world.local_agents_poss[local_agent_index][1]
                        pre_dis = pre_dis_x ** 2 + pre_dis_y ** 2
                        if pre_dis > 1:
                            next_poss = self.world.local_agents_poss[local_agent_index]
                    else:
                        if poss_index + k + 1 < len(self.sipps_path[local_agent_index]):
                            next_poss = self.sipps_path[local_agent_index][poss_index + k + 1]
                        else:
                            next_poss = self.sipps_path[local_agent_index][-1]
                    next_poss_list.append(next_poss)
                self.all_next_poss.append(next_poss_list)
        else:
            for local_agent_index in range(self.local_num_agents):
                next_poss_list = []
                for k in range(EnvParameters.K_STEPS):
                    if k+1<len(self.sipps_path[local_agent_index]):
                        next_poss = self.sipps_path[local_agent_index][k+1]
                    else:
                        next_poss = self.sipps_path[local_agent_index][-1]
                    next_poss_list.append(next_poss)
                self.all_next_poss.append(next_poss_list)

    def update_ulti(self):
        self.space_ulti_action=np.zeros((5,self.map.shape[0],self.map.shape[1]))
        self.space_ulti_vertex = np.zeros(self.map.shape)
        for t in EnvParameters.UTI_WINDOWS:
            if self.time_step + t + 1 < 0:
                continue
            if t < 0:
                self.space_ulti_action += self.agent_util_map_action[t + 2]
                self.space_ulti_vertex += self.agent_util_map_vertex[t + 2]

            if self.time_step + t + 1 >= self.dynamic_state.max_lens:
                self.space_ulti_action[0,:,:] += self.dynamic_state.state[-1]
                self.space_ulti_vertex += self.dynamic_state.state[-1]
            else:
                self.space_ulti_action += self.dynamic_state.util_map_action[self.time_step + t + 1]
                self.space_ulti_vertex += self.dynamic_state.state[self.time_step + t + 1]
        self.space_ulti_vertex=10*self.space_ulti_vertex/self.global_num_agent
        self.space_ulti_action = 10 * self.space_ulti_action / self.global_num_agent

    def joint_step(self, actions):
        """execute joint action and obtain reward"""
        self.time_step+=1
        dynamic_collision_status,agent_collision_status,reach_goal_status= self.joint_move(actions)

        rewards = np.zeros((1, self.local_num_agents), dtype=np.float32)
        obs = np.zeros((1, self.local_num_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.local_num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        next_valid_actions = []
        for i in range(self.local_num_agents):
            rewards[:, i] += EnvParameters.OVERALL_WEIGHT * EnvParameters.UTI_WEIGHT[1] * self.space_ulti_vertex[
                self.world.local_agents_poss[i]]
            rewards[:, i] += EnvParameters.OVERALL_WEIGHT * EnvParameters.UTI_WEIGHT[0] * \
                             self.space_ulti_action[int(actions[i]),
                                 self.world.local_agents_poss[i][0],self.world.local_agents_poss[i][1]]
        self.update_ulti()  # update space utilization before observe it
        self.predict_next()
        for i in range(self.local_num_agents):
            rewards[:, i] += EnvParameters.DY_COLLISION_COST*dynamic_collision_status[i]
            rewards[:, i] += EnvParameters.AG_COLLISION_COST*agent_collision_status[i]

            if reach_goal_status[i] == 1:
                rewards[:, i] += EnvParameters.GOAL_REWARD
            else:
                if actions[i] == opposite_actions[self.previous_action[i]]:
                    rewards[:, i] += EnvParameters.MOVE_BACK_COST
                if actions[i] == 0:
                    rewards[:, i] += self.idle_cost
                else:
                    rewards[:, i] += self.action_cost

                if self.time_step>self.sipps_max_len:
                    rewards[:, i] += EnvParameters.ADD_COST

            dis=np.sqrt(np.square(self.world.local_agents_poss[i][0] - self.world.local_agents_goal[i][0])+np.square(self.world.local_agents_poss[i][1] - self.world.local_agents_goal[i][1]))
            rewards[:, i]-=EnvParameters.DIS_FACTOR*(TrainingParameters.GAMMA*dis-self.world.old_dis[i])
            self.world.old_dis[i]=dis

            state = self.observe(i)
            rewards[:, i] += state[-1]
            obs[:, i, :, :, :] = state[0]
            vector[:, i, : 3] = state[1]
            next_valid_actions.append(self.world.list_next_valid_actions(i))

        num_dynamic_collide=sum( dynamic_collision_status)
        num_agent_collide=sum(agent_collision_status)
        num_on_goal=sum(reach_goal_status)
        real_r=EnvParameters.DY_COLLISION_COST*num_dynamic_collide+EnvParameters.AG_COLLISION_COST*num_agent_collide+EnvParameters.GOAL_REWARD*num_on_goal+(self.local_num_agents-num_on_goal)* self.action_cost
        self.previous_action=actions
        all_reach_goal=(num_on_goal == self.local_num_agents)
        vector[:, :, 3] = (self.sipp_coll_pair_num - len(self.new_collision_pairs)) / (self.sipp_coll_pair_num + 1)
        vector[:, :, 4] = self.time_step/self.episode_len
        vector[:, :, 5] = self.time_step/self.sipps_max_len
        vector[:, :, 6] = num_on_goal / self.local_num_agents
        vector[:, :, 7] = actions
        done=False
        success=False
        if all_reach_goal and self.time_step>=self.makespan:
            done = True
            if len(self.new_collision_pairs)<=self.sipp_coll_pair_num:
                success = True
        if self.time_step >= self.episode_len:
            done = True
        return obs, vector, rewards, done, next_valid_actions,num_on_goal, num_dynamic_collide,num_agent_collide,success,real_r

    def _global_reset(self,cl_num_task):
        """restart a new task"""
        self.global_set_world(cl_num_task)  # back to the initial situation
        self.lns2_model=my_lns2.MyLns2(self.env_id*123,self.map,self.start_list,self.goal_list,self.global_num_agent,self.map.shape[0])
        self.lns2_model.init_pp()
        self.paths=self.lns2_model.vector_path
        self.dynamic_state = DyState(self.paths,self.global_num_agent,self.map.shape)
        self.idle_cost= EnvParameters.IDLE_COST[cl_num_task]
        self.action_cost= EnvParameters.ACTION_COST[cl_num_task]
        self.off_route_factor=EnvParameters.OFF_ROUTE_FACTOR[cl_num_task]
        return

    def _local_reset(self, local_num_agents,first_time):
        """restart a new task"""
        self.local_num_agents = local_num_agents
        new_agents = random.sample(range(self.global_num_agent), local_num_agents)
        self.time_step=0
        self.previous_action=np.zeros(local_num_agents)
        if not first_time:
            prev_path ={}
            for local_index in range(self.local_num_agents):
                prev_path[self.world.local_agents[local_index]]=self.sipps_path[local_index]
            prev_agents=self.local_agents
        else:
            prev_agents = None
            prev_path = None  # old agents, new path
        path_new_agent ={}
        for global_index in new_agents:
            path_new_agent[global_index]=self.paths[global_index] # new agents. old path
        if not first_time:
            for local_index in range(self.local_num_agents):
                self.paths[self.world.local_agents[local_index]] = self.sipps_path[local_index] # old agents, new path
        self.local_agents = new_agents
        self.sipp_coll_pair_num=self.lns2_model.calculate_sipps(self.local_agents)
        self.makespan = self.lns2_model.makespan
        self.sipps_path=self.lns2_model.sipps_path
        self.sipps_max_len=max([len(path) for path in self.sipps_path])
        self.dynamic_state.reset_local_tasks(self.local_agents,path_new_agent, prev_agents, prev_path,self.makespan+1)
        self.world.reset_local_tasks(self.fix_state,self.fix_state_dict,self.start_list,self.local_agents)
        self.agent_util_map_action =[np.zeros((5, self.map.shape[0], self.map.shape[1])) for _ in range(2)]
        self.agent_util_map_vertex = [np.zeros((self.map.shape[0], self.map.shape[1])) for _ in range(2)]
        for local_i in range(self.local_num_agents):
            self.agent_util_map_vertex[-1][self.world.local_agents_poss[local_i]]+=1
        self.new_collision_pairs = set()
        self.update_ulti()
        self.predict_next()
        return

    def list_next_valid_actions(self,local_agent_index):
        return self.world.list_next_valid_actions(local_agent_index)
