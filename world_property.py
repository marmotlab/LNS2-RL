import copy
import numpy as np
from alg_parameters import *

dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0), 5: (1, 1), 6: (1, -1), 7: (-1, -1),
           8: (-1, 1)}  # x,y operation for corresponding action
actionDict = {v: k for k, v in dirDict.items()}
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

class State(object):  # world property
    def __init__(self, state,state_dict, global_num_agents,start_list,goal_list):
        """initialization"""
        self.state = copy.copy(state)
        self.state_dict=copy.deepcopy(state_dict)
        self.agents_poss=copy.copy(start_list)
        self.agents_goals = goal_list
        self.heuri_map={i:[] for i in range(global_num_agents)}
        assert (len(self.agents_poss) == global_num_agents)

    def reset_local_tasks(self,state,state_dict,start_list,local_agents):
        self.state = copy.copy(state)
        self.state_dict = copy.deepcopy(state_dict)
        self.agents_poss = copy.copy(start_list)
        self.local_agents=local_agents
        local_num_agents=len(local_agents)
        self.local_agents_poss=[]
        self.local_agents_goal=[]
        self.old_dis = np.zeros(local_num_agents)
        for local_i, i in enumerate(local_agents):
            self.local_agents_poss.append(self.agents_poss[i])
            self.local_agents_goal.append(self.agents_goals[i])
            self.old_dis[local_i] = np.sqrt(np.square(self.local_agents_poss[local_i][0] - self.local_agents_goal[local_i][0])+np.square(self.local_agents_poss[local_i][1] - self.local_agents_goal[local_i][1]))
        self.get_heuri_map()

    def get_dir(self, action):
        """obtain corresponding x,y operation based on action"""
        return dirDict[action]

    def get_action(self, direction):
        """obtain corresponding action based on x,y operation"""
        return actionDict[direction]

    def list_next_valid_actions(self, local_agent_index):
        """obtain the valid actions that can not lead to colliding with obstacles and boundaries
        or backing to previous position at next time step"""
        available_actions = [0]  # staying still always allowed

        agent_pos = self.local_agents_poss[local_agent_index]
        ax, ay = agent_pos[0], agent_pos[1]

        for action in range(1, EnvParameters.N_ACTIONS):  # every action except 0
            direction = self.get_dir(action)
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.state.shape[0] or ax + dx < 0 or ay + dy >= self.state.shape[
                    1] or ay + dy < 0:  # out of boundaries
                continue
            if self.state[ax + dx, ay + dy] < 0:  # collide with static obstacles
                continue
            # otherwise we are ok to carry out the action
            available_actions.append(action)
        return available_actions

    def get_heuri_map(self):
        for a in self.local_agents:
            if len(self.heuri_map[a])==0:
                dist_map = np.ones((self.state.shape), dtype=np.int32) * 2147483647
                open_list = list()
                x, y = tuple(self.agents_goals[a])
                open_list.append((x, y))
                dist_map[x, y] = 0

                while open_list:
                    x, y = open_list.pop(0)
                    dist = dist_map[x, y]

                    up = x - 1, y
                    if up[0] >= 0 and self.state[up] != -1 and dist_map[x - 1, y] > dist + 1:
                        dist_map[x - 1, y] = dist + 1
                        if up not in open_list:
                            open_list.append(up)

                    down = x + 1, y
                    if down[0] < self.state.shape[0] and self.state[down] != -1 and dist_map[x + 1, y] > dist + 1:
                        dist_map[ x + 1, y] = dist + 1
                        if down not in open_list:
                            open_list.append(down)

                    left = x, y - 1
                    if left[1] >= 0 and self.state[left] != -1 and dist_map[x, y - 1] > dist + 1:
                        dist_map[x, y - 1] = dist + 1
                        if left not in open_list:
                            open_list.append(left)

                    right = x, y + 1
                    if right[1] < self.state.shape[1] and self.state[right] != -1 and dist_map[x, y + 1] > dist + 1:
                        dist_map[x, y + 1] = dist + 1
                        if right not in open_list:
                            open_list.append(right)

                self.heuri_map[a] = np.zeros((4, *self.state.shape), dtype=bool)

                for x in range(self.state.shape[0]):
                    for y in range(self.state.shape[1]):
                        if self.state[x, y] != -1:
                            if x > 0 and dist_map[x - 1, y] < dist_map[x, y]:
                                assert dist_map[x - 1, y] == dist_map[x, y] - 1
                                self.heuri_map[a][0, x, y] = 1

                            if x < self.state.shape[0] - 1 and dist_map[ x + 1, y] < dist_map[x, y]:
                                assert dist_map[x + 1, y] == dist_map[x, y] - 1
                                self.heuri_map[a][1, x, y] = 1

                            if y > 0 and dist_map[ x, y - 1] < dist_map[ x, y]:
                                assert dist_map[ x, y - 1] == dist_map[ x, y] - 1
                                self.heuri_map[a][2, x, y] = 1

                            if y < self.state.shape[1] - 1 and dist_map[ x, y + 1] < dist_map[ x, y]:
                                assert dist_map[x, y + 1] == dist_map[x, y] - 1
                                self.heuri_map[a][3, x, y] = 1

    def get_pos(self, agent_id):
        """agent's current position"""
        return self.agents_poss[agent_id - 1]

    def get_goal(self, agent_id):
        """the position of agent's goal"""
        return self.agents_goals[agent_id - 1]
