import numpy as np
dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}  # 0: stay, 1:right, 2: down, 3:left, 4:up
actionDict = {v: k for k, v in dirDict.items()}


class DyState(object):  # world property
    def __init__(self, all_path,global_num_agents,world_shape):
        """initialization"""
        self.max_lens=max([len(path) for path in all_path])
        self.world_shape=world_shape
        self.global_num_agents = global_num_agents
        self.local_num_agents = None
        self.util_map_action=[np.zeros((5, world_shape[0], world_shape[1])) for _ in range(self.max_lens)]
        self.state = np.zeros((self.max_lens, world_shape[0], world_shape[1]))
        for a in range(self.global_num_agents):
            max_len = len(all_path[a])
            for t in range(self.max_lens):
                if max_len <= t:
                    poss = all_path[a][-1]
                else:
                    poss = all_path[a][t]
                self.state[t][poss]+=1
                if t>0:
                    if max_len <= t-1:
                        prev_poss = all_path[a][-1]
                    else:
                        prev_poss = all_path[a][t-1]
                    action=actionDict[(poss[0]-prev_poss[0],poss[1]-prev_poss[1])]
                    self.util_map_action[t][action][poss]+=1

    def reset_local_tasks(self, new_agents, path_new_agent, prev_agents, old_path, new_max_len):
        self.local_num_agents = len(new_agents)
        diff= self.max_lens - new_max_len
        self.max_lens = new_max_len
        if diff>0:
            self.state = self.state[:self.max_lens, :, :]
            self.util_map_action= self.util_map_action[:self.max_lens]
        if diff<0:
            for _ in range(int(-diff)):
                self.util_map_action.append(np.zeros((5, self.world_shape[0], self.world_shape[1])))
                self.util_map_action[-1][0]+=self.state[-1]
            last_state =np.expand_dims(self.state[-1],axis=0)
            last_state=np.repeat(last_state,-diff ,axis=0)
            self.state=np.concatenate([self.state,last_state],axis=0)

        if prev_agents != None:
            delete_agent=set(new_agents)-set(prev_agents)
        else:
            delete_agent =new_agents

        for i in delete_agent: # delete
            max_len = len(path_new_agent[i])
            for t in range(self.max_lens):
                if max_len <= t:
                    poss = path_new_agent[i][-1]
                else:
                    poss = path_new_agent[i][t]
                self.state[t][poss] -= 1
                assert self.state[t][poss]>=0
                if t>0:
                    if max_len <= t-1:
                        prev_poss = path_new_agent[i][-1]
                    else:
                        prev_poss = path_new_agent[i][t-1]
                    action=actionDict[(poss[0]-prev_poss[0],poss[1]-prev_poss[1])]
                    self.util_map_action[t][action][poss]-=1

        if prev_agents != None:
            add_agent=set(prev_agents)-set(new_agents)
            for i in add_agent:  # add
                max_len = len(old_path[i])
                for t in range(self.max_lens):
                    if max_len <= t:
                        poss = old_path[i][-1]
                    else:
                        poss = old_path[i][t]
                    self.state[t][poss] += 1
                    if t > 0:
                        if max_len <= t - 1:
                            prev_poss = old_path[i][-1]
                        else:
                            prev_poss = old_path[i][t - 1]
                        action = actionDict[(poss[0] - prev_poss[0], poss[1] - prev_poss[1])]
                        self.util_map_action[t][action][poss] += 1
        assert (sum(sum(sum(self.state))) == self.max_lens * (self.global_num_agents - self.local_num_agents))
