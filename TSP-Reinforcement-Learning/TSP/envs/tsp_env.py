from gymnasium import Env, spaces
import numpy as np


class TSPEnv(Env):
    def __init__(self, graph, names):
        self.N = len(graph)
        self.cities = names
        self.invalid_action_cost = -100
        self.distance_matrix = graph

        self.obs_dim = 1+self.N
        obs_space = spaces.Box(-1, self.N, shape=(self.obs_dim,), dtype=np.int64)
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(0, 1, shape=(self.N,), dtype=np.int64),
            "avail_actions": spaces.Box(0, 1, shape=(self.N,), dtype=np.int64),
            "state": obs_space
        })

        self.action_space = spaces.Discrete(self.N)
        self.cost = 0
        self.start_city = None
        
        self.reset()

    def step(self, action):
        done = False
        if not self.start_city:
            self.start_city = action

        # check if visited node
        if self.visit_log[action] > 0:
            # If all nodes have been visited
            if self.visit_log.sum() == self.N:
                # Check if the last action is the start city
                if action == self.start_city:
                    reward = -self.distance_matrix[self.current_node, action]
                    self.current_node = action
                    self.cost -= reward
                    done = True
                else:
                    # Penalize for invalid action
                    reward = self.invalid_action_cost
                    done = True
            else:
                # Node already visited
                reward = self.invalid_action_cost
                done = True
        else:
            if self.current_node == -1:
                # if first node, no cost
                reward = 0
            else:
                reward = -self.distance_matrix[self.current_node, action]
            self.current_node = action
            self.visit_log[self.current_node] = 1
            self.cost -= reward
            
        self.state = self._update_state()
            
        return self.state, reward, False, done, {"current_node": self.current_node}

    def reset(self, **kwargs):
        self.step_count = 0
        self.cost = 0
        self.current_node = -1
        self.visit_log = np.zeros(self.N)
        self.start_city = None
        
        self.state = self._update_state()
        return self.state, {"current_node": self.current_node}

    def _update_state(self):
        mask = np.where(self.visit_log==0, 0 , 1)
        obs = np.hstack([self.current_node, mask])
        state = {
            "avail_actions": np.ones(self.N),
            "action_mask": mask,
            "state": obs
        }
        return state
    
    def render(self):
        current_city = self.cities[int(self.current_node)]
        print("Current Node: ", current_city)
        visit_log = [self.cities[i] for i, _ in enumerate(self.visit_log)]
        print("Visit Log: ", self.visit_log)
        print("Travel Cost: ", self.cost)