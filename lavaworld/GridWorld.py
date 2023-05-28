import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
from collections import defaultdict

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "text.latex.preamble":r'\usepackage{pifont,marvosym,scalerel}'
})

# Define colors
COLOURS = {0: [1, 1, 1], 1: [0.0, 0.0, 0.0], 3: [0, 0.5, 0], 10: [0, 0, 1], 20:[1, 1, 0.0], 21:[0.8, 0.8, 0.8]}

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}
    MAP =   "1 1 1 1 1 1\n" \
            "1 0 0 0 0 1\n" \
            "1 0 1 0 0 1\n" \
            "1 0 0 0 0 1\n" \
            "1 1 1 1 1 1"
    # MAP =   "1 1 1 1 1 1 1 1 1 1 1 1\n" \
    #         "1 0 0 0 0 0 0 0 0 0 0 1\n" \
    #         "1 0 0 0 0 0 0 0 0 0 0 1\n" \
    #         "1 0 0 0 1 1 1 1 0 0 0 1\n" \
    #         "1 0 0 0 1 1 1 1 0 0 0 1\n" \
    #         "1 0 0 0 1 1 1 1 0 0 0 1\n" \
    #         "1 0 0 0 0 0 0 0 0 0 0 1\n" \
    #         "1 0 0 0 0 0 0 0 0 0 0 1\n" \
    #         "1 1 1 1 1 1 1 1 1 1 1 1"

    def __init__(self, MAP=MAP, dense_goal_rewards = False, dense_rewards = False, lava_reward=-1, goal_reward=1, step_reward=-0.1, wall_reward=-1, slip_prob=0, goals=None, T_states=None, start_position=None):

        self.n = None
        self.m = None

        self.grid = None
        self.hallwayStates = None
        self.possibleStates = []
        self.walls = []
        
        self.MAP = MAP
        self._map_init()  
        self.diameter = (self.n+self.m)-4

        self.done = False
        
        self.start_position = start_position
        self.position = self.start_position if start_position else (self.m-2, self.n-2)
        self.state = self.position
        self.slip_prob = slip_prob
        
        if goals:
            self.goals = goals
        else:
            self.goals = (1, 1) 
            
        if T_states:
            self.T_states = T_states
        else:
            self.T_states = self.goals

        # Rewards
        self.goal_reward = goal_reward
        self.lava_reward = lava_reward
        self.step_reward = step_reward
        self.wall_reward = wall_reward
        self.rmax = max([self.goal_reward, self.lava_reward, self.step_reward, self.wall_reward]) 
        self.rmin = min([self.goal_reward, self.lava_reward, self.step_reward, self.wall_reward]) 
        
        self.dense_goal_rewards = dense_goal_rewards
        self.dense_rewards = dense_rewards
        if self.dense_rewards:
            self.rmin = self.rmin*10

        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(len(self.possibleStates))
        self.action_space = spaces.Discrete(4)
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        np.random.rand(seed)
        return [seed]
    
    def pertube_action(self,action):        
        a = 0.8
        b = (1-a)/2
        if action == UP:
            probs = [a,b,0,b,0]
            action = np.random.choice(np.arange(len(probs)), p=probs)
        elif action == DOWN:
            probs = [0,b,a,b,0]
            action = np.random.choice(np.arange(len(probs)), p=probs)
        elif action == RIGHT:
            probs = [b,a,b,0,0]
            action = np.random.choice(np.arange(len(probs)), p=probs)
        elif action == LEFT:
            probs = [b,0,b,a,0]
            action = np.random.choice(np.arange(len(probs)), p=probs)       
        return action
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        x, y = self.state     
        # action_ = self.pertube_action(action)
        if np.random.random()>self.slip_prob:
            action_ = action
        else:
            action_ = self.action_space.sample()
            # action_ = np.random.choice([(action-1)%self.action_space.n, (action+1)%self.action_space.n])
        if action_ == UP:
            x = x - 1
        elif action_ == DOWN:
            x = x + 1
        elif action_ == RIGHT:
            y = y + 1
        elif action_ == LEFT:
            y = y - 1
        self.position = (x, y)
        new_state = self.position
        
        reward = self._get_reward(self.state, action, new_state)
        
        if self._get_grid_value(new_state) == 1:  # new_state in walls list
            # stay at old state if new coord is wall
            self.position = self.state
        else:
            self.state = new_state

        done = self.state in self.T_states
        unsafe = done and not (self.state in self.goals)
        
        return self.state, reward, done, {"unsafe": unsafe}
    
    def _get_dense_reward(self, state, action):
        g = np.array([g for g in self.goals])
        s = np.array([state]*len(g))
        reward = 0.1*np.mean(np.exp(-0.25*np.linalg.norm(s-g, axis=1)**2))
        return reward

    def _get_reward(self, state, action, new_state):      
        reward = 0        
        if self.dense_rewards:
            reward += self._get_dense_reward(new_state,action)

        if new_state in self.T_states:
            if self.dense_goal_rewards:
                g = np.array([g for g in self.goals])
                s = np.array([new_state]*len(g))
                # steps = np.abs(s-g).max()
                steps = np.abs(s-g).sum(axis=1).min()
                rewards = np.linspace(self.goal_reward,self.step_reward,self.n-2)
                reward += rewards[steps]
            else:
                reward += self.goal_reward if new_state in self.goals else self.lava_reward
        else:
            if self._get_grid_value(new_state) == 1:  # new_state in walls list
                reward += self.wall_reward
            else:
                reward += self.step_reward
        
        return reward
        
    def get_rewards(self):
        R = defaultdict(lambda: np.zeros(self.action_space.n))
        for state in self.possibleStates: 
            for action in range(self.action_space.n):
                x, y = state
                if state not in self.T_states:
                    if action == UP:
                        x = x - 1
                    elif action == DOWN:
                        x = x + 1
                    elif action == RIGHT:
                        y = y + 1
                    elif action == LEFT:
                        y = y - 1
                new_state = (x, y)
                R[state][action] = self._get_reward(state,action,new_state)
        return R

    def reset(self, start_position=None):
        self.done = False
        if start_position:
            self.position = start_position
        else:
            if not self.start_position:
                idx = np.random.randint(len(self.possibleStates))
                self.position = self.possibleStates[idx]  # self.start_state_coord
                while self.position in self.T_states:
                    idx = np.random.randint(len(self.possibleStates))
                    self.position = self.possibleStates[idx]  # self.start_state_coord
            else:
                self.position = self.start_position
        self.state = self.position
        return self.state
        
    def render(self, fig=None, ax=None, env_map=True, mode='human', 
                P=None, V = None, Q = None, R = None, T = None, Ta = None,
                Ta_true = None, title=None, grid=False, cmap=None, show_color_bar=False):

        if not cmap:
            cmap = 'RdYlBu_r' # 'YlOrRd' if R else 'RdYlBu_r'

        img = self._gridmap_to_img()        
        if not fig:
            fig = plt.figure(1, figsize=(20, 20), dpi=60, facecolor='w', edgecolor='k')
            params = {'font.size': 40}
            plt.rcParams.update(params)
            plt.clf()
            plt.xticks([])
            plt.yticks([])
            plt.grid(grid)
            if title:
                plt.title(title, fontsize=20)    
            if mode == 'human':
                fig.canvas.draw()
        if not ax:
            ax = fig.gca()
        
        if env_map:
            ax = fig.gca()
            for position in self.T_states:
                y, x = position
                if position in self.goals:
                    label = "$G$"
                else:
                    label = "$L$"
                # ax.text(y, x, label, style='oblique', fontweight="bold")
                ax.text(x+0.4, y+0.6, label, style='oblique', fontweight="bold", size=fig.get_figheight()*3)
            # y, x = self.position
            # ax.add_patch(patches.Circle((x+0.5, y+0.5), radius=0.3, fc='black', transform=ax.transData, zorder=10))
        
        ax.imshow(img, origin="upper", extent=[0, self.n, self.m, 0])
        if Q: # For showing action_value
            qvalues = np.array(list(Q.values()))
            vmax = qvalues.max() # self.rmax #1.8#
            vmin = qvalues.min() # self.rmin*self.n*self.m #0#
            cmap_ = cm.get_cmap(cmap)
            norm = colors.Normalize(vmin,vmax)
            for state, q in Q.items():
                y, x = (state)
                for action in range(self.action_space.n):
                    v = (q[action]-vmin)/(vmax-vmin) # 
                    self._draw_reward(ax, x, y, action, v, cmap_)
            if show_color_bar:
                m = cm.ScalarMappable(norm=norm, cmap=cmap_)
                m.set_array(ax.get_images()[0])
                fig.colorbar(m, ax=ax)
                    
        if V: # For showing values
            v = np.zeros((self.m,self.n))+float("-inf")
            for state, val in V.items():
                y, x = (state)
                v[y,x] = val  
            c = ax.imshow(v, origin="upper", cmap=cmap, extent=[0, self.n, self.m, 0])
            if show_color_bar:
                fig.colorbar(c, ax=ax)
                
        if P:  # For drawing arrows of policy
            for state, action in P.items():
                y, x = (state)
                self._draw_action(ax, x, y, action)
        
        if R: # For showing rewards
            cmap_ = cm.get_cmap(cmap)
            norm = colors.Normalize(vmin=self.rmin, vmax=self.rmax)
            for state, reward in R.items():
                y, x = state
                for action in range(self.action_space.n):
                    r = (reward[action]-self.rmin)/(self.rmax-self.rmin)
                    self._draw_reward(ax, x, y, action, r, cmap_)
            # if show_color_bar:
            #     m = cm.ScalarMappable(norm=norm, cmap=cmap_)
            #     m.set_array(ax.get_images()[0])
            #     fig.colorbar(m, ax=ax)
        
        if T:  # For showing transition probabilities of single action
            vprob = np.zeros((self.m,self.n))+float("-inf")
            for state, prob in T.items():
                y, x = (state)
                vprob[y,x] = prob  
            c = plt.imshow(vprob, origin="upper", cmap=cmap, extent=[0, self.n, self.m, 0])
            if show_color_bar:
                fig.colorbar(c, ax=ax)
            
        if Ta:  # For showing transition probabilities of all actions
            for state, probs in Ta.items():
                y, x = (state)
                for action in range(len(probs)):
                    if probs[action]:
                        if Ta_true and not Ta_true[state][action]:
                            self._draw_action(ax, x, y, action, color='red')
                        else:
                            self._draw_action(ax, x, y, action)


        
        return fig
    
    def fig_image(fig):
        fig.tight_layout(pad=0)
        fig.gca().margins(0)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    def _map_init(self):
        self.grid = []
        lines = self.MAP.split('\n')

        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError(
                    "Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(int(col))
                if col == "1":
                    self.walls.append((i, j))
                # possible states
                else:
                    self.possibleStates.append((i, j))
            self.grid.append(rowArray)
        self.m = i + 1

        self._find_hallWays()

    def _find_hallWays(self):
        self.hallwayStates = []
        for x, y in self.possibleStates:
            if ((self.grid[x - 1][y] == 1) and (self.grid[x + 1][y] == 1)) or \
                    ((self.grid[x][y - 1] == 1) and (self.grid[x][y + 1] == 1)):
                self.hallwayStates.append((x, y))

    def _get_grid_value(self, state):
        return self.grid[state[0]][state[1]]

    # specific for self.MAP
    def _getRoomNumber(self, state=None):
        if state == None:
            state = self.state
        # if state isn't at hall way point
        xCount = self._greaterThanCounter(state, 0)
        yCount = self._greaterThanCounter(state, 1)
        room = 0
        if yCount >= 2:
            if xCount >= 2:
                room = 2
            else:
                room = 1
        else:
            if xCount >= 2:
                room = 3
            else:
                room = 0

        return room

    def _greaterThanCounter(self, state, index):
        count = 0
        for h in self.hallwayStates:
            if state[index] > h[index]:
                count = count + 1
        return count

    def _draw_action(self, ax, x, y, action, color='black'):
        if action == UP:
            x += 0.5
            y += 1
            dx = 0
            dy = -0.4
        if action == DOWN:
            x += 0.5
            dx = 0
            dy = 0.4
        if action == RIGHT:
            y += 0.5
            dx = 0.4
            dy = 0
        if action == LEFT:
            x += 1
            y += 0.5
            dx = -0.4
            dy = 0

        ax.add_patch(ax.arrow(x,  # x1
                      y,  # y1
                      dx,  # x2 - x1
                      dy,  # y2 - y1
                      facecolor=color,
                      edgecolor=color,
                      width=0.005,
                      head_width=0.4,
                      )
                    )

    def _draw_reward(self, ax, x, y, action, reward, cmap):
        x += 0.5
        y += 0.5
        triangle = np.zeros((3,2))
        triangle[0] = [x,y]
        
        if action == UP:
            triangle[1] = [x-0.5,y-0.5]
            triangle[2] = [x+0.5,y-0.5]
        if action == DOWN:
            triangle[1] = [x-0.5,y+0.5]
            triangle[2] = [x+0.5,y+0.5]
        if action == RIGHT:
            triangle[1] = [x+0.5,y-0.5]
            triangle[2] = [x+0.5,y+0.5]
        if action == LEFT:
            triangle[1] = [x-0.5,y-0.5]
            triangle[2] = [x-0.5,y+0.5]

        ax.add_patch(plt.Polygon(triangle, color=cmap(reward)))


    def _gridmap_to_img(self):
        row_size = len(self.grid)
        col_size = len(self.grid[0])

        obs_shape = [row_size, col_size, 3]

        img = np.zeros(obs_shape)

        gs0 = int(img.shape[0] / row_size)
        gs1 = int(img.shape[1] / col_size)
        for i in range(row_size):
            for j in range(col_size):
                for k in range(3):
                    if [(i, j),(i, j)] in self.goals:
                        this_value = COLOURS[3][k]
                    else:
                        colour_number = int(self.grid[i][j])
                        this_value = COLOURS[colour_number][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                       * gs1, k] = this_value
        return img