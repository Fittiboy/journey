import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import imageio
import os

from IPython.display import Image


class WindyGridWorld:
    def __init__(self, name=None):
        self.name = name
        self.height = 7
        self.width = 10
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.num_states = self.grid.size
        self.num_actions = 8
        self.start = (3, 0)
        self.goal = (3, 7)
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=int)
        self.reset()

    def reset(self):
        self.state = self.start
        return np.ravel_multi_index(self.state, self.grid.shape)

    def step(self, action_index):
        terminal = False
        dr, dc = self.index_to_action(action_index)
        r, c = self.state
        w = self.wind[c]
        r = self.clamp_r(r + dr - w)
        c = self.clamp_c(c + dc)

        self.state = (r, c)
        if self.state == self.goal:
            self.reset()
            terminal=True

        return np.ravel_multi_index(self.state, self.grid.shape), -1, terminal

    def index_to_action(self, index):
        actions = np.array([
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ])
        return actions[index]

    def clamp(self, val, high):
        return max(0, min(high, val))

    def clamp_r(self, r):
        return self.clamp(r, self.height - 1)
        
    def clamp_c(self, c):
        return self.clamp(c, self.width - 1)

    def full_state(self, pos=None):
        if pos is None:
            pos = self.state
        grid = self.grid.copy()
        grid[self.start] = 1
        grid[self.goal] = 1
        grid[pos] = 2
        return grid

    def render_state(self):
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        sns.heatmap(self.full_state(), ax=ax, cbar=False)
        ax.axis('off')
        plt.show()

    def render_episode(self, episode):
        frames = []
        for state in episode:
            fig, ax = plt.subplots(figsize=(self.width, self.height))
            pos = np.unravel_index(state, self.grid.shape)
            sns.heatmap(self.full_state(pos), ax=ax, cbar=False)
            ax.axis('off')
            plt.savefig('animations/temp_frame.png', bbox_inches='tight', pad_inches=0)
            frames.append(imageio.v2.imread('animations/temp_frame.png'))
            plt.close()

        if self.name is None:
            filename = 'episode_heatmap_wgw.gif'
        else:
            filename = f'episode_heatmap_{self.name}.gif'
        imageio.mimsave('animations/' + filename, frames, fps=2, loop=0)
        os.remove('animations/temp_frame.png')


class StochasticWindyGridWorld(WindyGridWorld):
    def __init__(self, name=None):
        super().__init__(name)

    def step(self, action_index):
        terminal = False
        dr, dc = self.index_to_action(action_index)
        r, c = self.state
        w = self.wind[c]
        if w != 0:
            rand = np.random.random()
            if rand < 1/3:
                w -= 1
            elif rand > 2/3:
                w += 1
        r = self.clamp_r(r + dr - w)
        c = self.clamp_c(c + dc)

        self.state = (r, c)
        if self.state == self.goal:
            self.reset()
            terminal=True

        return np.ravel_multi_index(self.state, self.grid.shape), -1, terminal