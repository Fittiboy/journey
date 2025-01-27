import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, Pipe


class ParameterStudy:
    def __init__(self, agent_type, environment, param_dict, pool_size=20):
        self.agent_type = agent_type
        self.environment = environment
        self.param_dict = param_dict
        self.num_params = len(param_dict)
        self.pool_size = pool_size

    def sweep(self, num_runs, num_episodes):
        self.results = np.zeros(tuple(len(v) for v in self.param_dict.values()))
        self.num_episodes = num_episodes
        
        procs = []
        pipes = []
        argsets = []
        proc_num = 0
        for i in range(self.results.size):
            proc_num += 1
            pc, cc = Pipe()
            mi = np.unravel_index(i, self.results.shape)
            args = {k: self.param_dict[k][j] for k, j in zip(self.param_dict, mi)}
            argsets.append((cc, proc_num, self.agent_type, self.environment, mi, args, num_runs, num_episodes))
            pipes.append(pc)

        with Pool(self.pool_size) as p:
            p.map(self.evaluate, argsets)
        
        for pipe in pipes:
            mi, result = pipe.recv()
            self.results[mi] = result
        print("Sweep complete, results saved!")

    def evaluate(self, argtup):
        conn, proc_num, agent_type, environment, mi, args, num_runs, num_episodes = argtup
        total_steps = 0
        for run in range(num_runs):
            agent = agent_type(num_states=environment.num_states, num_actions=environment.num_actions, **args)
            agent.train(environment=environment, num_episodes=num_episodes, quiet=True)
            total_steps += agent.ep_lengths[-1]
        print(f"Process number {proc_num} finished!")
        conn.send((mi, total_steps / num_runs))
        conn.close()

    def plot_results(self, render_first=None, filename=None):
        if render_first is None:
            render_first = self.num_params
        fig, axes = plt.subplots(render_first, 1, figsize=(7, render_first * 4))
        fig.suptitle(f"Number of episodes: {self.num_episodes}")
        for i, param in enumerate(list(self.param_dict.keys())[:render_first]):
            averages = self.results
            for _ in range(i):
                averages = np.average(averages, axis=0)
            while len(averages.shape) > 1:
                averages = np.average(averages, axis=-1)
            axes[i].plot(self.param_dict[param], averages)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel("Number of steps in final episode")
        if filename is not None:
            fig.savefig(filename)
        plt.show()