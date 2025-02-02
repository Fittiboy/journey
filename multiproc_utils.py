import os

import agents

import numpy as np

from itertools import product
from dataclasses import dataclass, field
from multiprocessing import Pool, Pipe, cpu_count
from copy import deepcopy

from agents import Agent, Trainer
from environments import Environment

from tqdm.auto import tqdm


@dataclass
class TrainPool:
    agents: list[Agent]
    env: Environment
    quiet: bool = field(default=False)

    def train_process(self, argtup):
        i, trainer = argtup
        trainer.train(num_episodes=self._num_episodes)
        if not self.quiet:
            print(f"Training complete for agent {i}!")
        filename = f"train_pool_agent_{i}.pkl"
        trainer.agent.save(filename)

    def train(self, num_episodes):
        """Train the agents for num_episodes."""
        self._num_episodes = num_episodes
        trainers = [Trainer(agent, self.env) for agent in self.agents]

        arglist = list(enumerate(trainers))

        # To prevent 100% CPU utilization. We're not running on a server!
        proc_bound = min(cpu_count() * 0.9, len(arglist))
        
        with Pool(int(proc_bound)) as p:
            p.map(self.train_process, arglist)

        for i in range(len(self.agents)):
            filename = f"train_pool_agent_{i}.pkl"
            self.agents[i] = Agent.load(filename)
            os.remove(filename)

        return self.agents

    def train_average(
        self,
        num_episodes,
        num_runs,
    ):
        """Train the agents for num_runs, num_episodes each."""
        untrained_agents = deepcopy(self.agents)
        
        num_agents = len(self.agents)
        eps = np.zeros((num_agents, num_runs, num_episodes))
        rets = np.zeros((num_agents, num_runs, num_episodes))

        for run in tqdm(range(num_runs), desc="Training Runs"):
            agents = self.train(num_episodes)
            for i, agent in enumerate(agents):
                eps[i, run, :] = agent.ep_lengths
                rets[i, run, :] = agent.ep_returns
            # We keep the training results from the last run
            # if run < num_runs - 1:
            if run < num_runs - 1:
                self.agents = deepcopy(untrained_agents)

        eps = np.average(eps, axis=1)
        rets = np.average(rets, axis=1)

        for i in range(num_agents):
            self.agents[i].ep_lengths = eps[i]
            self.agents[i].ep_returns = rets[i]

        return self.agents