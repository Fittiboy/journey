import numpy as np

from math import inf
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import count

from tqdm.auto import tqdm

################################
# Abstract Agent components
################################

class ActionSelector(ABC):
    """Agent component that selects an action for the current state s."""
    
    @abstractmethod
    def __call__(self, agent: 'Agent', s: int) -> int:
        """
        Return an action to take, conditional on the current
        observation.
        """
        pass


class UpdateMethod(ABC):
    """
    Component to update the agent's internal state.

    Examples include learning value estimates, updating the
    value estimator's parameters, or planning with a model.
    """

    @abstractmethod
    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        """Update some of the agent's internal state."""
        pass

################################
# Action selectors
################################

class Greedy(ActionSelector):
    """The greedy action selector."""

    def __call__(self, agent: 'Agent', s: int) -> int:
        return np.argmax(agent.Q[s])


@dataclass
class EpsilonGreedy(ActionSelector):
    """The epsilon-greedy action selector."""
    eps: float
    rng: np.random.Generator = field(default_factory = lambda: np.random.default_rng())

    def __call__(self, agent: 'Agent', s: int) -> int:
        if self.rng.random() < self.eps:
            return self.rng.integers(0, agent.num_actions)
        return np.argmax(agent.Q[s])

################################
# Value update methods
################################

@dataclass
class QLearning(UpdateMethod):
    """Default Q-learning update method"""
    alpha: float
    gamma: float = field(default=1.0)

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        if T > t + 1:
            target = r + self.gamma * np.max(agent.Q[s_]) - agent.Q[s, a]
        else:
            target = r - agent.Q[s, a]
        agent.Q[s, a] += self.alpha * target

################################
# Parameter schedules
################################

@dataclass
class Schedule(UpdateMethod):
    """Schedule for updating an agent's parameter"""
    param_path: list[str]
    # The weight function should start at f(0)=0 and stay within [0, 1]
    weight_fn: Callable[[float], float]
    initial: float = field(default=None)
    final: float = field(default=None)

    def __post_init__(self):
        assert len(self.param_path) > 0
        x = np.linspace(0, 1)
        assert (
            np.all(0 <= self.weight_fn(x))
            and np.all(self.weight_fn(x) <= 1.0)
        )

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        """Update the agent's parameter, according to the weight function"""
        progress = min(ep / num_eps, 1)
        weight = self.weight_fn(progress)
        value = (1 - weight) * self.initial + weight * self.final

        # Get the parameter from the agent.
        # The parameter may be nested, for example:
        # ["selector", "epsilon"], which corresponds to
        # agent.selector.epsilon
        current = agent
        for part in self.param_path[:-1]:
            current = getattr(current, part)
        setattr(current, self.param_path[-1], value)


def linear_schedule(param_path: list[str], initial: float, final: float):
    """Create a linear parameter schedule"""
    def linear_weight(progress):
        return progress
    
    return Schedule(
        param_path=param_path,
        weight_fn=linear_weight,
        initial=initial,
        final=final,
    )


def sigmoid_schedule(param_path: list[str], initial: float, final: float, scale=6):
    """Create a sigmoid parameter schedule"""
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    zero_shift = sigmoid(0)
    scale_factor = 1 / (sigmoid(scale) + zero_shift)
    
    def sigmoid_weight(progress):
        return (sigmoid(scale * progress) + zero_shift) * scale_factor
    
    return Schedule(
        param_path=param_path,
        weight_fn=sigmoid_weight,
        initial=initial,
        final=final,
    )


def updown_schedule(param_path: list[str], initial: float, final: float):
    """Create a linear parameter schedule"""
    def updown_weight(progress: float) -> float:
        return (- (progress - 0.5) ** 2) * 4 + 1
    
    return Schedule(
        param_path=param_path,
        weight_fn=updown_weight,
        initial=initial,
        final=final,
    )

################################
# Planners
################################

class NoPlanner(UpdateMethod):
    """Does nothing"""
    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        """Update some of the agent's internal state."""
        pass

################################
# Agent
################################

@dataclass
class Agent:
    """
    An RL agent that is to be trained in a specific environment,
    holding value estimates, an action selection strategy, update
    rules, and, optionally a planning strategy and hyperparameter
    update schedules.
    """
    num_states: int
    num_actions: int
    Q: np.ndarray = field(init=False)
    selector: ActionSelector
    learner: UpdateMethod
    schedules: list[UpdateMethod] = field(default_factory = list)
    planner: UpdateMethod = field(default_factory = NoPlanner)
    ep_lengths: list[int] = field(default_factory = list, init=False)
    ep_returns: list[int] = field(default_factory = list, init=False)

    def __post_init__(self):
        self.Q = np.zeros((self.num_states, self.num_actions))

################################
# Training loop
################################

class TrainingInterrupt(Exception):
    pass


@dataclass
class Trainer:
    """
    A class that trains a given agent in a given environment,
    while keeping track of the progress along the way.
    """
    agent: Agent
    env: object
    episodes: int = field(default=0, init=False)

    def train(self, num_episodes, quiet=False, early_stop=-1):
        # Iterator that keeps track of progress even when training
        # loop is interrupted.
        train_iter = tqdm(
            range(self.episodes, num_episodes),
            desc="Episodes",
            initial=self.episodes,
            total=num_episodes,
            disable=quiet,
        )
        env = self.env
        agent = self.agent
        try:
            for ep in train_iter:
                self.episodes = ep
                # Interrupt the training process if our early stopping
                # point was reached
                if ep == early_stop:
                    raise TrainingInterrupt
                
                # Initialize the beginning of the episodes
                s = env.reset()
                a = agent.selector(agent, s)
                # The terminal time step, used for some update methods
                T = inf

                # Keep track of the episode's return
                ret = 0
                
                for t in count():
                    
                    # Take a step in the environment and record the reward
                    s_, r, done = env.step(a)
                    ret += r
    
                    # Select a next action to take, unless the episode
                    # has finished, in which case set the terminal time
                    # step to its true value.
                    if done:
                        T = t + 1
                    else:
                        a_ = agent.selector(agent, s_)
    
                    # Update the agent's value estimates through direct LR
                    agent.learner(agent, s, a, r, s_, a_, t, T, ep, num_episodes)

                    # Apply the agent's parameter update schedules
                    for schedule in agent.schedules:
                        schedule(agent, s, a, r, s_, a_, t, T, ep, num_episodes)

                    # Execute the agent's planning strategy
                    agent.planner(agent, s, a, r, s_, a_, t, T, ep, num_episodes)
    
                    if done:
                        agent.ep_lengths.append(T)
                        agent.ep_returns.append(ret)
                        break
                        
                    # Update values for the next time step
                    s, a = s_, a_
                
        except (KeyboardInterrupt, TrainingInterrupt):
            print(f"Training paused after {self.episodes} episodes.")