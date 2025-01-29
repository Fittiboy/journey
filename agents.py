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

    @abstractmethod
    def action_probs(self, agent: 'Agent', s:int) -> np.ndarray:
        """
        Return an array corresponding to the probabilities of each
        action being chosen.
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

    def action_probs(self, agent: 'Agent', s:int) -> np.ndarray:
        probs = np.zeros(len(agent.Q[s]))

        best_action = np.argmax(agent.Q[s])
        probs[best_action] = 1.0

        return probs


@dataclass
class EpsilonGreedy(ActionSelector):
    """The epsilon-greedy action selector."""
    epsilon: float
    rng: np.random.Generator = field(default_factory = lambda: np.random.default_rng())

    def __call__(self, agent: 'Agent', s: int) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, agent.num_actions)
        return np.argmax(agent.Q[s])

    def action_probs(self, agent: 'Agent', s:int) -> np.ndarray:
        num_actions = len(agent.Q[s])
        
        probs = np.ones(len(agent.Q[s])) * self.epsilon / num_actions
        
        best_action = np.argmax(agent.Q[s])
        probs[best_action] += 1 - self.epsilon

        return probs

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


@dataclass
class Sarsa(UpdateMethod):
    """Default Sarsa update method"""
    alpha: float
    gamma: float = field(default=1.0)

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        Q = agent.Q
        if T > t + 1:
            target = r + self.gamma * Q[s_, a_] - Q[s, a]
        else:
            target = r - Q[s, a]
        agent.Q[s, a] += self.alpha * target


@dataclass
class ExpectedSarsa(UpdateMethod):
    """Default Sarsa update method"""
    alpha: float
    gamma: float = field(default=1.0)

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        Q = agent.Q
        if T > t + 1:
            action_probs = agent.selector.action_probs(agent, s_)
            expected_value = np.dot(action_probs, Q[s_])
            
            target = r + self.gamma * expected_value - Q[s, a]
        else:
            target = r - Q[s, a]
        agent.Q[s, a] += self.alpha * target


@dataclass
class NStepSarsa(UpdateMethod):
    """n-step Sarsa update method"""
    n: int
    alpha: float
    gamma: float = field(default=1.0)
    states: list[int] = field(init=False)
    actions: list[int] = field(init=False)
    rewards: list[int] = field(init=False)

    def __post_init__(self):
        """Set up the n-step buffers"""
        self.states = [None] * (self.n + 1)
        self.actions = [None] * (self.n + 1)
        self.rewards = [None] * (self.n + 1)

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        Q = agent.Q
        n = self.n

        # Store newly encountered step in buffers
        self.states[t % (n+1)] = s
        self.actions[t % (n+1)] = a
        self.rewards[(t+1) % (n+1)] = r

        # The first timestep for which to update Q
        first_tau = t + 1 - n
        if first_tau >= 0:
            # If the episode has terminated, update for all remaining
            # time steps.
            tau_range_limit = T if T == t + 1 else first_tau + 1
            for tau in range(first_tau, tau_range_limit):
                # A running dicount factor
                discount = 1
                # Update target, into which the return is accumulated
                target = 0
                
                # The return is truncated if the episode has terminated
                k_range_limit = min(T, tau + n) + 1
                for k in range(tau + 1, k_range_limit):
                    target += discount * self.rewards[k % (n+1)]

                    # Update running discount factor
                    discount *= self.gamma

                # If the episode has not yet terminated, add the final term
                # to the udpate target
                index = tau % (n+1)
                update_state = self.states[index]
                update_action = self.actions[index]
                if T > t + 1:
                    target += discount * Q[s_, a_] - Q[update_state, update_action]
                else:
                    target += -Q[update_state, update_action]
                agent.Q[update_state, update_action] += self.alpha * target


@dataclass
class NStepExpectedSarsa(UpdateMethod):
    """n-step Sarsa update method"""
    n: int
    alpha: float
    gamma: float = field(default=1.0)
    states: list[int] = field(init=False)
    actions: list[int] = field(init=False)
    rewards: list[int] = field(init=False)

    def __post_init__(self):
        """Set up the n-step buffers"""
        self.states = [None] * (self.n + 1)
        self.actions = [None] * (self.n + 1)
        self.rewards = [None] * (self.n + 1)

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        Q = agent.Q
        n = self.n

        # Store newly encountered step in buffers
        self.states[t % (n+1)] = s
        self.actions[t % (n+1)] = a
        self.rewards[(t+1) % (n+1)] = r

        # The first timestep for which to update Q
        first_tau = t + 1 - n
        if first_tau >= 0:
            # If the episode has terminated, update for all remaining
            # time steps.
            tau_range_limit = T if T == t + 1 else first_tau + 1
            for tau in range(first_tau, tau_range_limit):
                # A running dicount factor
                discount = 1
                # Update target, into which the return is accumulated
                target = 0
                
                # The return is truncated if the episode has terminated
                k_range_limit = min(T, tau + n) + 1
                for k in range(tau + 1, k_range_limit):
                    target += discount * self.rewards[k % (n+1)]

                    # Update running discount factor
                    discount *= self.gamma

                # If the episode has not yet terminated, add the final term
                # to the udpate target
                index = tau % (n+1)
                update_state = self.states[index]
                update_action = self.actions[index]
                if T > t + 1:
                    action_probs = agent.selector.action_probs(agent, s_)
                    expected_value = np.dot(action_probs, Q[s_])
                    
                    target += discount * Q[s_, a_] - Q[update_state, update_action]
                else:
                    target += -Q[update_state, update_action]
                agent.Q[update_state, update_action] += self.alpha * target


@dataclass
class NStepTreeBackup(UpdateMethod):
    """n-step Tree Backup update method"""
    n: int
    alpha: float
    gamma: float = field(default=1.0)
    states: list[int] = field(init=False)
    actions: list[int] = field(init=False)
    rewards: list[int] = field(init=False)

    def __post_init__(self):
        """Set up the n-step buffers"""
        self.states = [None] * (self.n + 1)
        self.actions = [None] * (self.n + 1)
        self.rewards = [None] * (self.n + 1)

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        Q = agent.Q
        selector = agent.selector
        n = self.n

        # Store newly encountered step in buffers
        self.states[t % (n+1)] = s
        self.actions[t % (n+1)] = a
        self.rewards[(t+1) % (n+1)] = r
        self.states[(t+1) % (n+1)] = s_
        self.actions[(t+1) % (n+1)] = a_

        # The first timestep for which to update Q
        first_tau = t + 1 - n
        if first_tau >= 0:
            # If the episode has terminated, update for all remaining
            # time steps.
            tau_range_limit = T if T == t + 1 else first_tau + 1
            for tau in range(first_tau, tau_range_limit):
                # A running dicount factor
                discount = 1
                # Update target, into which the return is accumulated
                target = 0
                
                # The return is truncated if the episode has terminated
                k_range_limit = min(T, tau + n) + 1
                for k in range(tau + 1, k_range_limit):
                    index = k % (n+1)
                    reward = self.rewards[index]
                    next_state = self.states[index]
                    next_action = self.actions[index]
                    
                    target += discount * reward

                    # The final term includes only the terminal reward
                    if k == T:
                        break

                    # Get the probabilities for each action, and extract
                    # the probability for the action that was actually
                    # taken
                    probs = selector.action_probs(agent, next_state)
                    pi = probs[next_action]

                    # Calculate expectation, excluding the action that was
                    # actually taken (except for the final state, where the
                    # full expectation is used)
                    if k < k_range_limit - 1:
                        probs[next_action] = 0
                    expected_value = np.dot(probs, Q[next_state])
                    target += discount * self.gamma * expected_value

                    # Update running discount factor
                    discount *= self.gamma * pi

                # If the episode has not yet terminated, add the final term
                # to the udpate target
                index = tau % (n+1)
                update_state = self.states[index]
                update_action = self.actions[index]
                target += -Q[update_state, update_action]
                agent.Q[update_state, update_action] += self.alpha * target

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
        try:
            current = agent
            for part in self.param_path[:-1]:
                current = getattr(current, part)
            setattr(current, self.param_path[-1], value)
        except AttributeError:
            raise ValueError(f"Invalid parameter path: {self.param_path}")


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
    """
    Create a parameter schedule that reaches final at 0.5, then going back
    down to initial towards the end.
    """
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


class Dyna(UpdateMethod):
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
    Trains a given agent in a given environment,
    while keeping track of the progress along the way.
    """
    agent: Agent
    env: object
    episodes: int = field(default=0, init=False)

    def train(self, num_episodes, quiet=False, early_stop=-1):
        """
        Run the training loop for a given number of episodes, while
        allowing for early stopping and continuing, as well as muting
        the output, which is useful during parameter studies.
        """
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
                # point was reached. Ensure that training is only
                # continued if the early stopping point is removed.
                if early_stop != -1 and ep == early_stop:
                    raise TrainingInterrupt
                
                # Initialize the beginning of the episodes
                s = env.reset()
                a = agent.selector(agent, s)
                # The terminal time step, used for some update methods
                T = inf

                # Keep track of the episode's return
                ret = 0
                discount = 1
                
                for t in count():
                    
                    # Take a step in the environment and record the reward
                    s_, r, done, *info = env.step(a)
                    ret += discount * r
                    discount *= agent.learner.gamma
    
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

    def play_episode(self, max_steps=100):
        env = self.env
        agent = self.agent

        s = env.reset()
        a = agent.selector(agent, s)
        states = [s]
        actions = [a]
        rewards = [0]
        for _ in range(max_steps):
            s, r, done, *info = env.step(a)
            a = agent.selector(agent, s)
            rewards.append(r)
            states.append(s)
            actions.append(a)
            if done:
                break

        return states, actions, rewards