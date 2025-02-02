import numpy as np
import pickle

from math import inf
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import count, product
from pathlib import Path
from datetime import datetime

from tqdm.auto import tqdm

################################
# Abstract Agent components
################################

class ActionSelector(ABC):
    """Agent component that selects an action for the current state s."""
    
    @abstractmethod
    def __call__(self, agent: 'Agent', s: int, t: int) -> int:
        """
        Return an action to take, conditional on the current
        observation.
        """
        pass

    @abstractmethod
    def action_probs(self, agent: 'Agent', s:int, t: int) -> np.ndarray:
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

@dataclass
class Greedy(ActionSelector):
    """The greedy action selector."""

    def __call__(self, agent: 'Agent', s: int, t: int) -> int:
        return np.argmax(agent.Q[s])

    def action_probs(self, agent: 'Agent', s:int, t: int) -> np.ndarray:
        probs = np.zeros(len(agent.Q[s]))

        best_action = np.argmax(agent.Q[s])
        probs[best_action] = 1.0

        return probs


@dataclass
class EpsilonGreedy(ActionSelector):
    """The epsilon-greedy action selector."""
    epsilon: float
    rng: np.random.Generator = field(default_factory = lambda: np.random.default_rng())

    def __call__(self, agent: 'Agent', s: int, t: int) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, agent.num_actions)
        return np.argmax(agent.Q[s])

    def action_probs(self, agent: 'Agent', s:int, t: int) -> np.ndarray:
        num_actions = len(agent.Q[s])
        
        probs = np.ones(len(agent.Q[s])) * self.epsilon / num_actions
        
        best_action = np.argmax(agent.Q[s])
        probs[best_action] += 1 - self.epsilon

        return probs


@dataclass
class EpsilonGreedyExpBonus(ActionSelector):
    epsilon: float
    kappa: float = field(default=0.01)
    last_used_a: np.ndarray = field(init=False, default=None)
    rng: np.random.Generator = field(default_factory = lambda: np.random.default_rng())

    def __call__(self, agent: 'Agent', s: int, t: int) -> int:
        if self.last_used_a is None or t == 0:
            # Initialize array that keeps track of timesteps during
            # which actions were last taken
            self.last_used_a = np.zeros_like(agent.Q)

        Q_s = agent.Q[s].copy()
        tau_s = t - self.last_used_a[s]
        bonus = self.kappa + np.sqrt(tau_s)
        Q_s += bonus
        
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, agent.num_actions)

        a = np.argmax(Q_s)
        self.last_used_a[s, a] = t
        return a

    def action_probs(self, agent: 'Agent', s:int, t: int) -> np.ndarray:
        if self.last_used_a is None:
            # Initialize array that keeps track of timesteps during
            # which actions were last taken
            self.last_used_a = np.zeros_like(agent.Q)

        Q_s = agent.Q[s].copy()
        tau_s = t - self.last_used_a[s]
        bonus = self.kappa + np.sqrt(tau_s)
        Q_s += bonus
        
        num_actions = len(Q_s)
        
        probs = np.ones(len(Q_s)) * self.epsilon / num_actions
        
        best_action = np.argmax(Q_s)
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
            action_probs = agent.selector.action_probs(agent, s_, t)
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
                    action_probs = agent.selector.action_probs(agent, s_, t)
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
                    probs = selector.action_probs(agent, next_state, t)
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
    initial: float = field(default=None)
    final: float = field(default=None)

    def __post_init__(self):
        assert len(self.param_path) > 0
        x = np.linspace(0, 1)
        assert (
            np.all(0 <= self.weight(x))
            and np.all(self.weight(x) <= 1.0)
        )

    def weight(self, progress):
        raise NotImplementedError

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        """Update the agent's parameter, according to the weight function"""
        progress = min(ep / num_eps, 1)
        weight = self.weight(progress)
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


@dataclass
class LinearSchedule(Schedule):
    """A linear parameter schedule"""
    def weight(self, progress):
        return progress


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class SigmoidSchedule(Schedule):
    """A sigmoid parameter schedule"""
    scale: int = 6
    
    def __post_init__(self):
        self.zero_shift = sigmoid(0)
        self.scale_factor = 1 / (sigmoid(SigmoidSchedule.scale) + self.zero_shift)
    
    def weight(self, progress):
        return (sigmoid(SigmoidSchedule.scale * progress) + self.zero_shift) * self.scale_factor


@dataclass
class UpDownSchedule(Schedule):
    """
    A parameter schedule that reaches final at 0.5, then going back
    down to initial towards the end.
    """
    def weight(self, progress: float) -> float:
        return (- (progress - 0.5) ** 2) * 4 + 1

################################
# Planners
################################

@dataclass
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


@dataclass
class Dyna(UpdateMethod):
    """Default Dyna planner"""
    plan_steps: int
    selector: ActionSelector
    learner: UpdateMethod
    plus: bool = field(default=False)
    kappa: float = field(default=0.01)
    model: dict = field(default_factory=dict)
    last_used_a: np.ndarray = field(init=False, default=None)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __call__(
        self,
        agent: 'Agent',
        s: int, a: int, r: float, s_: int, a_: int,
        t: int, T: int,
        ep: int, num_eps: int,
    ):
        if self.plus and self.last_used_a is None or self.plus and t == 0:
            self.last_used_a = np.zeros_like(agent.Q)
        done = t + 1 == T
        # Update the model naively by storing transition
        self.model[(s, a)] = (r, s_, done)

        # Record the timestep at which this action was last
        # performed
        if self.plus:
            self.last_used_a[s, a] = t

        # Perform actual planning
        for _ in range(self.plan_steps):
            # Sample random S, A pair from the model
            s, a = self.rng.choice(list(self.model))
            # Retreive reward last experienced from S, A
            r, s_, done = self.model[(s, a)]
            # If running Dyna+, add the exploration bonus
            if self.plus:
                # How long it has been since the action was last
                # chosen
                tau = t - self.last_used_a[s, a]
                # The Dyna+ exploration bonus
                if tau < 0:
                    print(tau)
                bonus = self.kappa * np.sqrt(tau)
            # Aside from Dyna-Q, this Dyna method also works with other update
            # methods, like Sarsa, in which case there is a need for providing
            # A_{t+1} as well
            a_ = self.selector(agent, s_, t)
            # Planning involves simply learning from a simulated transition as if
            # it was actually experienced again, so the learner is called with this
            # synthetic data
            self.learner(agent, s, a, r, s_, a_, 0, 1 if done else inf, None, None)

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

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath) -> 'Agent':
        with open(filepath, "rb") as f:
            return pickle.load(f)
            
    @classmethod
    def combinations(
        cls,
        num_states: int,
        num_actions: int,
        selectors: list[ActionSelector],
        learners: list[UpdateMethod],
        scheduless: list[list[Schedule]],
        planners: list[UpdateMethod],
    ):
        param_space = product(selectors, learners, scheduless, planners)
        
        agents = [
            Agent(
                num_states=num_states,
                num_actions=num_actions,
                selector=selector,
                learner=learner,
                schedules=schedules,
                planner=planner,
            ) for (selector, learner, schedules, planner) in param_space
        ]
    
        return agents

    ### Evaulation methods ###
    def smoothed_ep_lengths(self, trail_length: int) -> (list[int], list[int]):
        """
        Return xs and ys for plotting smoothed out episode lengths.

        Args:
            trail_length: Used to calculate the update weight, 1/trail_length
        """
        xs = np.arange(len(self.ep_lengths))
        ys = np.zeros(len(self.ep_lengths))
        ys[0] = self.ep_lengths[0]
        
        weight = 1 / trail_length
        
        for i in range(1, len(ys)):
            ys[i] = weight * self.ep_lengths[i] + (1 - weight) * ys[i-1]

        return xs, ys
    
    def smoothed_ep_returns(self, trail_length: int) -> (list[int], list[int]):
        """
        Return xs and ys for plotting smoothed out episode returns.

        Args:
            trail_length: Used to calculate the update weight, 1/trail_length
        """
        
        xs = np.arange(len(self.ep_returns))
        ys = np.zeros(len(self.ep_returns))
        ys[0] = self.ep_returns[0]
        
        weight = 1.0 / trail_length
        assert 0 < weight <= 1
        
        for i in range(1, len(ys)):
            ys[i] = weight * self.ep_returns[i] + (1 - weight) * ys[i-1]

        return xs, ys
    
    def cumulative_eps(self) -> (list[int], list[int]):
        """Return xs and ys for plotting cumulative episodes over timesteps."""
        xs = np.cumsum(self.ep_lengths)
        ys = list(range(len(self.ep_lengths)))

        return xs, ys

    def cumulative_returns(self) -> (list[int], list[int]):
        """Return xs and ys for plotting cumulative rewards over episodes."""
        xs = list(range(len(self.ep_lengths)))
        ys = np.cumsum(self.ep_returns)

        return xs, ys
            
    def play_episode(self, env, max_steps=100):
        s = env.reset()
        a = self.selector(self, s, 0)
        states = [s]
        actions = [a]
        rewards = [0]
        for t in range(max_steps):
            s, r, done, *info = env.step(a)
            a = self.selector(self, s, t)
            rewards.append(r)
            states.append(s)
            actions.append(a)
            if done:
                break
    
        return states, actions, rewards
        
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
                a = agent.selector(agent, s, 0)
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
                        a_ = agent.selector(agent, s_, t)
    
                    # Update the agent's value estimates through direct LR
                    agent.learner(agent, s, a, r, s_, a_, t, T, ep, num_episodes)

                    # Apply the agent's parameter update schedules
                    for schedule in agent.schedules:
                        schedule(agent, s, a, r, s_, a_, t, T, ep, num_episodes)

                    # Execute the agent's planning strategy
                    if agent.planner:
                        agent.planner(agent, s, a, r, s_, a_, t, T, ep, num_episodes)
    
                    if done:
                        agent.ep_lengths.append(T)
                        agent.ep_returns.append(ret)
                        break
                        
                    # Update values for the next time step
                    s, a = s_, a_
                
        except (KeyboardInterrupt, TrainingInterrupt):
            print(f"Training paused after {self.episodes} episodes.")