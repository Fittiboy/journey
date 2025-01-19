import numpy as np
from random import choices


class KArmedBandit:
    """A multi-armed bandit whose true rewards can drift over time."""
    def __init__(self, k, drift=False, seed=None):
        """
        Args:
            k (int): Number of arms.
            drift (bool): Whether the true rewards should drift.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32)
        self.random = np.random.RandomState(seed)
        self.k = k
        self.R = np.ones(k) if drift else self.random.normal(size=self.k)
        self.drift=drift

    def step(self, action):
        """
        Draw a reward from a (normal) distribution centered around the
        true value of the selected arm (self.R[action]), and then shift
        all arm values by a small random walk, if self.drift=True.
        
        Args:
            action (int): Arm index to pull.
        
        Returns:
            float: Observed reward.
        """
        R = self.random.normal(self.R[action])
        if self.drift:
            self.R += self.random.normal(scale=0.01, size=self.k)
        return R


class BanditAgent:
    """An agent estimating the rewards of a KArmedBandit."""
    def __init__(self, k, step_size=None):
        """
        Args:
            k (int): Number of actions (bandit's arms).
            step_size (float): (Optional) To use a constant step-size
                instead of the true sample average.
        """
        self.Q = np.zeros(k, dtype=float)
        self.N = np.zeros(k, dtype=int)
        self.step_size = step_size
        self.rewards = []
        self.optimals = []

    def policy(self):
        """
        Returns:
            int: Index of the selected action
        """
        return np.argmax(self.Q)

    def update_values(self, action, reward):
        """
        Update the value estimates

        Args:
            action (int): Index of the chosen action.
            reward (float): Reward received for taking action.
        """
        error = reward - self.Q[action]
        if not self.step_size:
            self.N[action] += 1
            self.Q[action] += (1 / self.N[action]) * error
        else:
            self.Q[action] += self.step_size * error

    def learn(self, bandit, steps):
        """
        Learn the bandit's action values.

        Args:
            bandit (KArmedBandit): The bandit of which the action values
                are to be estimated.
            steps (int): Number of actions to take to learn.
        """
        for _ in range(steps):
            action = self.policy()
            reward = bandit.step(action)
            self.update_values(action, reward)
            self.rewards.append(reward)
            self.optimals.append(action == np.argmax(bandit.R))

    def mse(self, bandit):
        """
        Returns:
            float: Mean Squared Error of the action value estimates.
        """
        return np.mean((bandit.R - self.Q) ** 2)


class EpsilonGreedyBanditAgent(BanditAgent):
    """
    An epsilon-greedy agent estimating the rewards of a KArmedBandit,
    using either a true sample average, or updating with constant
    step-size.
    """
    def __init__(self, k, epsilon, step_size=None):
        """
        Args:
            epsilon (float): Value for epsilon-greedy action selection.
        """
        super().__init__(k, step_size)
        self.epsilon = epsilon

    def policy(self):
        """
        Perform epsilon-greedy action selection

        Returns:
            int: Arm index to pull.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(self.Q))
        else:
            return np.argmax(self.Q)


class OptimisticGreedyBanditAgent(BanditAgent):
    """
    A greedy agent estimating the rewards of a KArmedBandit, using either
    a true sample average, or updating with constant step-size. Initial
    action value estimates are optimistic.
    """
    def __init__(self, k, initial_values=5.0, step_size=None):
        """
        Args:
            initial_values (float): Initial value estimate for the
                agent's actions.
        """
        super().__init__(k, step_size)
        self.Q[:] = initial_values


class UCBBanditAgent(BanditAgent):
    """
    A UCB agent estimating the rewards of a KArmedBandit, using either
    a true sample average, or updating with constant step-size.
    """
    def __init__(self, k, c, step_size=None):
        """
        Args:
            c (float): UCB's confidence parameter.
        """
        super().__init__(k, step_size)
        self.c = c
        self.t = 1
    
    def policy(self):
        new = np.arange(len(self.N))[self.N == 0]
        if len(new) > 0:
            return new[0]
        return np.argmax(self.Q + self.c * np.sqrt(np.log(self.t) / self.N))

    def update_values(self, action, reward):
        self.t += 1
        if self.step_size:
            self.N[action] += 1
        super().update_values(action, reward)


class GradientBanditAgent(BanditAgent):
    """
    A gradient bandit agent estimating the rewards of a KArmedBandit.
    """
    def __init__(self, k, r_step_size, step_size):
        """
        Args:
            r_step_size (float): Step size for tracking the average
                reward.
        """
        super().__init__(k, step_size)
        self.r_step_size = r_step_size
        self.average_reward = None

    def pi(self):
        prefs = self.Q - np.max(self.Q) # For numerical stability
        return np.exp(prefs) / np.sum(np.exp(prefs))

    def policy(self):
        softmax = self.pi()
        return choices(list(range(len(softmax))), weights=softmax)[0]

    def update_values(self, action, reward):
        if self.average_reward is None:
            self.average_reward = reward
        else:
            self.average_reward += self.r_step_size * (reward - self.average_reward)
        softmax = self.pi()
        indices = list(range(len(softmax)))
        indices.remove(action)
        step = self.step_size * (reward - self.average_reward)
        self.Q[indices] -= step * softmax[indices]
        self.Q[action] += step * (1 - softmax[action])