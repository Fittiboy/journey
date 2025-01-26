import numpy as np

from agent_components import LinearSchedule, SigmoidSchedule
from itertools import count
from math import inf
from tqdm.notebook import tqdm


class Agent:
    def __init__(self,
                 epsilon,
                 alpha,
                 discount,
                 num_states,
                 num_actions,
                 eps_schedule=None,
                 alp_schedule=None,
                ):
        self.orig_epsilon = epsilon
        self.orig_alpha = alpha
        if eps_schedule:
            self.eps_schedule = eps_schedule
        else:
            self.eps_schedule = LinearSchedule(epsilon, epsilon)
        if alp_schedule:
            self.alp_schedule = alp_schedule
        else:
            self.alp_schedule = LinearSchedule(alpha, alpha)
        self.update_params(0, 1)
        self.discount = discount
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.ep_lengths = []
        self.ep = 0

    def select_action(self, state, greedy=False):
        raise NotImplementedError

    def update(self, t, T, state, action, reward, next_state, next_action):
        raise NotImplementedError

    def train_episode(self, environment):
        state = environment.reset()
        action = self.select_action(state)
        T = inf
        for t in count():
            next_state, reward, done = environment.step(action)
            if done:
                T = t + 1
            next_action = self.select_action(next_state)
            self.update(t, T, state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            if done:
                self.ep_lengths.append(t)
                break

    def update_params(self, ep, num_episodes):
        self.alpha = self.alp_schedule.value(ep, num_episodes)
        self.epsilon = self.eps_schedule.value(ep, num_episodes)

    def post_ep_adjustments(self, num_episodes):
        self.update_params(self.ep, num_episodes)
            
    def train(self, environment, num_episodes, quiet=False):
        try:
            for ep in tqdm(range(self.ep, num_episodes),
                           disable=quiet,
                           desc="Episodes",
                           initial=self.ep,
                           total=num_episodes):
                self.ep = ep
                self.train_episode(environment)
                self.post_ep_adjustments(num_episodes)
        except KeyboardInterrupt:
            print(f"Training paused after {self.ep} episodes.")

    def play_episode(self, environment, greedy=False, max_steps=-1):
        state = environment.reset()
        action = self.select_action(state, greedy=greedy)
        states = [state]
        actions = [action]
        rewards = [0]
        for _ in range(max_steps):
            state, reward, done = environment.step(action)
            action = self.select_action(state, greedy=greedy)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            if done:
                break

        return states, actions, rewards


class StateValueAgent(Agent):
    def __init__(self,
                 epsilon,
                 alpha,
                 discount,
                 num_states,
                 num_actions,
                 eps_schedule=None,
                 alp_schedule=None,
                ):
        super().__init__(epsilon,
                         alpha,
                         discount,
                         num_states,
                         num_actions,
                         eps_schedule,
                         alp_schedule,
                        )
        self.V = np.zeros((num_states))


class QValueAgent(Agent):
    def __init__(self,
                 epsilon,
                 alpha,
                 discount,
                 num_states,
                 num_actions,
                 eps_schedule=None,
                 alp_schedule=None,
                ):
        super().__init__(epsilon,
                         alpha,
                         discount,
                         num_states,
                         num_actions,
                         eps_schedule,
                         alp_schedule,
                        )
        self.Q = np.zeros((num_states, num_actions))

    def select_action(self, state, greedy=False):
        action = np.argmax(self.Q[state])
        if not greedy and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        return action


class NStepStorage:
    def __init__(self, n, dtype=float):
        self.data = np.zeros(n + 1, dtype=dtype)
        self.n = n
        
    def __getitem__(self, key):
        return self.data[key % (self.n + 1)]

    def __setitem__(self, key, value):
        self.data[key % (self.n + 1)] = value


class NStepSarsa(QValueAgent):
    def __init__(self, n,
                 epsilon,
                 alpha,
                 discount,
                 num_states,
                 num_actions,
                 eps_schedule=None,
                 alp_schedule=None,
                ):
        super().__init__(epsilon,
                         alpha,
                         discount,
                         num_states,
                         num_actions,
                         eps_schedule,
                         alp_schedule,
                        )
        self.n = n
        self.states = NStepStorage(n, dtype=int)
        self.actions = NStepStorage(n, dtype=int)
        self.rewards = NStepStorage(n, dtype=int)

    def update(self, t, T, state, action, reward, next_state, next_action):
        self.states[t] = state
        self.actions[t] = action
        self.rewards[t+1] = reward

        tau = t + 1 - self.n
        if tau >= 0:
            if t + 1 < T:
                target = 0
                for k in range(tau, tau + self.n):
                    target += self.discount ** (k - tau) * self.rewards[k+1]
                target += self.discount ** self.n * self.Q[next_state, next_action]
                self.Q[self.states[tau], self.actions[tau]] += self.alpha * (
                    target - self.Q[self.states[tau], self.actions[tau]]
                )
            else:
                for tau_ in range(tau, T):
                    target = 0
                    for k in range(tau_, T):
                        target += self.discount ** (k - tau_) * self.rewards[k+1]
                    self.Q[self.states[tau_], self.actions[tau_]] += self.alpha * (
                        target - self.Q[self.states[tau_], self.actions[tau_]]
                    )