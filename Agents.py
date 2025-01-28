import numpy as np

from AgentComponents import LinearSchedule, SigmoidSchedule
from itertools import count
from math import inf
from tqdm.notebook import tqdm


class Agent:
    def __init__(
        self,
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
            
    def train(self, environment, num_episodes, quiet=False, early_stop=-1):
        try:
            for ep in tqdm(range(self.ep, num_episodes),
                           disable=quiet,
                           desc="Episodes",
                           initial=self.ep,
                           total=num_episodes):
                self.ep = ep
                if ep == early_stop:
                    raise KeyboardInterrupt
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
    def __init__(
        self,
        epsilon,
        alpha,
        discount,
        num_states,
        num_actions,
        eps_schedule=None,
        alp_schedule=None,
    ):
        super().__init__(
            epsilon,
            alpha,
            discount,
            num_states,
            num_actions,
            eps_schedule,
            alp_schedule,
        )
        self.V = np.zeros((num_states))


class QValueAgent(Agent):
    def __init__(
        self,
        epsilon,
        alpha,
        discount,
        num_states,
        num_actions,
        eps_schedule=None,
        alp_schedule=None,
    ):
        super().__init__(
            epsilon,
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


class ExpectedSarsa(QValueAgent):
    def __init__(
        self,
        epsilon,
        alpha,
        discount,
        num_states,
        num_actions,
        eps_schedule=None,
        alp_schedule=None,
    ):
        super().__init__(
            epsilon,
            alpha,
            discount,
            num_states,
            num_actions,
            eps_schedule,
            alp_schedule,
        )

    def update(self, t, T, state, action, reward, next_state, next_action):
        exp_ret = sum(self.Q[next_state]) * self.epsilon / self.num_actions
        exp_ret += (1 - self.epsilon) * np.max(self.Q[next_state])
        target = reward + self.discount * exp_ret
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


class DynaExpectedSarsa(ExpectedSarsa):
    def __init__(
        self,
        plan_steps,
        epsilon,
        alpha,
        discount,
        num_states,
        num_actions,
        eps_schedule=None,
        alp_schedule=None,
    ):
        super().__init__(
            epsilon,
            alpha,
            discount,
            num_states,
            num_actions,
            eps_schedule,
            alp_schedule,
        )
        self.model = dict()
        self.plan_steps = plan_steps
        self.rng = np.random.default_rng()

    def update_rule(self, t, T, state, action, reward, next_state, next_action):
        super().update(t, T, state, action, reward, next_state, next_action)

    def update(self, t, T, state, action, reward, next_state, next_action):
        # Update the environment model
        self.model[(state, action)] = (reward, next_state, t)
        # Perform the direct RL step
        self.update_rule(t, T, state, action, reward, next_state, next_action)
        # Perform the planning step, plan_step times
        for _ in range(self.plan_steps):
            state, action = self.rng.choice(list(self.model))
            reward, next_state, timestamp = self.model[(state, action)]
            next_action = self.select_action(state)
            self.update_rule(t, T, state, action, reward, next_state, next_action)


class DynaQ(DynaExpectedSarsa):
    def __init__(
        self,
        plan_steps,
        epsilon,
        alpha,
        discount,
        num_states,
        num_actions,
        eps_schedule=None,
        alp_schedule=None,
    ):
        super().__init__(
            plan_steps,
            epsilon,
            alpha,
            discount,
            num_states,
            num_actions,
            eps_schedule,
            alp_schedule,
        )
        self.model = dict()
        self.plan_steps = plan_steps
        self.rng = np.random.default_rng()

    def update_rule(self, t, T, state, action, reward, next_state, next_action):
        target = reward + self.discount * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    
class NStepStorage:
    def __init__(self, n, dtype=float):
        self.data = np.zeros(n + 1, dtype=dtype)
        self.n = n
        
    def __getitem__(self, key):
        return self.data[key % (self.n + 1)]

    def __setitem__(self, key, value):
        self.data[key % (self.n + 1)] = value


class NStepSarsa(QValueAgent):
    def __init__(
        self,
        n,
        epsilon,
        alpha,
        discount,
        num_states,
        num_actions,
        eps_schedule=None,
        alp_schedule=None
    ):
        super().__init__(
            epsilon,
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

    def update_target(self, tau, limit):
        target = 0
        for k in range(tau, limit):
            target += self.discount ** (k - tau) * self.rewards[k+1]
        return target

    def target_final_term(self, next_state, next_action):
        return self.discount ** self.n * self.Q[next_state, next_action]

    def update_rule(self, tau, target):
        self.Q[self.states[tau], self.actions[tau]] += self.alpha * (
            target - self.Q[self.states[tau], self.actions[tau]]
        )

    def update(self, t, T, state, action, reward, next_state, next_action):
        self.states[t] = state
        self.actions[t] = action
        self.rewards[t+1] = reward

        tau = t + 1 - self.n
        if tau >= 0:
            if t + 1 < T:
                target = self.update_target(tau, t + 1)
                target += self.target_final_term(next_state, next_action)
                self.update_rule(tau, target)
            else:
                for tau_ in range(tau, T):
                    target = self.update_target(tau_, T)
                    self.update_rule(tau_, target)


class NStepExpectedSarsa(NStepSarsa):
    def __init__(
        self,
        n,
        epsilon,
        alpha,
        discount,
        num_states,
        num_actions,
        eps_schedule=None,
        alp_schedule=None,
    ):
        super().__init__(
            n,
            epsilon,
            alpha,
            discount,
            num_states,
            num_actions,
            eps_schedule,
            alp_schedule,
        )

    def target_final_term(self, next_state, next_action):
        val = np.sum(self.Q[next_state]) * self.epsilon / self.num_actions
        val += (1 - self.epsilon) * np.max(self.Q[next_state])
        return self.discount ** self.n * val


class NStepTreeBackup(NStepExpectedSarsa):
    def __init__(
        self,
        n,
        epsilon,
        alpha,
        discount,
        num_states,
        num_actions,
        eps_schedule=None,
        alp_schedule=None,
    ):
        super().__init__(
            n,
            epsilon,
            alpha,
            discount,
            num_states,
            num_actions,
            eps_schedule,
            alp_schedule,
        )

    def update_target(self, tau, limit):
        target = 0
        self._weight = 1
        for k in range(tau, limit - 1):
            # Received reward and expectation over unchosen actions
            exp_ret = np.sum(self.Q[self.states[k+1]]) * self.epsilon / self.num_actions
            exp_ret += (1 - self.epsilon) * np.max(self.Q[self.states[k+1]])
            
            # Weight for the chosen action
            pi_taken = self.epsilon / self.num_actions
            if np.argmax(self.Q[self.states[k+1]]) == self.actions[k+1]:
                pi_taken += 1 - self.epsilon
            # Remove chosen action's value from the expectation term
            exp_ret -= pi_taken * self.Q[self.states[k+1], self.actions[k+1]]
            
            target += self._weight * (self.rewards[k+1] + self.discount * exp_ret)
            
            # Updated weight for the next term
            self._weight *= self.discount * pi_taken
        target += self._weight * self.rewards[limit]
        return target

    def target_final_term(self, next_state, next_action):
        val = np.sum(self.Q[next_state]) * self.epsilon / self.num_actions
        val += (1 - self.epsilon) * np.max(self.Q[next_state])
        return self._weight * self.discount * val