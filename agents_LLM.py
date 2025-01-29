import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict
from itertools import count
from math import inf
from tqdm.notebook import tqdm

########################################
# Schedules for parameter updates
########################################

@dataclass
class Schedule:
    initial: float
    final: float
    weight: Callable[[int, int], float]


def linear_schedule(initial: float, final: float) -> Schedule:
    """Create a linear parameter schedule"""
    def linear_weight(progress: float) -> float:
        return progress
    return Schedule(initial, final, linear_weight)


def sigmoid_schedule(initial: float, final: float) -> Schedule:
    """Create a sigmoid parameter schedule"""
    def sigmoid_weight(progress: float) -> float:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
            
        return (sigmoid(6 * progress) - 1) / (np.exp(6) - 1)
    return Schedule(initial, final, sigmoid_weight)


def up_down_schedule(initial: float, final: float) -> Schedule:
    """
    Create a schedule that rises to final,
    then drops back to initial.
    """
    def up_down_weight(progress: float) -> float:
        return (- (progress - 0.5) ** 2) * 4 + 1
    return Schedule(initial, final, up_down_weight)

########################################
# AgentState: Holds hyperparameters, Q/V tables, schedules, etc.
########################################

class AgentState:
    def __init__(
        self,
        num_states,
        num_actions,
        gamma=1.0,
        alpha=0.1,
        epsilon=0.1,
        schedules=Dict[str, Schedule],
        plan_steps=0,
        n_steps=1,
        use_value=False,  # If you want to store and learn V instead of Q.
    ):
        """Hold all agent parameters, learned values, and ephemeral data structures."""
        assert not (n_steps > 1 and plan_steps > 0) # Cannot currently handle both
        
        self.num_states = num_states
        self.num_actions = num_actions

        # Hyperparameters
        self.gamma = gamma
        self.alpha_init = alpha
        self.epsilon_init = epsilon

        # Schedules
        self.schedules = schedules

        # After each episode, we can call update_params to refresh alpha and epsilon
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode_count = 0  # track which episode we are on

        # Q or V
        self.use_value = use_value
        if use_value:
            self.V = np.zeros(num_states)
            self.Q = None
        else:
            self.Q = np.zeros((num_states, num_actions))
            self.V = None

        # Dyna related
        self.plan_steps = plan_steps
        self.model = dict()  # For storing transitions (s, a) -> (r, s')
        self.rng = np.random.default_rng()

        # N-step related
        self.n_steps = n_steps
        self.n_states = None  # will be allocated at start of episode
        self.n_actions = None
        self.n_rewards = None

########################################
# Update agent parameters (schedules)
########################################

def update_params(agent_state: AgentState, total_episodes: int):
    ep = agent_state.episode_count
    for param, schedule in agent_state.schedules.items():
        weight = schedule.weight(min(1, ep / total_episodes))
        new_value = (1 - weight) * schedule.initial + weight * schedule.final
        agent_state.__dict__[param] = new_value

########################################
# Action selection
########################################

def select_action(agent_state: AgentState, obs, greedy=False):
    if agent_state.use_value:
        raise NotImplementedError
    # For Q-based
    if greedy:
        return np.argmax(agent_state.Q[obs])
    if np.random.rand() < agent_state.epsilon:
        return np.random.randint(agent_state.num_actions)
    else:
        return np.argmax(agent_state.Q[obs])

########################################
# Single-step update methods
########################################

def q_learning_update(agent_state: AgentState, _t, _T, s, a, r, s_next, _a_next=None):
    Q = agent_state.Q
    alpha = agent_state.alpha
    gamma = agent_state.gamma

    td_target = r + gamma * np.max(Q[s_next])
    Q[s, a] += alpha * (td_target - Q[s, a])


def sarsa_update_fn(expected=False):
    def sarsa_update(agent_state: AgentState, _t, _T, s, a, r, s_next, a_next=None):
        assert expected or a_next is not None
        Q = agent_state.Q
        alpha = agent_state.alpha
        gamma = agent_state.gamma
        eps = agent_state.epsilon
    
        if expected:
            # Expected value under epsilon-greedy
            policy_probs = np.ones(agent_state.num_actions) * (eps / agent_state.num_actions)
            best_a = np.argmax(Q[s_next])
            policy_probs[best_a] += 1 - eps
        
            expected_val = np.sum(Q[s_next] * policy_probs)
            td_target = r + gamma * expected_val
        else:
            td_target = r + gamma * Q[s_next, a_next]
        Q[s, a] += alpha * (td_target - Q[s, a])
    return sarsa_update

########################################
# N-step updates
########################################

def init_nstep_buffers(agent_state: AgentState):
    n = agent_state.n_steps
    agent_state.n_states = np.zeros(n + 1, dtype=int)
    agent_state.n_actions = np.zeros(n + 1, dtype=int)
    agent_state.n_rewards = np.zeros(n + 1, dtype=float)


def nstep_sarsa_update_fn(expected=False):
    def nstep_sarsa_update(agent_state: AgentState, t, T, _s, _a, _r, s_next=None, a_next=None):
        """Implements the core n-step Sarsa update logic."""
        n = agent_state.n_steps
        states = agent_state.n_states
        actions = agent_state.n_actions
        rewards = agent_state.n_rewards
        Q = agent_state.Q
    
        alpha = agent_state.alpha
        gamma = agent_state.gamma
        eps = agent_state.epsilon
    
        first_tau = t + 1 - n  # index of the obs to update
        if first_tau >= 0:
            # In case the episode has ended, perform the trailing updates
            if t + 1 == T:
                range_limit = T
            else:
                range_limit = first_tau + 1
            for tau in range(first_tau, range_limit):
                # compute the return
                G = 0
                # sum over rewards
                for k in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (k - (tau + 1))) * rewards[k % (n+1)]
                # if not terminal, add bootstrap
                if tau + n < T:
                    if expected:
                        # Expected value under epsilon-greedy
                        policy_probs = np.ones(agent_state.num_actions) * (eps / agent_state.num_actions)
                        best_a = np.argmax(Q[s_next])
                        policy_probs[best_a] += 1 - eps
                    
                        expected_val = np.sum(Q[s_next] * policy_probs)
                        G += (gamma ** n) * expected_val
                    else:
                        G += (gamma ** n) * Q[s_next, a_next]
                # update Q
                s_tau = states[tau % (n+1)]
                a_tau = actions[tau % (n+1)]
                Q[s_tau, a_tau] += alpha * (G - Q[s_tau, a_tau])
    return nstep_sarsa_update

########################################
# Dyna: planning
########################################

def dyna_plan(agent_state: AgentState, t, T, update_fn):
    (s, a) = agent_state.rng.choice(list(agent_state.model.keys()))
    (r, s_next) = agent_state.model[(s, a)]
    update_fn(agent_state, t, T, s, a, r, s_next, None)

########################################
# Main loop: train & train_episode
########################################

def train(
    agent_state: AgentState,
    environment,
    num_episodes=1000,
    update_fn=q_learning_update,  # or "expected_sarsa", "dyna_q", "dyna_expected", "nstep_sarsa", etc.
    quiet=False,
    early_stop=-1,
):
    """Trains the agent for num_episodes, returning a list of episode lengths."""
    episode_lengths = []

    try:
        for ep in tqdm(
            range(agent_state.episode_count, num_episodes),
            disable=quiet,
            desc="Episodes",
            initial=agent_state.episode_count,
            total=num_episodes,
        ):
            if ep == early_stop:
                break
            agent_state.episode_count = ep
            update_params(agent_state, num_episodes)
    
            length = train_episode(agent_state, environment, update_fn)
            episode_lengths.append(length)
    except KeyboardInterrupt:
        print(f"Training paused after {ep} episodes!")
    
    return episode_lengths


def train_episode(agent_state: AgentState, environment, update_fn):
    s = environment.reset()
    a = select_action(agent_state, s)

    if agent_state.n_steps > 1:
        init_nstep_buffers(agent_state)
    T = inf

    for t in count():
        s_next, r, done = environment.step(a)
        if agent_state.n_steps > 1:
            idx = t % (agent_state.n_steps + 1)
            agent_state.n_states[idx] = s
            agent_state.n_actions[idx] = a
            agent_state.n_rewards[idx] = r

        if done:
            T = t + 1
        else:
            a_next = select_action(agent_state, s_next)

        # Single-step or multi-step updates
        update_fn(agent_state, t, T, s, a, r, s_next, a_next)
        if agent_state.plan_steps > 0:
            agent_state.model[(s, a)] = (r, s_next)
            for _ in range(agent_state.plan_steps):
                dyna_plan(agent_state, t, T, update_fn)

        if done:
            break
        s, a = s_next, a_next
    return t

########################################
# Policy Execution (play episode greedily)
########################################

def play_episode(agent_state: AgentState, environment, max_steps=-1):
    s = environment.reset()
    a = select_action(agent_state, s, greedy=True)
    states = [s]
    actions = [a]
    rewards = [0]

    step_count = 0
    while True:
        s_next, r, done = environment.step(a)
        step_count += 1
        a_next = select_action(agent_state, s_next, greedy=True)
        states.append(s_next)
        actions.append(a_next)
        rewards.append(r)

        if done or (max_steps > 0 and step_count >= max_steps):
            break

        s, a = s_next, a_next

    return states, actions, rewards
