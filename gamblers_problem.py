import numpy as np
import matplotlib.pyplot as plt

from value_iteration import ValueIteration


class GamblersProblem:
    def __init__(self, p_h=0.4):
        """
        Args:
            p_h (float): Probability that the coin comes up heads, in which case
            the gambler wins.
        """
        # Gambler's Problem
        goal = 100
        # The state space includes the capital, from 1-99, as well as the terminal
        # loss and win states of 0 and 100.
        num_states = goal + 1
        # Bets range from 0 to 50. The gambler can bet more than 50 only if he has
        # has that much. But in that case, he needs less than 50 to win, so it is
        # pointless to consider those actions.
        num_actions = (goal // 2) + 1
        self.transitions = np.zeros((num_states, num_states, num_actions), dtype=float)
        self.rewards = np.zeros_like(self.transitions, dtype=float)
        # Reaching the goal state from any state, using any action, gives a reward
        # of 1. We later give a zero probability for reaching the goal state when
        # the best is greater than the available capital, eliminating impossible
        # strategies, like betting 85 when the capital is 15.
        win_reward = 1
        loss_penalty = 0
        self.rewards[:,goal,:] = win_reward
        self.rewards[goal,goal,:] = 0
        self.rewards[:,0,:] = loss_penalty
        self.rewards[0,0,:] = 0
        
        capitals = np.arange(num_states).reshape(num_states, 1, 1)
        bets = np.arange(num_actions).reshape(1, 1, num_actions)
        targets = np.swapaxes(capitals, 0, 1)
        legal_losses = np.asarray(targets - (capitals - bets) == 0)
        legal_wins = np.asarray(targets - (capitals + bets) == 0)
        self.transitions = p_h * legal_wins + (1 - p_h) * legal_losses
        # Transitions should not be possible when trying to bet more
        # than the available capital.
        debt = capitals - bets < 0
        self.transitions *= ~debt
        # Betting more than available will simply lead to 0 capital.
        diagonal = np.eye(num_states, num_actions)
        summed_diagonal = debt[:,0,:]
        self.transitions[:,0,:] += summed_diagonal
        # The terminal states can only transition to themselves.
        self.transitions[0,:,:] = 0
        self.transitions[goal,:,:] = 0
        self.transitions[0,0,:] = 1.0
        self.transitions[goal,goal,:] = 1.0

        assert np.all(self.rewards[:,1:goal,:] == 0)
        assert np.all(self.rewards[0,0,:] == 0)
        assert np.all(self.rewards[1:,0,:] == loss_penalty)
        assert np.all(self.rewards[goal,goal,:] == 0)
        assert np.all(self.rewards[:goal,goal,:] == win_reward)
        # Betting zero is boring!
        self.rewards -= np.identity(num_states).reshape(num_states, num_states, 1)
        self.rewards[0,0,:] = 0
        self.rewards[100,100,:] = 0
        
        assert self.transitions[15,30,15] == p_h
        assert self.transitions[15,0,15] == 1 - p_h
        assert self.transitions[15,31,15] == 0.0
        assert self.transitions[15,1,15] == 0.0
        assert self.transitions[15,65,50] == 0.0
        assert self.transitions[15,0,50] == 1.0
        assert self.transitions[50,0,50] == 1 - p_h
        assert self.transitions[50,100,50] == p_h
        assert np.all(self.transitions[goal,:goal,:] == 0)
        assert np.all(self.transitions[goal,goal,:] == 1.0)
        assert np.all(self.transitions[0,1:,:] == 0)
        assert np.all(self.transitions[0,0,:] == 1.0)


def run_gamblers_problem(p_h=0.4):
    gp = GamblersProblem(p_h=p_h)
    iterator = ValueIteration(gp.transitions, gp.rewards)
    iterator.find_optimal_policy()
    return iterator


def plot_gamblers_problem(iterator, p_h=0.4, goal=100):
    final_policy = np.argmax(iterator.policy, axis=2)[1:goal]
    capital = np.arange(1, goal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r"Gambler's Problem, $p_h=" + str(p_h) + r"$")
    ax1.plot(capital, final_policy)
    ax1.set_xticks([1, 25, 50, 75, 100])
    ax1.set_xlabel("Capital")
    ax1.set_ylabel("Final policy (stake)")
    
    history = iterator.value_history
    hist_len = len(history)
    for i in range(5):
        ax2.plot(capital, history[i][1:goal], label=f"Sweep {i+1}")
    ax2.plot(capital, iterator.values[1:goal], label=f"Sweep {hist_len}, final")
    ax2.legend()
    ax2.set_xticks([1, 25, 50, 75, 100])
    ax2.set_xlabel("Capital")
    ax2.set_ylabel("Final value estimate")
    plt.show()