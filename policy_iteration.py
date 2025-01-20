import numpy as np


class PolicyIteration:
    """
    Run iterated policy evaluation on a finite MDP.
    
    The shape of the arrays follows the
    convention of "from->to," meaning that the first axis of length num_states is the
    "from" axis, and the second (if present) is the "to" axis. This means that, for
    example, transitions[12,15,3] is the probability of transitioning from state 12 to
    state 15, given that action 3 was taken.
    """
    def __init__(self, transitions, rewards, discount, policy):
        """
        Args:
            transitions (np.ndarray, shape (num_states, num_states, num_actions)):
                Transition probabilities, where transitions[s,s',a]=p(s'|s,a).
            rewards (np.ndarray, shape(num_states, num_states, num_actions)):
                Rewards received for each transitions, where rewards[s,s',a]=r(s',s,a).
            policy (np.ndarray, shape(num_states, num_actions)):
                Policy pi to be evaluated, where policy[s,a]=pi(a|s).
        """
        self.num_actions = transitions.shape[-1]
        self.num_states = transitions.shape[0]
        
        assert transitions.shape == rewards.shape
        assert discount >= 0 and discount <= 1
        assert policy.shape == (self.num_states, self.num_actions)

        self.transitions = transitions
        self.rewards = rewards
        self.discount = discount
        # Reshaping the policy allows us to broadcast it across `transitions`, to
        # calculate expected transitions under the policy.
        self.policy = policy.reshape(self.num_states, 1, self.num_actions)

        # The shape is chosen to make it a column vector, allowing for proper
        # calculations later on.
        self.values = np.zeros((self.num_states, 1))
        
        self.update_probs()

    def update_probs(self):
        """Update probabilities and expectations under current policy"""
        # Given that the values in `transitions` are the the probabilities
        # p(s'|s,a), multiplying each element by the policy, broadcasting
        # across all s', gives us p(s'|s,a)*pi(s|a)=p(s',a|s).
        self.policy_transitions = self.transitions * self.policy
        # Here, we simply calculate the sum over all s' and a of r(s',s,a)*p(s',a|s),
        # giving us the expected rewards received under the policy.
        # We discard the action dimension, but keep the "to states" dimension, giving
        # us the shape (num_states, 1), to match self.value's shape.
        self.exp_rewards = np.sum(
            np.sum(self.policy_transitions * self.rewards, axis=2),
            axis=1,
            keepdims=True,
        )
        # After using self.policy_transitions to calculate the expected rewards, we no
        # longer need the action dimension, so we sum over it to receive the expected
        # transitions p(s'|s)
        self.exp_transitions = np.sum(self.policy_transitions, axis=2)
        

    def evaluate(self, min_delta=0.01, max_iterations=10_000):
        """
        Args:
            min_delta (float): Terminate policy evaluation if no state changes its value
                by more than this.
        """
        assert min_delta > 0
        for _ in range(max_iterations):
            v = self.values.copy()
            self.values = self.exp_rewards + self.discount * np.matmul(self.exp_transitions, v)
            deltas = np.abs(v-self.values)
            if np.max(deltas) < min_delta:
                break
        else:
            print((
                f"Warning: with min_delta={min_delta}, "
                f"policy evaluation did not converge within max_iterations={max_iterations}!"
            ))

    def improve_policy(self):
        """Make the policy greedy with respect to the current value function estimate"""
        # -----------------------OUTDATED COMMENT, SEE BELOW-----------------------------------
        # As self.values is of shape (state, 1), it would broadcast across self.reward of shape
        # (from_state, to_state, action) with the from_states, while we are looking to sum over
        # the to_states. So we first transpose it to be row vector, and then reshape it to ensure
        # correct broadcasting.
        # ----------------------------UPDATED COMMENT-------------------------------------------
        # After testing, it turns out self.rewards + self.discount * self.values also broadcasts
        # correctly. This is because to broadcast, the dimensions are lined up like this:
        #
        #                        from states  to states    actions
        #                        -------------------------------------
        # self.rewards           num_states   num_states   num_actions
        # self.values                         num_states   1
        #                        -------------------------------------
        # self.rewards + values  num_states   num_states   num_actions
        #
        # Where the values' from_states line up with the rewards' to_states. Neat!
        action_values = self.transitions * (self.rewards + self.discount * self.values)
        action_values = np.sum(action_values, axis=1)
        best_actions = np.argmax(action_values, axis=1)

        # np.identity(n) is the n x n matrix with ones on the diagonal, and zeros
        # everywhere else. So np.identity(4)[2] is [0, 0, 1, 0]. As best_actions is
        # a vector of the indices of the best actions, like [[2], [0], [1], ...]
        # np.identity(self.num_actions)[best_actions] is the greedy policy in the shape
        # [[0, 0, 1, 0],
        #  [1, 0, 0, 0],
        #  [0, 1, 0, 0],
        #  ...         ]
        self.policy = np.identity(self.num_actions)[best_actions].reshape(self.policy.shape)
        self.update_probs()

    def find_optimal_policy(self):
        policy_stable = False
        while not policy_stable:
            old_policy = self.policy.copy()
            self.evaluate()
            self.improve_policy()
            policy_stable = np.all(old_policy == self.policy)