import numpy as np


class PolicyEvaluation:
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
        num_actions = transitions.shape[-1]
        num_states = transitions.shape[0]
        
        assert transitions.shape == rewards.shape
        assert discount >= 0 and discount <= 1
        assert policy.shape == (num_states, num_actions)
        
        self.discount = discount
        # Reshaping the policy allows us to broadcast it across `transitions`, to
        # calculate expected transitions under the policy.
        self.policy = policy.reshape(num_states, 1, num_actions)

        # The shape is chosen to make it a column vector, allowing for proper
        # calculations later on.
        self.values = np.zeros((num_states, 1))
        # Given that the values in `transitions` are the the probabilities
        # p(s'|s,a), multiplying each element by the policy, broadcasting
        # across all s', gives us p(s'|s,a)*pi(s|a)=p(s',a|s).
        self.policy_transitions = transitions * self.policy
        # Here, we simply calculate the sum over all s' and a of r(s',s,a)*p(s',a|s),
        # giving us the expected rewards received under the policy.
        # We discard the action dimension, but keep the "to states" dimension, giving
        # us the shape (num_states, 1), to match self.value's shape.
        self.exp_rewards = np.sum(
            np.sum(self.policy_transitions * rewards, axis=2),
            axis=1,
            keepdims=True)
        # After using self.policy_transitions to calculate the expected rewards, we no
        # longer need the action dimension, so we sum over it to receive the expected
        # transitions p(s'|s)
        self.exp_transitions = np.sum(self.policy_transitions, axis=2)

    def evaluate(self, min_delta=0.01):
        """
        Args:
            min_delta (float): Terminate policy evaluation if no state changes its value
                by more than this.
        """
        assert min_delta > 0
        while True:
            v = self.values.copy()
            self.values = self.exp_rewards + self.discount * np.matmul(self.exp_transitions, v)
            deltas = np.abs(v-self.values)
            if np.max(deltas) < min_delta:
                break