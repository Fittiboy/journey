from tqdm import tqdm


class Sarsa:
    def __init__(self, epsilon, alpha, grid_world):
        self.alpha = alpha
        self.epsilon = epsilon
        self.grid_world = grid_world
        self.num_states = grid_world.num_states
        self.num_actions = grid_world.num_actions
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.ep_lengths = []

    def select_action(self, state):
        action = np.argmax(self.Q[state])
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        return action

    def train(self, num_episodes=8_000):
        for _ in tqdm(range(num_episodes)):
            state = self.grid_world.reset()
            action = self.select_action(state)
            done = False
            ep_length = 0
            while not done:
                ep_length += 1
                new_state, reward, done = self.grid_world.step(action)
                new_action = self.select_action(new_state)
                if done:
                    self.Q[state, action] += self.alpha * (reward - self.Q[state, action])
                else:
                    self.Q[state, action] += self.alpha * (
                        reward + self.Q[new_state, new_action] - self.Q[state, action]
                    )
                state, action = new_state, new_action
            self.ep_lengths.append(ep_length)

    def plot_ep_lengths(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.ep_lengths)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps to solve")
        plt.show()

    def play_episode(self):
        state = self.grid_world.reset()
        done = False
        episode = []
        while not done:
            episode.append(state)
            action = self.select_action(state)
            state, _, done = self.grid_world.step(action)

        return episode