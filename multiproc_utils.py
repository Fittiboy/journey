from multiprocessing import Pool, Pipe
from copy import deepcopy


class TrainPool:
    def __init__(self, trainers: list, num_episodes: int):
        self.trainers = trainers
        self.num_episodes = num_episodes

    def train_process(self, arglist):
        trainer, conn = arglist
        trainer.train(num_episodes=self.num_episodes)
        print("Training complete!")
        agent = trainer.agent
        #Q = agent.Q.copy().tolist()
        ep_lengths = deepcopy(agent.ep_lengths)
        ep_returns = deepcopy(agent.ep_returns)
        conn.send((ep_lengths, ep_returns))
        conn.close()

    def __call__(self):
        parents = []
        children = []
        for _ in range(len(self.trainers)):
            pc, cc = Pipe()
            parents.append(pc)
            children.append(cc)
    
        arglist = [
            (trainer, child) for trainer, child in zip(self.trainers, children)
        ]
        
        with Pool(len(arglist)) as p:
            p.map(self.train_process, arglist)

        agent_updates = []
        for parent in parents:
            print("Receiving!")
            agent_updates.append(parent.recv())

        return agent_updates