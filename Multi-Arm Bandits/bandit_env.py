import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class BanditEnv():
    def __init__(self, payout_list, reward_list):
        # payout_list: list of probabilities of payouts for each arm
        # reward_list: reward for each arm
        self.payout_list = payout_list
        self.reward_list = reward_list
    
    def step(self, action):
        # Random likelihood of being rewarded
        if torch.rand(1).item() < self.payout_list[action]:
            return self.reward_list[action]
        return 0


def bandit_visual(num_actions, action_avg_reward):
    for action in range(num_actions):
        plt.plot(action_avg_reward[action])
    plt.legend(["Arm {}".format(action) for action in range(num_actions)])
    plt.title("Average reward over time")
    plt.xscale('log')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.show()