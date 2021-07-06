from bandit_env import BanditEnv, bandit_visual
from policy_functions import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

bandit_payout = [0.01, 0.015, 0.03]      # CTR 1% for ad 1, 1.5% for ad 2 and 3% for ad 3
bandit_reward = [1, 1, 1]
bandit_env = BanditEnv(bandit_payout, bandit_reward)

num_episodes = 10000
num_actions = len(bandit_payout)

def ucb_policy_run():
    Q = torch.zeros(num_actions)
    action_count = torch.tensor([0 for i in range(num_actions)])     
    action_reward = [0 for i in range(num_actions)]
    action_avg_reward = [[] for i in range(num_actions)]

    for episode in tqdm(range(num_episodes)):
        action = upper_confidence_bound(Q, action_count, episode)
        reward = bandit_env.step(action)
        action_count[action] += 1
        action_reward[action] += reward
        Q[action] = action_reward[action] / action_count[action]
        # Visualizations
        for a in range(num_actions):
            if action_count[a] > 0:
                action_avg_reward[a].append(action_reward[a] / action_count[a])
            else:
                action_avg_reward[a].append(0)

    print(f"Average reward over {num_episodes} episodes: {sum(action_reward) / num_episodes}")
    bandit_visual(num_actions, action_avg_reward)


def thompson_sampling_run():
    Q = torch.zeros(num_actions)
    action_count = torch.tensor([0 for i in range(num_actions)])     
    action_reward = [0 for i in range(num_actions)]
    action_avg_reward = [[] for i in range(num_actions)]
    alpha = torch.ones(num_actions)
    beta = torch.ones(num_actions)

    for episode in tqdm(range(num_episodes)):
        action = thompson_sampling(alpha, beta)
        reward = bandit_env.step(action)
        action_count[action] += 1
        action_reward[action] += reward
        if reward > 0:
            alpha[action] += 1
        else:
            beta[action] += 1
        # Visualizations
        for a in range(num_actions):
            if action_count[a] > 0:
                action_avg_reward[a].append(action_reward[a] / action_count[a])
            else:
                action_avg_reward[a].append(0)

    print(f"Average reward over {num_episodes} episodes: {sum(action_reward) / num_episodes}")
    bandit_visual(num_actions, action_avg_reward)


def contextual_bandits_run():
    bandit_payout_machines = [[0.01, 0.015, 0.03], [0.025, 0.01, 0.015]]
    bandit_reward_machines = [[1, 1, 1], [1, 1, 1]]

    # List of bandit environments
    num_machines = len(bandit_payout_machines)
    bandit_env_machines = [BanditEnv(payout, reward) for payout, reward in 
                          zip(bandit_payout_machines, bandit_reward_machines)]
    num_episodes = 100000

    # Add another dimension for the additional machines
    num_actions = len(bandit_payout_machines[0])
    action_count = torch.zeros(num_machines, num_actions)      # Convert to Tensor
    action_reward = torch.zeros(num_machines, num_actions)
    action_avg_reward = [[[] for i in range(num_actions)] for machine in range(num_machines)]

    Q_machines = torch.empty(num_machines, num_actions)
    
    for episode in tqdm(range(num_episodes)):
        state = torch.randint(0, num_machines, (1,)).item()
        action = upper_confidence_bound(Q_machines[state], action_count[state], episode)
        reward = bandit_env_machines[state].step(action)
        action_count[state][action] += 1
        action_reward[state][action] += reward
        Q_machines[state][action] = action_reward[state][action] / action_count[state][action]
        for a in range(num_actions):
            if action_count[state][a] > 0:
                action_avg_reward[state][a].append(action_reward[state][a] / action_count[state][a])
            else:
                action_avg_reward[state].append(0)

    for state in range(num_machines):
        for action in range(num_actions):
            plt.plot(action_avg_reward[state][action])
        plt.legend(["Arm {}".format(action) for action in range(num_actions)])
        plt.title(f"Average reward over time for state {state}")
        plt.xscale('log')
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.show()


if __name__ == "__main__":
    ucb_policy_run()
    thompson_sampling_run()
    contextual_bandits_run()