from bandit_env import BanditEnv, bandit_visual
from policy_functions import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

bandit_payout = [0.1, 0.15, 0.3]
bandit_reward = [4, 3, 1]
bandit_env = BanditEnv(bandit_payout, bandit_reward)
num_actions = len(bandit_payout)

def random_policy_run():
    num_episodes = 10000
    action_count = [0 for i in range(num_actions)]
    action_reward = [0 for i in range(num_actions)]
    action_avg_reward = [[] for i in range(num_actions)]

    random_policy = get_random_policy(num_actions)
    for episode in range(num_episodes):
        action = random_policy()
        reward = bandit_env.step(action)
        # Visualizations
        action_count[action] += 1
        action_reward[action] += reward
        for a in range(num_actions):
            if action_count[a] > 0:
                action_avg_reward[a].append(action_reward[a] / action_count[a])
            else:
                action_avg_reward[a].append(0)

    print(f"Average reward over {num_episodes} episodes: {sum(action_reward) / num_episodes}")
    bandit_visual(num_actions, action_avg_reward)

def policy_run(policy):
    num_episodes = 10000
    action_count = [0 for i in range(num_actions)]
    action_reward = [0 for i in range(num_actions)]
    action_avg_reward = [[] for i in range(num_actions)]
    Q = torch.zeros(num_actions)

    for episode in tqdm(range(num_episodes)):
        action = policy(Q)
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


def ubc_policy_run():
    num_episodes = 10000
    action_count = [0 for i in range(num_actions)]
    action_reward = [0 for i in range(num_actions)]
    action_avg_reward = [[] for i in range(num_actions)]
    Q = torch.zeros(num_actions)

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

if __name__ == "__main__":
    random_policy_run()

    epsilon = 0.2
    eps_policy = gen_eps_greedy_policy(num_actions, epsilon)
    policy_run(eps_policy)

    tau = 0.1
    softmax_policy = gen_softmax_exploration_policy(tau)
    policy_run(softmax_policy)