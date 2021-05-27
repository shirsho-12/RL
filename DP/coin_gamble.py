# Solving the coin flipping gambling problem with head probability = 0.4 and goal = $100 using
# value iteration

import torch
import matplotlib.pyplot as plt


def value_iteration(env, gamma, threshold):
    num_states = env["num_states"]
    head_prob = env["head_prob"]
    max_capital = env["max_capital"]
    rewards = env["rewards"]

    tail_prob = 1 - head_prob
    V = torch.zeros(num_states)
    max_delta = 1 + threshold
    while max_delta > threshold:
        temp = torch.zeros(num_states)
        # Game ends if the gambler runs out of money or reaches max_capital
        for state in range(1, max_capital):
            num_actions = min(state, max_capital - state) + 1
            actions = torch.zeros(num_actions)
            for action in range(1, num_actions):
                actions[action] += head_prob * (rewards[state + action] + gamma * V[state + action])+                                    tail_prob * (rewards[state - action] + gamma * V[state - action])
            temp[state] = torch.max(actions)
            
        max_delta = torch.max(torch.abs(V - temp))
        V = temp.clone()
    return V


def extract_optimal_policy(env, V, gamma):
    num_states = env["num_states"]
    head_prob = env["head_prob"]
    max_capital = env["max_capital"]
    rewards = env["rewards"]

    tail_prob = 1 - head_prob
    policy = torch.zeros(num_states)

    for state in range(num_states):
        num_actions = min(state, max_capital - state) + 1
        actions = torch.zeros(num_actions)
        for action in range(1, num_actions):
            actions[action] += head_prob * (rewards[state + action] + gamma * V[state + action]) +                                    tail_prob * (rewards[state - action] + gamma * V[state - action])
        policy[state] = torch.argmax(actions)
    return policy


if __name__ == "__main__":
    max_capital = 100            # $100
    num_states = max_capital + 1  # {0, 1, ..., 100}
    gamma = 1                    # Undiscounted process, all previous actions matter
    threshold = 0.0001 
    rewards = torch.zeros(num_states)
    rewards[-1] = 1              # Reward 1 if $100 is reached

    head_prob = 0.4              # Probability of the coin landing on heads

    env = {"max_capital": max_capital, "head_prob": head_prob,
       "rewards": rewards, "num_states": num_states}
    optimal_values = value_iteration(env, gamma, threshold)
    optimal_policy = extract_optimal_policy(env, optimal_values, gamma)
    print("Optimal values:\n",  optimal_values)
    print("Optimal policy", optimal_policy)
    plt.plot(optimal_policy[:100].numpy())
    plt.title("Optimal policy values (i.e. bets)")
    plt.xlabel("Capital")
    plt.ylabel("Policy value/Bet")
    plt.show