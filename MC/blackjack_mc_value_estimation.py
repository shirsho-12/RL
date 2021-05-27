import torch
import gym
from collections import defaultdict
from tqdm import tqdm
from pprint import pprint
from blackjack_helper import plot_blackjack_values

def run_episode(env, hold_score):
    state = env.reset()
    rewards, states, is_done = [], [state], False
    while not is_done:
        action = 0 if state[0] >= hold_score else 1
        state, reward, is_done, info = env.step(action)
        states.append(state)
        rewards.append(reward)
    return states, rewards


def mc_prediction_first_visit(env, hold_score, gamma, num_episodes):
    V, N = defaultdict(float), defaultdict(int)
    for episode in range(num_episodes):
        states, rewards = run_episode(env, hold_score)
        return_prob = 0
        G = {}
        for state, reward in zip(states[1::-1], rewards[::-1]):
            return_prob = gamma * return_prob + reward
            G[state] = return_prob
        for state, return_prob in G.items():
            if state[0] <= 21:
                V[state] += return_prob
                N[state] += 1
    for state in V:
        V[state] /= N[state]
    return V


if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    for hold_score in [15, 18, 20]:
        gamma = 1
        num_episodes = 500000
        values = mc_prediction_first_visit(env, hold_score, gamma, num_episodes)
        print("First visit MC calculated values:")
        pprint(dict(values))
        plot_blackjack_values(values)
