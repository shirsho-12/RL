from helper import gen_eps_greedy_policy, plot_rate
from td_algos import q_learning, sarsa, double_q_learning

import gym

import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
# print(sys.path)
from text_recorder import record


env = gym.make("Taxi-v3")
num_actions, num_states = env.action_space.n, env.observation_space.n
print(num_actions, num_states)

num_episodes = 1000 
gamma = 1
alpha = 0.4
epsilon = 0.1
behaviour_policy = gen_eps_greedy_policy(env.action_space.n, epsilon)

optimal_Q, optimal_policy, info = q_learning(env, behaviour_policy, gamma, num_episodes, alpha)

record(env, optimal_policy, './video/taxi.json')
# # print("\nQ-Learning Optimal policy: \n", optimal_policy)
# plot_rate(info["length"], info["rewards"], "Taxi Service: Q-Learning")

# num_episodes = 1000 
# gamma = 1
# alpha = 0.6
# epsilon = 0.01        # Grid Search parameters
# eps_policy = gen_eps_greedy_policy(env.action_space.n, epsilon)
# optimal_Q, optimal_policy, info = sarsa(env, gamma, num_episodes, alpha, eps_policy)
# # print("\nSARSA Optimal policy: \n", optimal_policy)
# plot_rate(info["length"], info["rewards"], "Taxi Service: SARSA")

# num_episodes = 3000
# gamma = 1
# alpha = 0.4
# epsilon = 0.1       
# eps_policy = gen_eps_greedy_policy(env.action_space.n, epsilon)

# optimal_Q, optimal_policy, info = double_q_learning(env, gamma, num_episodes, alpha, eps_policy)
# # print("\nDouble Q-Learning Optimal policy: \n", optimal_policy)
# plot_rate(info["length"], info["rewards"], "Taxi Service: Double Q-Learning")