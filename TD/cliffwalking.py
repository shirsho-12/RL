# If imports do not work, add a TD. to each of the files

import gym
from td_algos import q_learning
from helper import gen_eps_greedy_policy, plot_rate
env = gym.make("CliffWalking-v0")
num_states, num_actions = env.observation_space.n, env.action_space.n
print(num_states, num_actions)

gamma = 1
num_episodes = 500
alpha = 0.4
epsilon = 0.1

behaviour_policy = gen_eps_greedy_policy(env.action_space.n, epsilon)
optimal_Q, optimal_policy, info = q_learning(env, behaviour_policy, gamma, num_episodes, alpha)
print("\nQ-Learning Optimal policy: \n", optimal_policy)
plot_rate(info["length"], info["rewards"], "Cliff Walking: Q-Learning")
