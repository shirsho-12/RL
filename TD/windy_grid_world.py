# If imports do not work, add a TD. to each of the files

from WindyGridWorld import WindyGridWorld
from helper import gen_eps_greedy_policy, plot_rate
from td_algos import sarsa

env = WindyGridWorld()
num_episodes = 500

gamma = 1
alpha = 0.4
epsilon = 0.1
eps_policy = gen_eps_greedy_policy(env.action_space.n, epsilon)

optimal_Q, optimal_policy, info = sarsa(env, gamma, num_episodes, alpha, eps_policy)
print("\nSARSA Optimal policy: \n", optimal_policy)
plot_rate(info["length"], info["rewards"], "Windy GridWorld: SARSA")
