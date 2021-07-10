import gym
from visual import reward_visual
from sarsa_fa import sarsa
from q_learning_fa import q_learning, exp_q_learning
from estimator import QEstimator, NNEstimator

env = gym.make("MountainCar-v0")
num_actions = env.action_space.n
print(num_actions)
env.reset()

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
num_features = 200
lr = 0.03
estimator = QEstimator(num_features, num_states, num_actions, lr)

num_episodes = 300
total_reward = q_learning(env, estimator, num_episodes)

reward_visual(total_reward, "Q-learning FA reward over time")

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
num_features = 200
lr = 0.03
estimator = QEstimator(num_features, num_states, num_actions, lr)

num_episodes = 300
total_reward = sarsa(env, estimator, num_episodes)

reward_visual(total_reward, "Sarsa FA reward over time")

num_episodes = 1000    # More episodes because learning per episode is reduced
replay_size = 200

total_reward = exp_q_learning(env, estimator, num_episodes, replay_size)

reward_visual(total_reward, "Exp-Q-learning FA reward over time" )        # Results are more stable and better

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
num_features = 200
num_hidden = 50   # Hidden Layer Size
lr = 0.001        # Slower learning rate

estimator = NNEstimator(num_features, num_states, num_actions, num_hidden, lr)

num_episodes = 1000
replay_size = 200

total_reward = exp_q_learning(env, estimator, num_episodes, replay_size, memo_size=300)

reward_visual(total_reward, "Exp-Q-learning Neural FA reward over time")