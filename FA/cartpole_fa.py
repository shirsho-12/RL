import gym
from estimator import NNEstimator
from q_learning_fa import q_learning
from visual import reward_visual

env = gym.make("CartPole-v0")

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
num_features = 400
num_hidden = 100
lr = 0.01
estimator = NNEstimator(num_features, num_states, num_actions, num_hidden, lr)
# estimator = QEstimator(num_features, num_states, num_actions, lr)

num_episodes = 1000
total_reward = q_learning(env, estimator, num_episodes)

reward_visual(total_reward)
