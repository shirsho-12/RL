# Monte Carlo Policy Gradient

import torch
import gym
from random_search import eval_run
import torch.nn as nn


def run_grad_episode(env, weight):
    state = env.reset()
    grads = []
    total_reward = 0
    is_done = False
    softmax = nn.Softmax(dim=0)
    while not is_done:
        state = torch.from_numpy(state).float()
        """ Gradient Update """
        z = torch.matmul(state, weight)
        proba = softmax(z)
        # print(z, proba)
        # action = 
        # is_done = True
        action = int(torch.bernoulli(proba[1]).item())       
        """ probability -> binary selection (Monte Carlo) """
        d_softmax = torch.diag(proba) - proba.view(-1, 1) * proba
        d_log = d_softmax[action] / proba[action]
        grad = state.view(-1, 1) * d_log
        grads.append(grad)

        state, reward, is_done, info = env.step(action)
        total_reward += reward
    return total_reward, grads


def mc_policy_gradients(env, dims, num_episodes, eta=0.01):
    weight = torch.rand(dims)
    total_rewards = []

    for episode in range(num_episodes):
        reward, grads = run_grad_episode(env, weight)
        for i, grad in enumerate(grads):
            weight += eta * grad * (reward - i)
        if episode % 20 == 19:
            print(f"Episode: {episode + 1} \t Total reward: {reward}")
        total_rewards.append(reward)
    
    return weight, total_rewards

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    num_states = env.observation_space.shape[0]  
    num_actions = env.action_space.n            # (4, 2)
    num_episodes = 1000
    
    
    best_weight, total_rewards = mc_policy_gradients(env, (num_states, num_actions), num_episodes)
    print(f"Average reward, {num_episodes} iterations: {sum(total_rewards) / num_episodes}")
    eval_run(env, best_weight)
    env.close()