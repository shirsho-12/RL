import torch
import gym
from random_search import run_episode, eval_run

def hill_climb(env, num_states, num_actions, num_episodes):
    best_reward = 0
    best_weight = torch.rand(num_states, num_actions)
    total_rewards = []
    eta = 0.01              # Noise scaling parameter
    for episode in range(num_episodes):
        weight = best_weight + torch.rand(num_states, num_actions) * eta         # Change 2
        reward = run_episode(env, weight)
        if episode % 20 == 19 or reward > best_reward:
            print(f"Episode: {episode + 1} \t Total reward: {reward}")
        if reward > best_reward:
            best_reward = reward
            best_weight = weight
        total_rewards.append(reward)
    return best_reward, best_weight, total_rewards

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    num_states = env.observation_space.shape[0]  
    num_actions = env.action_space.n            # (4, 2)
    num_episodes = 1000
    
    best_reward, best_weight, total_rewards = hill_climb(env, num_states, num_actions, num_episodes)
    print(f"Average reward, {num_episodes} iterations: {sum(total_rewards) / num_episodes}")
    eval_run(env, best_weight)
    env.close()