import torch
import gym


def run_episode(env, weight):
    state = env.reset()
    total_reward = 0
    is_done = False
    # _ = True
    while not is_done:
        state = torch.from_numpy(state).float()                         # For matrix multiplication
        action = torch.argmax(torch.matmul(state, weight))            
        # Index with highest value from 1x2 matrix is selected
        state, reward, is_done, info = env.step(action.item())
        # if _:
        #     print(state, "\n", action, "\n", torch.matmul(torch.from_numpy(state).float(), weight))
        #     _ = False
        total_reward += reward
    return total_reward


def eval_run(env, best_weight):
    eval_num_episodes = 100
    eval_total_rewards = []
    for episode in range(eval_num_episodes):
        reward = run_episode(env, best_weight)
        eval_total_rewards.append(reward)  
    print(f"Average reward, {eval_num_episodes} evaluation iterations:", 
    f"{sum(eval_total_rewards) / eval_num_episodes}")


def random_search(env, num_states, num_actions, num_episodes):
    best_reward = 0
    best_weight = None
    total_rewards = []
    for episode in range(num_episodes):
        weight = torch.rand(num_states, num_actions)
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
    
    best_reward, best_weight, total_rewards = random_search(env, num_states, num_actions, num_episodes)
    print(f"Average reward, {num_episodes} iterations: {sum(total_rewards) / num_episodes}")
    eval_run(env, best_weight)
    env.close()
