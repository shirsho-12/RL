from tqdm import tqdm
import torch
from collections import deque
import random 


# Epsilon Greedy Policy for Q-Learning

def gen_eps_greedy_policy(estimator, epsilon, num_actions):
    def policy(state):
        probs = torch.ones(num_actions) * epsilon / num_actions
        q_values = estimator.predict(state)
        best_action = torch.argmax(q_values).item()
        probs[best_action] += 1 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy



def q_learning(env, estimator, num_episodes, gamma = 1.0, epsilon = 0.1, eps_decay = 0.99):
    total_reward_episode = [0] * num_episodes # For graphing

    for episode in tqdm(range(num_episodes)):
        policy = gen_eps_greedy_policy(estimator, epsilon * eps_decay ** episode, env.action_space.n)
        state = env.reset()
        is_done = False
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            next_q_values = estimator.predict(next_state)
            td_target = reward + gamma * torch.max(next_q_values)
            estimator.update(state, action, td_target)
            state = next_state
            total_reward_episode[episode] += reward
    return total_reward_episode



def exp_q_learning(env, estimator, num_episodes, replay_size, gamma=1.0, epsilon=0.1, eps_decay=0.99,
                    memo_size=400):
    memo = deque(maxlen=memo_size)
    total_reward_episode = [0] * num_episodes # For graphing

    for episode in tqdm(range(num_episodes)):
        policy = gen_eps_greedy_policy(estimator, epsilon * eps_decay ** episode, env.action_space.n)
        state = env.reset()
        is_done = False
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            next_q_values = estimator.predict(next_state)
            td_target = reward + gamma * torch.max(next_q_values)
            # Storing experience
            memo.append((state, action, td_target))
            state = next_state
            total_reward_episode[episode] += reward

        replay_data = random.sample(memo, min(replay_size, len(memo)))
        for state, action, td_target in replay_data:
            estimator.update(state, action, td_target)

    return total_reward_episode