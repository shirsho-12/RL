import torch
from tqdm import tqdm
from collections import defaultdict


def q_learning(env, behaviour_policy, gamma, num_episodes, alpha):
    episode_length = [0] * num_episodes
    total_reward_episode = [0] * num_episodes
    num_actions = env.action_space.n

    Q = defaultdict(lambda: torch.zeros(num_actions))
    for episode in tqdm(range(num_episodes)):
        # if (episode + 1) % 400 == 0:
        #     alpha /= 2
        state = env.reset()
        is_done = False
        while not is_done:
            action = behaviour_policy(state, Q)
            next_state, reward, is_done, info = env.step(action)
            # Time step update
            del_td = reward + gamma * torch.max(Q[next_state]) - Q[state][action]   
            Q[state][action] += alpha * del_td
            state = next_state
            total_reward_episode[episode] += reward
            episode_length[episode] += 1
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy, {"rewards": total_reward_episode, "length": episode_length}


def sarsa(env, gamma, num_episodes, alpha, eps_policy):
    episode_length = [0] * num_episodes
    total_reward_episode = [0] * num_episodes
    num_actions = env.action_space.n

    Q = defaultdict(lambda: torch.zeros(num_actions))
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        is_done = False
        action = eps_policy(state, Q)
        while not is_done:
            next_state, reward, is_done, info = env.step(action)
            next_action = eps_policy(next_state, Q)
            del_td = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * del_td
            # For plotting
            episode_length[episode] += 1
            total_reward_episode[episode] += reward
            state = next_state
            action = next_action
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy, {"rewards": total_reward_episode, "length": episode_length}


def double_q_learning(env, gamma, num_episodes, alpha, eps_policy):
    episode_length = [0] * num_episodes
    total_reward_episode = [0] * num_episodes
    num_actions, num_states = env.action_space.n, env.observation_space.n
    # Create defaultdict if number of states unknown
    # Easier to create tensors here instead of writing a way to add defaultdicts
    Q1, Q2 = torch.zeros(num_states, num_actions), torch.zeros(num_states, num_actions)
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        is_done = False
        while not is_done:
            action = eps_policy(state, Q1 + Q2)
            next_state, reward, is_done, info = env.step(action)
            if torch.rand(1).item() < 0.5:             # update 1 randomly
                best_next_action = torch.argmax(Q1[next_state])
                del_td = reward + gamma * Q2[next_state][best_next_action] - Q1[state][action]
                Q1[state][action] += del_td * alpha
            else:
                best_next_action = torch.argmax(Q2[next_state])
                del_td = reward + gamma * Q1[next_state][best_next_action] - Q2[state][action]
                Q2[state][action] += del_td * alpha
            state = next_state
            
            episode_length[episode] += 1
            total_reward_episode[episode] += reward
    policy = {}
    Q = Q1 + Q2            # Combine both policies into 1
    for state in range(num_states):
        policy[state] = torch.argmax(Q[state]).item()
    return Q, policy, {"rewards": total_reward_episode, "length": episode_length}